import json
import os
import copy
import sys
from collections import deque
from typing import Optional
from dataclasses import dataclass

import numpy as np
import torch
import tyro
import utils3d
import gymnasium as gym
from xvfbwrapper import Xvfb

from dataset_toolkits.util import seed_everything
from dataset_toolkits.trellis_random_utils import sphere_hammersley_sequence
from craftium.wrappers import NueToEnuVoxelObs, enu_to_nue

EMPTY_SPACE_NODE_IDS = [126, 127]
ENV_ID_2_CAM_OFFSET = {
    "OpenWorldDataset-v0": (0, 0, 1.6), # mineclone based envs
    "TestRoom-v0": (0, 0, 1.47), # minetest based envs
}

@dataclass
class Args:
    dataset_meta_only: bool = False
    """If True, only the dataset metadata will be generated. Script must be run again with this set to False to start generating the data."""
    dataset_name: str = "testdataset"
    dataset_dir: str = "datasets"
    env_id: str = "OpenWorldDataset-v0"
    async_envs: bool = False
    mt_port: int = 49155
    """TCP port used by Minetest server and client communication. Multiple envs will use successive ports."""
    mt_run_dir: str = "mt_runs"
    """Directory where the Minetest working directories will be created (defaults to the current one)"""
    seed: int = 0
    """Random seed that will be used to generate new downstream seeds for each environment"""
    num_levels: int = 4
    """Number of levels to generate"""
    ep_timesteps: int = 6
    "number of timesteps for each episode (1 episode per level)"
    voxel_obs_rx: int = 32
    "x radius of the voxel observation"
    voxel_obs_ry: int = 32
    "y radius of the voxel observation"
    voxel_obs_rz: int = 32
    "z radius of the voxel observation"
    resolution: int = 518
    "resolution of the generated RGB observations"
    fov: int = 90
    "field of view of the generated RGB observations in degrees"
    actions: str = "noop"
    "actions to take in the environment, supported: noop, mouse_look, mouse_look+move, full, or a comma separated list of action labels"
    init_frames: int = 200
    "number of frames to wait before starting the episode"
    fps_max: int = 24
    "target fps for the environment"
    pmul: Optional[int] = None
    """Physics speed multiplier. Defaults to the default value of CraftiumEnv."""
    rank: int = 0
    """Process rank"""
    world_size: int = 1
    """Total number of processes"""
    num_env_subprocesses: int = 1
    """Number of concurrent env instances per instance of this script"""
    num_levels_per_subprocess: int = 1
    """Number of levels to generate per env instance launched by this script"""
    overwrite_existing: bool = False
    """If True, existing data for a given level seed will be overwritten"""
    debug: bool = False
    """Enable debug mode. Will run the code in a single process."""

def make_env(craftium_kwargs, mt_port_offset):
    craftium_kwargs["voxel_obs_rx"], craftium_kwargs["voxel_obs_ry"], craftium_kwargs["voxel_obs_rz"] = enu_to_nue(
        craftium_kwargs["voxel_obs_rx"], craftium_kwargs["voxel_obs_ry"], craftium_kwargs["voxel_obs_rz"])
    craftium_kwargs["mt_port"] += mt_port_offset
    craftium_kwargs["enable_voxel_obs"] = True
    env = gym.make(**craftium_kwargs)
    env = NueToEnuVoxelObs(env)
    return env

def set_trellis_input_voxel_info(dataset_params, args):
    mt_vox_shape = np.array(dataset_params["minetest_voxel_info"]["dims"][:3])
    output_vox_grid = np.array(dataset_params["trellis_input_voxel_info"]["dims"][:3])
    crop_left = (mt_vox_shape - output_vox_grid) // 2
    crop_right = crop_left + output_vox_grid
    dataset_params["trellis_input_voxel_info"]["active_voxel_preprocessing"] = {
        "empty_space_node_ids": EMPTY_SPACE_NODE_IDS,
        "crop": {"left": crop_left.tolist(), "right": crop_right.tolist()},
        "pad": False,
        "scale": False,
    }
    dataset_params["trellis_input_voxel_info"]["origin_idx"] = (
            np.array(dataset_params["minetest_voxel_info"]["origin_idx"]) - crop_left).tolist()

def compute_intrisincs_extrinsics(level_data, level_meta, dataset_params):
    metadata = list(level_meta.values())
    data = list(level_data.values())
    player_pos = torch.tensor(np.array([(d["player_pos"]) for d in data]), dtype=torch.float32).cuda()
    cam_pos_local = compute_cam_pose_local(player_pos, dataset_params)

    intrinsics = compute_intrinsics(metadata)
    extrinsics_local = compute_extrinsics(cam_pos_local, data)

    for i, s in enumerate(level_data):
        level_data[s]["intrinsics"] = intrinsics[i].cpu().numpy()
        level_data[s]["extrinsics_local"] = extrinsics_local[i].cpu().numpy()
        level_data[s]["extrinsics_global"] = None

def compute_cam_pose_local(player_pos, dataset_params):

    vox_preprocessing = dataset_params["trellis_input_voxel_info"]["active_voxel_preprocessing"]
    assert (vox_preprocessing["pad"] == False and vox_preprocessing["scale"] == False and
            vox_preprocessing["crop"] is not None), \
        "Only cropping is supported when computing intrinsics and extrinsics"
    assert all(vox_preprocessing["crop"]["left"]) == 0, \
        "Only right crop is supported when computing intrinsics and extrinsics"

    cam_pos_local = compute_local_pose_offset(player_pos)
    cam_offset = torch.tensor(dataset_params["rgb_info"]["camera_offset"], dtype=torch.float32).reshape(1, 1,
                                                                                                        3).cuda()
    cam_pos_local += cam_offset #TODO: different cam offset when sneaking
    voxel_grid_size = torch.tensor(dataset_params["trellis_input_voxel_info"]["dims"][:3],
                                   dtype=torch.float32).cuda()
    cam_pos_local = cam_pos_local / voxel_grid_size.reshape(1, 1, 3)

    return cam_pos_local

def compute_local_pose_offset(player_pos):
    pos_local = player_pos.abs() % 1
    pos_local = torch.where(torch.round(pos_local, decimals=6) >= 0.5, -(1 - pos_local), pos_local)

    # numpy version
    # def compute_local_pose_offset(player_pos):
    #     pos_local = np.abs(player_pos) % 1
    #     pos_local = np.where(np.round(pos_local, 6) >= 0.5, -(1 - pos_local), pos_local)
    #     return pos_local

    return pos_local

def compute_intrinsics(metadata):
    fovs_x = torch.deg2rad(torch.stack([torch.tensor(m["fov_x"]) for m in metadata]).cuda())
    fovs_y = torch.deg2rad(torch.tensor([torch.tensor(m["fov_y"]) for m in metadata]).cuda())
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(fovs_x, fovs_y)

    def _for_loop_intrinsics(fovs_x, fovs_y):
        intrinsics = []
        for fov_x, fov_y in zip(fovs_x, fovs_y):
            intrinsics.append(utils3d.torch.intrinsics_from_fov_xy(fov_x, fov_y))
        return torch.stack(intrinsics)

    # intrinsics_test = _for_loop_intrinsics(fovs_x, fovs_y)
    # assert torch.allclose(intrinsics, intrinsics_test) #test passed

    return intrinsics

def compute_extrinsics(cam_pos, data):
    yaws = torch.tensor(np.array([d["player_yaw"] for d in data]), dtype=torch.float32).cuda()
    pitchs = torch.tensor(np.array([d["player_pitch"] for d in data]), dtype=torch.float32).cuda()
    yaws = torch.deg2rad(yaws)
    pitchs = torch.deg2rad(pitchs)
    extrinsics = yaw_pitch_cam_pose_to_extrinsics(yaws, pitchs, cam_pos)

    return extrinsics

def yaw_pitch_cam_pose_to_extrinsics(yaw_rad, pitch_rad, cam_pos):

    look_at_pos = torch.stack([
        torch.sin(yaw_rad) * torch.cos(pitch_rad),
        torch.cos(yaw_rad) * torch.cos(pitch_rad),
        torch.sin(pitch_rad),
    ], -1)
    look_at_pos += cam_pos
    ups = torch.tile(torch.tensor([0, 0, 1], dtype=torch.float32), cam_pos.shape[:-1] + (1,)).to(cam_pos.device)
    extrinsics = utils3d.torch.extrinsics_look_at(cam_pos, look_at_pos, ups)
    return extrinsics

# kept in case needed later
def compute_cam_pose_global(player_pos, dataset_params):
    # cam_offset = torch.tensor(dataset_params["rgb_info"]["camera_offset"], dtype=torch.float32).reshape(1, 1, 3).cuda()
    # cam_pos_global = player_pos + cam_offset
    # origin_xyz = torch.tensor(np.array([torch.tensor(m["spawn_pos"]["player_pos"]) for m in metadata])).cuda()
    # cam_pos_global -= origin_xyz.reshape(-1, 1, 3)
    raise NotImplementedError

def yaw_pitch_controller(current_yaw, current_pitch, target_yaw, target_pitch, action_labels, tol=20):
    pitch_error = target_pitch - current_pitch
    yaw_error = min(target_yaw - current_yaw, (target_yaw - 360) - current_yaw, key=abs)
    if abs(pitch_error) < tol and abs(yaw_error) < tol:
        return action_labels.index("noop"), True
    if abs(pitch_error) > abs(yaw_error):
        act_label = "mouse y+" if pitch_error > 0 else "mouse y-"
    else:
        act_label = "mouse x+" if yaw_error > 0 else "mouse x-"
    return action_labels.index(act_label), False

def generate_pitch_yaw_setpoints(num_setpoints=10):
    offset = (np.random.rand(), np.random.rand())
    # Returns [(yaw_t, pitch_t), ...], yaw in [0, 2*pi], pitch in [-pi/2, pi/2]
    setpoints = np.array([sphere_hammersley_sequence(i, num_setpoints, offset=offset) for i in range(num_setpoints)])
    # convert from rad to deg, yaw setpoints to range [0, 360], pitch setpoints to range [-90, 90]
    setpoints = np.rad2deg(setpoints)
    # shuffle setpoints
    np.random.shuffle(setpoints)
    return setpoints.tolist()

def generate_dataset_meta(args):
    seed_everything(args.seed)

    # dataset metadata
    dataset_params = \
        {
        "info": {
            "name": args.dataset_name,
            "description": "Minetest dataset",
            "script_name": os.path.basename(__file__),
            "script_args": vars(args),
            "version": "0.0.1",
            "minetest_rundir_root": args.mt_run_dir,
            "codebase_commit": {
                "craftium": {"url": "placeholder", "commit": "placeholder"}, #TODO
                "minetest": {"url": "placeholder", "commit": "placeholder"},
                "trellis": {"url": "placeholder", "commit": "placeholder"}
            },
        "agent_info": {
            "action_space": args.actions,
        },
        "model_pipeline_info": {} #TODO
        },
        "minetest_voxel_info": {
            "dims": (2 * args.voxel_obs_rx + 1, 2 * args.voxel_obs_ry + 1, 2 * args.voxel_obs_rz + 1, 3),
            "dtype": str(np.uint32),
            "xyz": "ENU",
            "origin": "agent",
            "origin_idx": (args.voxel_obs_rx, args.voxel_obs_ry, args.voxel_obs_rz),
        },
        "rgb_info": {
            "dims": (args.resolution, args.resolution, 3),
            "dtype": str(np.uint8),
            "camera_offset": ENV_ID_2_CAM_OFFSET[args.env_id],
            "hud": False, # can be set to a dict in the future if enabled
        },
        "trellis_input_voxel_info": {
            "dims": (64, 64, 64, 1024),
            "dtype": str(np.float32),
            "xyz": "ENU",
            "origin": "agent",
            "origin_idx": None,
            "active_voxel_preprocessing": None,
            "extrinsics_key": "extrinsics_local",
        }

    }
    set_trellis_input_voxel_info(dataset_params, args)
    dataset_root = os.path.join(args.dataset_dir, args.dataset_name)
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)
    with open(os.path.join(dataset_root, "dataset_params.json"), "w") as f:
        json.dump(dataset_params, f, indent=4)

    seeds = np.random.randint(0, 2 ** 31, args.num_levels).tolist()
    with open(os.path.join(dataset_root, "level_seeds.txt"), "w") as f:
        for s in seeds:
            f.write(f"{s}\n")

def generate_level_chunk(seeds, args, dataset_params):

    level_meta_template = {
        "seed": None,
        "fov_x": args.fov,
        "fov_y": args.fov,
        "spawn_pos": {"player_pos": None, "player_pitch": None, "player_yaw": None},
        "minetest_conf": None,
    }

    level_data_template = {
        "timestep_craftium": [], # [Nsteps]
        "dt_minetest": [], # [Nsteps] #TODO
        "player_pos": [], # [Nsteps, 3]
        "player_vel": [], # [Nsteps, 3]
        "player_pitch": [], # [Nsteps]
        "player_yaw": [], # [Nsteps]
        "obs_rgb": [], # [Nsteps, C, H, W]
        "obs_voxel_mt": [], # [Nsteps, C, X, Y, Z]
        "action": [], # [Nsteps]
        "reward": [], # [Nsteps]
        "termination_flag": [], # [Nsteps]
        "truncation_flag": [], # [Nsteps]
        "intrinsics": [], # [3, 3]
        "extrinsics_local": [], # [Nsteps, 4, 4]
        "extrinsics_global": [], # [Nsteps, 4, 4]
    }

    if args.actions != "noop" and args.env_id != "OpenWorldDataset-v0":
        print("Only noop actions are supported outside of OpenWorldDataset-v0, switching to noop")
        args.actions = "noop"
    env_action_labels = ["noop"]
    env_action_labels.extend(["forward", "backward", "left", "right", "jump", "sneak",
                           "dig", "place", "slot_1", "slot_2", "slot_3", "slot_4",
                           "slot_5", "mouse x+", "mouse x-", "mouse y+", "mouse y-"])
    if args.actions == "noop":
        action_ids = env_action_labels.index("noop")
    elif args.actions == "mouse_look":
        action_ids = [env_action_labels.index("noop"),
                      env_action_labels.index("mouse x+"), env_action_labels.index("mouse x-"),
                      env_action_labels.index("mouse y+"), env_action_labels.index("mouse y-")]
    elif args.actions == "mouse_look+move":
        action_ids = [env_action_labels.index("noop"),
                      env_action_labels.index("mouse x+"), env_action_labels.index("mouse x-"),
                      env_action_labels.index("mouse y+"), env_action_labels.index("mouse y-"),
                      env_action_labels.index("forward"), env_action_labels.index("backward"),
                      env_action_labels.index("left"), env_action_labels.index("right"),
                      env_action_labels.index("jump"), env_action_labels.index("sneak"),]
    elif args.actions == "full":
        action_ids = list(range(len(env_action_labels)))
    else:
        action_labels = args.actions.split(",")
        action_ids = [env_action_labels.index(a) for a in action_labels]

    seeds = seeds.reshape(-1, args.num_env_subprocesses, args.num_levels_per_subprocess)
    try:
        with ThreadPoolExecutor(max_workers=args.num_env_subprocesses) as executor:
            def env_process_executor(seed_sequence, process_idx):

                seeds_to_gen = []
                if not args.overwrite_existing:
                    for seed in seed_sequence:
                        data_path = os.path.join(args.dataset_dir, args.dataset_name, "raw", args.env_id, str(seed),
                                                 "data.npz")
                        if os.path.exists(data_path):
                            print(f"Level {seed} already exists, skipping")
                        else:
                            seeds_to_gen.append(seed)
                else:
                    seeds_to_gen = seed_sequence

                if not seeds_to_gen:
                    return

                seed_everything(args.seed + args.rank * 1000 + process_idx)
                # env setup
                craftium_kwargs = dict(
                    id=f"Craftium/{args.env_id}",
                    mt_port=args.mt_port,
                    run_dir_prefix=args.mt_run_dir,
                    rgb_observations=True,
                    enable_voxel_obs=True,
                    obs_width=args.resolution,
                    obs_height=args.resolution,
                    voxel_obs_rx=args.voxel_obs_rx,
                    voxel_obs_ry=args.voxel_obs_ry,
                    voxel_obs_rz=args.voxel_obs_rz,
                    init_frames=args.init_frames,
                    fps_max=args.fps_max,
                    pmul=args.pmul,
                    minetest_conf={"fov": args.fov, "world_start_time": np.random.randint(0, 23999)},
                )
                env = make_env(craftium_kwargs, mt_port_offset = args.rank * 1000 + process_idx)
                seeds_to_gen = deque(seeds_to_gen)

                level_meta = {}
                data = {}
                for i in range(args.ep_timesteps * len(seeds_to_gen)):
                    if i % args.ep_timesteps == 0:
                        ts = 0
                        seed = seeds_to_gen.popleft()
                        obs, info = env.reset(
                            seed=seed,
                            options={"minetest_conf": {"world_start_time": np.random.randint(0, 23999)}}
                        )
                        setpoints = deque(generate_pitch_yaw_setpoints(num_setpoints=args.ep_timesteps // 5))
                        yaw_setpoint, pitch_setpoint = setpoints.popleft()
                        level_meta[seed] = copy.deepcopy(level_meta_template)
                        data[seed] = copy.deepcopy(level_data_template)
                        level_meta[seed]["seed"] = seed
                        level_meta[seed]["spawn_pos"] = {"player_pos": info["player_pos"].tolist(),
                                                      "pitch": info["player_pitch"],
                                                      "yaw": info["player_yaw"]}
                        level_meta[seed]["minetest_conf"] = env.unwrapped.get_mt_config()

                    if args.actions == "mouse_look":
                        action, setpoint_reached = yaw_pitch_controller(info["player_yaw"], info["player_pitch"],
                                                                        yaw_setpoint, pitch_setpoint, env_action_labels)
                        if setpoint_reached and len(setpoints) > 0:
                            yaw_setpoint, pitch_setpoint = setpoints.popleft()
                    else:
                        random_sample = np.random.choice(len(action_ids))
                        action = action_ids[random_sample]
                    data[seed]["obs_rgb"].append(obs)
                    data[seed]["obs_voxel_mt"].append(info["voxel_obs"])
                    data[seed]["timestep_craftium"].append(ts)
                    data[seed]["dt_minetest"].append(info["mt_dtime"])
                    data[seed]["action"].append(action)
                    data[seed]["player_pitch"].append(info["player_pitch"])
                    data[seed]["player_yaw"].append(info["player_yaw"])
                    data[seed]["player_pos"].append(info["player_pos"].tolist())
                    data[seed]["player_vel"].append(info["player_vel"].tolist())
                    obs, reward, term, trunc, info = env.step(action)
                    data[seed]["reward"].append(reward)
                    data[seed]["termination_flag"].append(term)
                    data[seed]["truncation_flag"].append(trunc)
                    ts += 1

                env.close()

                data = {s: {k: np.array(v) for k, v in d.items()} for s, d in data.items()}
                compute_intrisincs_extrinsics(data, level_meta, dataset_params)

                # save data according to the following structure:
                #     - raw
                #         - dataset_name (allows for mixing datasets later)
                #             - seed_folder (one per level)
                #                 - level_metadata.json
                #                 - data.npz
                dataset_root = os.path.join(args.dataset_dir, args.dataset_name)
                if not os.path.exists(os.path.join(dataset_root, "raw", args.env_id)):
                    os.makedirs(os.path.join(dataset_root, "raw", args.env_id))
                for s in level_meta:
                    level_folder = os.path.join(dataset_root, "raw", args.env_id, str(s))
                    os.makedirs(level_folder, exist_ok=True)
                    with open(os.path.join(level_folder, "level_metadata.json"), "w") as f:
                        json.dump(level_meta[s], f, indent=4)
                    np.savez_compressed(os.path.join(level_folder, "data.npz"), **data[s])

            for seed_chunk in seeds:
                executor.map(env_process_executor, seed_chunk.tolist(), [p_id for p_id in range(len(seed_chunk))])
    except Exception as e:
        print(f"Error happened during processing. Traceback: {e}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.debug:
        from util import MockThreadPoolExecutor as ThreadPoolExecutor
    else:
        from concurrent.futures import ThreadPoolExecutor

    if args.dataset_meta_only:
        generate_dataset_meta(args)
        sys.exit(0)

    dataset_params = json.load(open(os.path.join(args.dataset_dir, args.dataset_name, "dataset_params.json")))
    level_seeds = np.loadtxt(os.path.join(args.dataset_dir, args.dataset_name, "level_seeds.txt"), dtype=int)
    start = len(level_seeds) * args.rank // args.world_size
    end = len(level_seeds) * (args.rank + 1) // args.world_size
    level_seeds = level_seeds[start:end]

    vdisplay = Xvfb()
    vdisplay.start()
    generate_level_chunk(level_seeds, args, dataset_params)
    vdisplay.stop()