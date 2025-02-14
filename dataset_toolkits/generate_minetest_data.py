import json
import os
import copy
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
from craftium.wrappers import NueToEnuVoxelObs, enu_to_nue

EMPTY_SPACE_NODE_IDS = [126, 127]

@dataclass
class Args:
    dataset_name: str = "testdataset"
    dataset_dir: str = "datasets"
    env_id: str = "OpenWorldDataset-v0"
    num_envs: int = 2
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
    init_frames: int = 200
    "number of frames to wait before starting the episode"
    fps_max: int = 200
    "target fps for the environment"
    pmul: Optional[int] = None
    """Physics speed multiplier. Defaults to the default value of CraftiumEnv."""

    # Runtime args
    num_iters: int = 0

# def make_env(env_id, run_dir_prefix, mt_port, obs_width, obs_height, voxel_obs_rx, voxel_obs_ry, voxel_obs_rz, init_frames,
#              seed, fps_max, pmul, minetest_conf):
def make_env(craftium_kwargs, env_idx, seed):
    def thunk():
        # craftium_kwargs = dict(
        #     run_dir_prefix=run_dir_prefix,
        #     mt_port=mt_port,
        #     rgb_observations=True,
        #     obs_width=obs_width,
        #     obs_height=obs_height,
        #     voxel_obs_rx=voxel_obs_rx,
        #     voxel_obs_ry=voxel_obs_ry,
        #     voxel_obs_rz=voxel_obs_rz,
        #     init_frames=init_frames,
        #     seed=seed,
        #     fps_max=fps_max,
        #     pmul=pmul,
        #     minetest_conf=minetest_conf,
        # )
        craftium_kwargs["voxel_obs_rx"], craftium_kwargs["voxel_obs_ry"], craftium_kwargs["voxel_obs_rz"] = enu_to_nue(
            craftium_kwargs["voxel_obs_rx"], craftium_kwargs["voxel_obs_ry"], craftium_kwargs["voxel_obs_rz"])
        craftium_kwargs["seed"] = seed
        craftium_kwargs["mt_port"] += env_idx
        craftium_kwargs["enable_voxel_obs"] = True
        env = gym.make(**craftium_kwargs)
        env = NueToEnuVoxelObs(env)
        return env

    return thunk

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
    num_levels = len(level_meta)
    metadata = list(level_meta.values())
    data = list(level_data.values())
    pos = torch.tensor(np.array([(d["player_pos"]) for d in data]), dtype=torch.float32).cuda()
    pos += torch.tensor(dataset_params["rgb_info"]["camera_offset"], dtype=torch.float32).reshape(1, 1, 3).cuda()
    origin_xyz = torch.tensor(np.array([torch.tensor(m["spawn_pos"]["player_pos"]) for m in metadata])).cuda()
    pos -= origin_xyz.reshape(num_levels, 1, 3)

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

    yaws = torch.tensor(np.array([d["player_yaw"] for d in data]), dtype=torch.float32).cuda()
    pitchs = torch.tensor(np.array([d["player_pitch"] for d in data]), dtype=torch.float32).cuda()
    extrinsics_global = yaw_pitch_cam_pos_to_extrinsics(yaws, pitchs, pos)
    extrinsics_local = yaw_pitch_cam_pos_to_extrinsics(yaws, pitchs, torch.zeros_like(pos))

    for i, s in enumerate(level_data):
        level_data[s]["intrinsics"] = intrinsics[i].cpu().numpy()
        level_data[s]["extrinsics_local"] = extrinsics_local[i].cpu().numpy()
        level_data[s]["extrinsics_global"] = extrinsics_global[i].cpu().numpy()

def yaw_pitch_cam_pos_to_extrinsics(yaws, pitchs, poses):
    yaws = torch.deg2rad(yaws)
    pitchs = torch.deg2rad(pitchs)
    look_at_poses = torch.stack([
        torch.sin(yaws) * torch.cos(pitchs),
        torch.cos(yaws) * torch.cos(pitchs),
        torch.sin(pitchs),
    ], -1).cuda()
    look_at_poses += poses
    ups = torch.tile(torch.tensor([0, 0, 1], dtype=torch.float32).cuda(), poses.shape[:-1] + (1,))
    extrinsics = utils3d.torch.extrinsics_look_at(poses, look_at_poses, ups)

    # for loop version
    def _for_loop_test(yaws, pitchs, poses):
        extrinsics = []
        yaws = yaws.unsqueeze(-1)
        pitchs = pitchs.unsqueeze(-1)
        for s in range(yaws.shape[0]):
            for i in range(yaws.shape[1]):
                yaw = yaws[s][i]
                pitch = pitchs[s][i]
                pose = poses[s][i]
                look_at_pose = torch.tensor([
                    torch.sin(yaw) * torch.cos(pitch),
                    torch.cos(yaw) * torch.cos(pitch),
                    torch.sin(pitch),
                ]).cuda()
                look_at_pose += pose
                extr = utils3d.torch.extrinsics_look_at(pose, look_at_pose, torch.tensor([0, 0, 1]).float().cuda())
                extrinsics.append(extr)
        extrinsics = torch.stack(extrinsics)
        extrinsics = torch.reshape(extrinsics, (yaws.shape[0], yaws.shape[1], 4, 4))
        return extrinsics
    # extrinsics_test = _for_loop_test(yaws, pitchs, poses)
    # assert torch.allclose(extrinsics, extrinsics_test) #test passed

    return extrinsics



def main(args):
    args.num_iters = args.num_levels * args.ep_timesteps // args.num_envs
    seed_everything(args.seed)

    # dataset metadata
    dataset_params = \
        {
        "info": {
            "name": args.dataset_name,
            "version": "0.0.1",
            "minetest_logpath": None, #TODO
            "codebase_commit": {
                "craftium": {"url": "placeholder", "commit": "placeholder"}, #TODO
                "minetest": {"url": "placeholder", "commit": "placeholder"},
                "trellis": {"url": "placeholder", "commit": "placeholder"}
            },
        "craftium_env_info": None, #TODO query full minetest_conf
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
            "camera_offset": (0, 0, 0), #TODO
            "hud": False, # can be set to a dict in the future if enabled
        },
        "trellis_input_voxel_info": {
            "dims": (64, 64, 64, 1024), #TODO: check
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

    levelmeta_template = {
        "seed": None,
        "fov_x": args.fov,
        "fov_y": args.fov,
        "spawn_pos": {"player_pos": None, "player_pitch": None, "player_yaw": None},
        "timeofday": None #TODO
    }

    leveldata_template = {
        "timestep_craftium": [], # [Nsteps]
        "timestep_minetest": [], # [Nsteps] #TODO
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
        minetest_conf= {"fov": args.fov},
    )
    vector_env = gym.vector.SyncVectorEnv if not args.async_envs else gym.vector.AsyncVectorEnv
    seeds = deque(np.random.randint(0, 2 ** 31, args.num_levels))
    envs = vector_env([make_env(craftium_kwargs, idx, 0) for idx in range(args.num_envs)])

    level_meta = {}
    data = {}
    for i in range(args.num_iters):
        if i % args.ep_timesteps == 0:
            ts = 0
            curr_seeds = [int(seeds.popleft()) for _ in range(args.num_envs)]
            observations, infos = envs.reset(seed=curr_seeds)
            for s_idx, s in enumerate(curr_seeds):
                level_meta[s] = copy.deepcopy(levelmeta_template)
                data[s] = copy.deepcopy(leveldata_template)
                level_meta[s]["seed"] = s
                level_meta[s]["spawn_pos"] = {"player_pos": infos["player_pos"][s_idx].tolist(), "pitch": infos["player_pitch"][s_idx], "yaw": infos["player_yaw"][s_idx]}
        actions = np.zeros(args.num_envs).astype(int) # NO-OP
        for s_idx, s in enumerate(curr_seeds):
            data[s]["obs_rgb"].append(observations[s_idx])
            data[s]["obs_voxel_mt"].append(infos["voxel_obs"][s_idx])
            data[s]["timestep_craftium"].append(ts)
            data[s]["action"].append(actions[s_idx].tolist())
            data[s]["player_pitch"].append(infos["player_pitch"][s_idx])
            data[s]["player_yaw"].append(infos["player_yaw"][s_idx])
            data[s]["player_pos"].append(infos["player_pos"][s_idx].tolist())
            data[s]["player_vel"].append(infos["player_vel"][s_idx].tolist())
        observations, rewards, terms, truncs, infos = envs.step(actions)
        for s_idx, s in enumerate(curr_seeds):
            data[s]["reward"].append(rewards[s_idx])
            data[s]["termination_flag"].append(terms[s_idx])
            data[s]["truncation_flag"].append(truncs[s_idx])
        ts += 1

    data = {s: {k: np.array(v) for k, v in d.items()} for s, d in data.items()}
    compute_intrisincs_extrinsics(data, level_meta, dataset_params)

    # save data according to the following structure:
    #     - raw
    #         - dataset_name (allows for mixing datasets later)
    #             - seed_folder (one per level)
    #                 - level_metadata.json
    #                 - data.npz
    if not os.path.exists(os.path.join(dataset_root, "raw", args.env_id)):
        os.makedirs(os.path.join(dataset_root, "raw", args.env_id))
    for s in level_meta:
        level_folder = os.path.join(dataset_root, "raw", args.env_id, str(s))
        os.makedirs(level_folder, exist_ok=True)
        with open(os.path.join(level_folder, "level_metadata.json"), "w") as f:
            json.dump(level_meta[s], f, indent=4)
        np.savez_compressed(os.path.join(level_folder, "data.npz"), **data[s])


if __name__ == "__main__":

    args = tyro.cli(Args)
    vdisplay = Xvfb()
    vdisplay.start()
    main(args)
    vdisplay.stop()