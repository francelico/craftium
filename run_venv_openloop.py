from xvfbwrapper import Xvfb
import os
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import tyro
import gymnasium as gym
from dataclasses import dataclass
from typing import Optional

from dataset_toolkits.util import plot_voxels, plot_rgb

@dataclass
class Args:
    env_id: str = "OpenWorld-v0"
    num_envs: int = 4
    async_envs: bool = False
    mt_port: int = 49155
    """TCP port used by Minetest server and client communication. Multiple envs will use successive ports."""
    mt_wd: str = "./"
    """Directory where the Minetest working directories will be created (defaults to the current one)"""
    seed: int = 0
    model: str = "models/3x.model"
    "path to the model skeleton (not used)"
    weights: str = "models/BC-house-3x.weights (not used)"
    "path to the model weights"
    record_video: bool = False
    "capture a video of the agent's view"
    plot_voxel_obs: bool = False
    "plot the voxel observation"
    save_video_to: str = "videos/latest.mp4"
    "save path for the video"
    num_frames: int = 24*60
    "number of frames to record when recording a video"
    ep_timesteps: int = 240
    "number of timesteps for each episode"
    voxel_obs_rx: int = 12
    "x radius of the voxel observation"
    voxel_obs_ry: int = 12
    "y radius of the voxel observation"
    voxel_obs_rz: int = 12
    "z radius of the voxel observation"
    fov: int = 90
    "vertical field of view of the agent in degrees"
    init_frames: int = 200
    "number of frames to wait before starting the episode"
    fps_max: int = 200
    "target fps for the environment"
    pmul: Optional[int] = None
    """Physics speed multiplier. Defaults to the default value of CraftiumEnv."""

def make_env(env_id, idx, fps_max, pmul, mt_port, mt_wd, seed):
    def thunk():
        craftium_kwargs = dict(
            run_dir_prefix=mt_wd,
            mt_port=mt_port,
            rgb_observations=True,
            obs_width=256,
            obs_height=256,
            voxel_obs_rx=args.voxel_obs_rx,
            voxel_obs_ry=args.voxel_obs_ry,
            voxel_obs_rz=args.voxel_obs_rz,
            init_frames=args.init_frames,
            seed=seed,
            fps_max=fps_max,
            pmul=pmul,
            minetest_conf={"fov": args.fov},
        )
        # if capture_video and idx == 0:
        #     env = gym.make(env_id, render_mode="rgb_array", **craftium_kwargs)
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # else:
        env = gym.make(env_id, **craftium_kwargs)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk

def main(args):

    # env setup
    vector_env = gym.vector.SyncVectorEnv if not args.async_envs else gym.vector.AsyncVectorEnv
    envs = vector_env(
        [make_env(
            f"Craftium/{args.env_id}",
            i,
            args.fps_max,
            args.pmul,
            args.mt_port + i,
            args.mt_wd,
            args.seed + i
        ) for i in range(args.num_envs)],
    )

    recorded_frames = [] #{f'env_{i}': [] for i in range(args.num_envs)}
    for t in range(args.num_frames):
        if t % args.ep_timesteps == 0:
            observations, infos = envs.reset()
        if args.plot_voxel_obs and np.size(infos['voxel_obs'][0])>0:
            plot_rgb(observations[0])
            voxels = np.transpose(infos["voxel_obs"][0], (0,2,1,3)) # to go from NUE to ENU
            plot_voxels(voxels[...,0])
            args.plot_voxel_obs = False
        if args.record_video:
            recorded_frames.append(observations)
        actions = np.zeros(args.num_envs).astype(int) # NO-OP (can also do env.action_space.sample() for a random action)
        observations, rewards, terms, truncs, infos = envs.step(actions)

        # if terms or truns: # Note: this is absolutely necessary, lua script does not reset the player.
        #     observations, infos = envs.reset()

    if args.record_video:
        frames_by_env = {f'env_{i}': [] for i in range(args.num_envs)}
        for frame in recorded_frames:
            for i, f in enumerate(frame):
                frames_by_env[f'env_{i}'].append(f)
        for k, v in frames_by_env.items():
            clip = ImageSequenceClip(v, fps=24)
            if not os.path.exists(os.path.dirname(args.save_video_to)):
                os.makedirs(f"videos", exist_ok=True)
            clip.write_videofile(args.save_video_to.replace(".mp4", f"_{k}.mp4"), logger="bar")

    envs.close()

if __name__ == "__main__":

    args = tyro.cli(Args)
    vdisplay = Xvfb()
    vdisplay.start()
    main(args)
    vdisplay.stop()