from xvfbwrapper import Xvfb
import os
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import tyro
import gymnasium as gym
from dataclasses import dataclass

from dataset_toolkits.util import plot_voxels, plot_rgb

@dataclass
class Args:
    env_id: str = "OpenWorld-v0"
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

def main(args):
    env = gym.make(f"Craftium/{args.env_id}",
                   obs_width=256,
                   obs_height=256,
                   seed=args.seed,
                   enable_voxel_obs=True,
                   voxel_obs_rx=args.voxel_obs_rx,
                   voxel_obs_ry=args.voxel_obs_ry,
                   voxel_obs_rz=args.voxel_obs_rz,
                   init_frames=args.init_frames,
                   minetest_conf={"fov": args.fov},
                   )

    recorded_frames = []
    for t in range(args.num_frames):
        if t % args.ep_timesteps == 0:
            observation, info = env.reset()
        if args.plot_voxel_obs and np.size(info['voxel_obs'])>0:
            plot_rgb(observation)
            voxels = np.transpose(info["voxel_obs"], (0,2,1,3)) # to go from NUE to ENU
            fig,ax = plot_voxels(voxels[...,0])
            fig.show()
            args.plot_voxel_obs = False
        if args.record_video:
            recorded_frames.append(observation)
        action = 0 # NO-OP (can also do env.action_space.sample() for a random action)
        observation, reward, terminated, truncated, _info = env.step(action)

        # if terminated or truncated: # Note: this is absolutely necessary, lua script does not reset the player.
        #     observation, info = env.reset()

    if args.record_video:
        clip = ImageSequenceClip(recorded_frames, fps=24)
        if not os.path.exists(os.path.dirname(args.save_video_to)):
            os.makedirs(f"videos", exist_ok=True)
        clip.write_videofile(args.save_video_to, logger="bar")

    env.close()

if __name__ == "__main__":

    args = tyro.cli(Args)
    vdisplay = Xvfb()
    vdisplay.start()
    main(args)
    vdisplay.stop()