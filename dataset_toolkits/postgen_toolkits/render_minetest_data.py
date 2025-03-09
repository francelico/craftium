import os
import copy
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import tyro
import utils3d
from tqdm import tqdm
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image
import matplotlib.pyplot as plt

from dataset_toolkits.util import plot_voxels, set_axes_equal

#TODO:
# - remove factor of 10 in player positions and vel.

def _save_video(local_path, output_path):
    data_path = os.path.join(args.output_dir, local_path, 'data.npz')
    data = np.load(data_path)
    images = data["obs_rgb"]

    clip = ImageSequenceClip([im for im in images], fps=10)
    save_dir = os.path.dirname(output_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    clip.write_videofile(output_path, logger="bar")

def _save_voxel_obs(local_path, output_path):
    data_path = os.path.join(args.output_dir, local_path, 'data.npz')
    data = np.load(data_path)
    vox_obs_mt = data["obs_voxel_mt"]
    fig, ax = plot_voxels(vox_obs_mt[-1,...,0])
    save_dir = os.path.dirname(output_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(os.path.dirname(output_path), 'voxel_obs.png'))

def _save_multiview(local_path, output_path, sha256):
    CAM_OFFSET = np.array([0, 0, 1.47]) # 1.6 (mineclone) or 1.47 (minetest)
    PLAYER_IN_NODE_OFFSET = np.array([0, 0, -.5])
    CAM_POSE = (-PLAYER_IN_NODE_OFFSET + CAM_OFFSET)/64 # TODO: check if PLAYER_IN_NODE_OFFSET should be always neg
    rawdata_path = os.path.join(args.output_dir, local_path, 'data.npz')
    rawdata = np.load(rawdata_path)
    positions = utils3d.io.read_ply(os.path.join(args.output_dir, 'voxels', f'{sha256}.ply'))[0]
    positions = torch.from_numpy(positions).float()
    batch_images = torch.tensor(rawdata["obs_rgb"]).float().permute(0,3,1,2)
    yaw_orig = torch.tensor(rawdata["player_yaw"]).float()
    pitch_orig = torch.tensor(rawdata["player_pitch"]).float()

    save_dir = os.path.dirname(output_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    possible_combinations = [
        # {"ycoef": -1, "yoff": 270, "pcoef": 1, "poff": 0}, # THIS pWORKS for extrinsics2 (maybe)
        # {"ycoef": -1, "yoff": 90, "pcoef": -1, "poff": 0}, # THIS WORKS for extrinsics4
        {"ycoef": 1, "yoff": 0, "pcoef": 1, "poff": 0}, # THIS WORKS for extrinsics4 + updated NUEtoENU wrapper
    ]

    for comb in possible_combinations:
        ycoef = comb["ycoef"]
        yoff = comb["yoff"]
        pcoef = comb["pcoef"]
        poff = comb["poff"]
        yaw = (ycoef*yaw_orig + yoff) % 360
        pitch = pcoef*pitch_orig + poff
        cam_pose = torch.tile(torch.tensor(CAM_POSE), (len(batch_images), 1)).float()
        batch_extrinsics = yaw_pitch_cam_pos_to_extrinsics4(yaw, pitch, cam_pose)
        # batch_extrinsics = torch.tensor(rawdata["extrinsics_local"])
        batch_intrinsics = torch.tensor(rawdata['intrinsics'])
        # fov_x = np.sqrt(16/10) * 90 # this is the horizontal fov in minetest #TODO: remove when ready
        # fov_y = np.sqrt(16/10) * 90 # this is the vertical fov in minetest
        # batch_intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.deg2rad(torch.tensor([fov_x])), torch.deg2rad(torch.tensor([fov_y]))).float()
        batch_intrinsics = torch.tile(batch_intrinsics, (len(batch_images), 1, 1))
        uv, depth = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)
        uv = uv * 2 - 1
        uv[depth<0] = float('nan')
        f3d = F.grid_sample(
            batch_images,
            uv.unsqueeze(1),
            mode='bilinear',
            align_corners=False,
        ).squeeze(2).permute(0, 2, 1).cpu().numpy()
        save_dir = os.path.dirname(output_path)
        save_dir = os.path.join(save_dir, 'yaw_pitch_cam_pos_to_extrinsics4')
        os.makedirs(save_dir, exist_ok=True)
        for k in [0,10,20,]:
            fig, ax = render_multiview(positions, f3d[k])
            os.makedirs(os.path.join(save_dir, f'frame_{k}'), exist_ok=True)
            frame = Image.fromarray(rawdata["obs_rgb"][k])
            # save frame and figure side by side in a single image
            save_to = os.path.join(save_dir, f'frame_{k}', f'YAW[{ycoef}]o{yoff}_P[{pcoef}]o{poff}_S{k}.png')
            combine_images_horizontally(frame, fig, save_to)
        f3d = np.nanmean(f3d, axis=0).astype(np.uint8)
        fig, ax = render_multiview(positions, f3d)
        save_dir = os.path.dirname(output_path)
        save_dir = os.path.join(save_dir, 'yaw_pitch_cam_pos_to_extrinsics4', 'combined')
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'YAW[{ycoef}]o{yoff}_P[{pcoef}]o{poff}.png'))

        plt.close(fig)


def combine_images_horizontally(image1, image2, output_path):
    """
    Combines two images horizontally and saves the result.

    Parameters:
    image1: Can be either a PIL Image object or a matplotlib figure
    image2: Can be either a PIL Image object or a matplotlib figure
    output_path: Path where the combined image will be saved

    Returns:
    PIL Image object of the combined image
    """
    # Convert matplotlib figure to PIL Image if needed
    processed_imgs = []
    for img in [image1, image2]:
        if hasattr(img, 'canvas'):
            img.set_tight_layout(True)
            img.subplots_adjust(left=0, right=1, top=1, bottom=0)
            img.canvas.draw()
            img = Image.fromarray(np.array(img.canvas.renderer.buffer_rgba()))
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        processed_imgs.append(img)
    image1, image2 = processed_imgs

    # Create a new image with width = sum of individual widths, and max height
    total_width = image1.width + image2.width
    max_height = max(image1.height, image2.height)
    combined_image = Image.new('RGB', (total_width, max_height))

    # Paste the images side by side
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))

    # Save the combined image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_image.save(output_path)

    return combined_image


def yaw_pitch_cam_pos_to_extrinsics2(yaws, pitchs, poses):
    yaws = torch.deg2rad(yaws)
    pitchs = torch.deg2rad(pitchs)
    look_at_poses = -torch.stack([
        torch.sin(yaws) * torch.cos(pitchs),
        torch.cos(yaws) * torch.cos(pitchs),
        torch.sin(pitchs),
    ], -1)
    look_at_poses += poses
    ups = torch.tile(torch.tensor([0, 0, 1], dtype=torch.float32), poses.shape[:-1] + (1,))
    extrinsics = utils3d.torch.extrinsics_look_at(poses, look_at_poses, ups)
    return extrinsics

def yaw_pitch_cam_pos_to_extrinsics4(yaws, pitchs, poses):
    yaws = torch.deg2rad(yaws)
    pitchs = torch.deg2rad(pitchs)
    look_at_poses = torch.stack([
        torch.sin(yaws) * torch.cos(pitchs),
        torch.cos(yaws) * torch.cos(pitchs),
        torch.sin(pitchs),
    ], -1)
    look_at_poses += poses
    ups = torch.tile(torch.tensor([0, 0, 1], dtype=torch.float32), poses.shape[:-1] + (1,))
    extrinsics = utils3d.torch.extrinsics_look_at(poses, look_at_poses, ups)
    return extrinsics

# TODO: remove when ready
def yaw_pitch_cam_pos_to_extrinsics3(yaws, pitchs, poses):
    yaws = torch.deg2rad(yaws)
    pitchs = torch.deg2rad(pitchs)
    look_at_poses = torch.stack([
        torch.cos(yaws) * torch.cos(pitchs),
        torch.sin(yaws) * torch.cos(pitchs),
        torch.sin(pitchs),
    ], -1)
    look_at_poses += poses
    ups = torch.tile(torch.tensor([0, 0, 1], dtype=torch.float32), poses.shape[:-1] + (1,))
    extrinsics = utils3d.torch.extrinsics_look_at(poses, look_at_poses, ups)
    return extrinsics
#
# def new_yaw_pitch_cam_pos_to_extrinsics(yaws, pitchs, poses, invert=False):
#     batch_size = yaws.shape[0]
#     device = yaws.device
#     dtype = yaws.dtype
#     yaws = torch.deg2rad(yaws)
#     pitchs = torch.deg2rad(pitchs)
#     extrinsics = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
#
#     # x-axis rotation
#     extrinsics[..., 0, 0] = torch.cos(yaws)
#     extrinsics[..., 1, 0] = torch.cos(pitchs) * torch.sin(yaws)
#     extrinsics[..., 2, 0] = torch.sin(pitchs) * torch.sin(yaws)
#
#     # y-axis rotation
#     extrinsics[..., 0, 1] = -torch.sin(yaws)
#     extrinsics[..., 1, 1] = torch.cos(pitchs) * torch.cos(yaws)
#     extrinsics[..., 2, 1] = torch.sin(pitchs) * torch.cos(yaws)
#
#     # z-axis rotation
#     extrinsics[..., 0, 2] = 0
#     extrinsics[..., 1, 2] = -torch.sin(pitchs)
#     extrinsics[..., 2, 2] = torch.cos(pitchs)
#
#
#     # translation
#     extrinsics[..., :3, 3] = poses
#
#     if invert:
#         extrinsics[...,:3, 1:3] *= -1
#         extrinsics = torch.inverse(extrinsics)
#
#     return extrinsics
#
# def extrinsics_from_angles_opencv(
#         yaw: torch.Tensor,  # [...] in degrees
#         pitch: torch.Tensor,  # [...] in degrees
#         invert = False
# ) -> torch.Tensor:  # [..., 4, 4]
#     """
#     Compute camera extrinsics for a camera at origin with given yaw and pitch.
#     Following OpenCV convention:
#     - Y points down
#     - X points right
#     - Z points forward
#     - Positive pitch means looking down (rotation around X)
#     - Positive yaw means looking right (rotation around Y)
#     """
#     batch_size = yaw.shape[0]
#     device = yaw.device
#     dtype = yaw.dtype
#
#     yaw = torch.deg2rad(yaw)
#     pitch = torch.deg2rad(pitch)
#     cos_yaw = torch.cos(yaw)
#     sin_yaw = torch.sin(yaw)
#     cos_pitch = torch.cos(pitch)
#     sin_pitch = torch.sin(pitch)
#
#     R = torch.zeros((batch_size, 3, 3), device=device, dtype=dtype)
#
#     # OpenCV convention rotation matrix
#     R[..., 0, 0] = cos_yaw
#     R[..., 0, 1] = sin_yaw * sin_pitch
#     R[..., 0, 2] = sin_yaw * cos_pitch
#
#     R[..., 1, 0] = 0
#     R[..., 1, 1] = cos_pitch
#     R[..., 1, 2] = -sin_pitch
#
#     R[..., 2, 0] = -sin_yaw
#     R[..., 2, 1] = cos_yaw * sin_pitch
#     R[..., 2, 2] = cos_yaw * cos_pitch
#
#     extrinsics = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
#     extrinsics[..., :3, :3] = R
#
#     if invert:
#         # extrinsics[...,:3, 1:3] *= -1
#         extrinsics = torch.inverse(extrinsics)
#
#     return extrinsics

def render_multiview(positions, f3d_rgb, size=80, alpha=0.4,):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(projection='3d')
    # Make axes equal/orthonormal
    ax.set_box_aspect([1, 1, 1])
    colors = []
    for color in f3d_rgb:
        colors.append(tuple([color[0]/255, color[1]/255, color[2]/255]))
    positions = positions.cpu().numpy()
    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
    img = ax.scatter(xs,ys,zs, c=colors, s=size, alpha=alpha)
    set_axes_equal(ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig, ax

def save_videos(metadata, max_workers=None, desc='Rendering recorded trajectories'):
    # filter out objects that are already processed
    output_paths = []
    # TODO: better output path handling?
    for sha256 in copy.copy(metadata['sha256'].values):
        out_p = os.path.join(args.output_dir, f'assets/{sha256}', f'recorded_obs.mp4')
        output_paths.append(out_p)

    # load metadata
    metadata = metadata.to_dict('records')

    # processing objects
    max_workers = max_workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(metadata), desc=desc) as pbar:
        def worker(metadatum, output_path):
            try:
                local_path = metadatum['local_path']
                sha256 = metadatum['sha256']
                _save_video(local_path, output_path)
                _save_voxel_obs(local_path, output_path)
                _save_multiview(local_path, output_path, sha256) #TODO: uncomment when cleaned up
                pbar.update()
            except Exception as e:
                print(f"Error processing object {sha256}: {e}")
                pbar.update()

        executor.map(worker, metadata, output_paths)
        executor.shutdown(wait=True)

@dataclass
class Args:  # Inheriting from DatasetArguments to include those options
    """Command line arguments for the program."""
    output_dir: str
    """Directory to save the metadata"""
    instances: Optional[str] = None
    """Instances to process"""
    rank: int = 0
    """Process rank"""
    world_size: int = 1
    """Total number of processes"""
    max_workers: Optional[int] = None
    """Maximum number of worker processes"""
    debug: bool = False

if __name__ == '__main__':
    args = tyro.cli(Args)
    if args.debug:
        from dataset_toolkits.util import MockThreadPoolExecutor as ThreadPoolExecutor
    else:
        from concurrent.futures import ThreadPoolExecutor

    os.makedirs(os.path.join(args.output_dir, 'voxels'), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(args.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(args.output_dir, 'metadata.csv'))

    if args.instances:
        if os.path.exists(args.instances):
            with open(args.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = args.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * args.rank // args.world_size
    end = len(metadata) * (args.rank + 1) // args.world_size
    metadata = metadata[start:end]
    records = []

    print(f'Processing {len(metadata)} objects...')
    # save videos
    save_videos(copy.deepcopy(metadata), max_workers=args.max_workers)