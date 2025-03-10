import os
import copy
import warnings
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

def save_assets(metadata, max_workers=None, desc='Rendering...'):
    output_paths = []
    for sha256 in copy.copy(metadata['sha256'].values):
        out_p = os.path.join(args.output_dir, f'assets/{sha256}')
        output_paths.append(out_p)

    # load metadata
    metadata = metadata.to_dict('records')

    # processing objects
    max_workers = max_workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(metadata), desc=desc) as pbar:
        def worker(metadatum, output_path):
            local_path = metadatum['local_path']
            sha256 = metadatum['sha256']
            try:
                _save_video(local_path, os.path.join(output_path, f'recorded_obs.mp4'))
                _save_multiviews(local_path, output_path, sha256, save_frames=args.save_frames)
                pbar.update()
            except Exception as e:
                print(f"Error processing object {sha256}: {e}")
                pbar.update()

        executor.map(worker, metadata, output_paths)
        executor.shutdown(wait=True)

def _save_video(local_path, output_path):
    if not args.overwrite and os.path.exists(output_path):
        return

    data_path = os.path.join(args.output_dir, local_path, 'data.npz')
    data = np.load(data_path)
    images = data["obs_rgb"]

    clip = ImageSequenceClip([im for im in images], fps=10)
    save_dir = os.path.dirname(output_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    clip.write_videofile(output_path, logger="bar")

def _save_multiviews(local_path, save_dir, sha256, save_frames=True):
    if not args.overwrite and os.path.exists(os.path.join(save_dir, 'voxel_multiview.png')):
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    rawdata_path = os.path.join(args.output_dir, local_path, 'data.npz')
    rawdata = np.load(rawdata_path)
    positions = utils3d.io.read_ply(os.path.join(args.output_dir, 'voxels', f'{sha256}.ply'))[0]
    positions = torch.from_numpy(positions).float()
    batch_images = torch.tensor(rawdata["obs_rgb"]).float().permute(0,3,1,2)
    batch_extrinsics = torch.tensor(rawdata["extrinsics_local"])
    batch_intrinsics = torch.tensor(rawdata['intrinsics'])
    batch_intrinsics = torch.tile(batch_intrinsics, (len(batch_images), 1, 1))
    uv, depth = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)
    uv = uv * 2 - 1
    uv[depth<0] = float('nan')
    mask = ((uv[..., 0] < -1) + (uv[..., 1] < -1) + (uv[..., 0] > 1) + (uv[..., 1] > 1)) > 0
    mask = mask.unsqueeze(-1).expand(-1, -1, 2)
    uv[mask] = float('nan')
    f3d = F.grid_sample(
        batch_images,
        uv.unsqueeze(1),
        mode='bilinear',
        align_corners=False,
    ).squeeze(2).permute(0, 2, 1).cpu().numpy()
    f3d = np.clip(f3d, 0, 255)

    # save individual frames and corresponding RGB 3D projections
    if save_frames:
        frame_ids = np.arange(0, len(batch_images), 10)
        for k in frame_ids:
            fig, ax = render_multiview(positions, f3d[k])
            os.makedirs(os.path.join(save_dir, f'frames'), exist_ok=True)
            frame = Image.fromarray(rawdata["obs_rgb"][k])
            # save frame and figure side by side in a single image
            save_to = os.path.join(save_dir, f'frames', f'frame_{k}.png')
            combine_images_horizontally(frame, fig, save_to)
            plt.close(fig)

    # save voxel ground truth and multiview RGB 3D projection
    vox_obs_mt = rawdata["obs_voxel_mt"]
    fig_voxmt, ax_voxmt = plot_voxels(vox_obs_mt[-1,...,0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f3d = np.nanmean(f3d, axis=0).astype(np.uint8)
    # f3d = np.nan_to_num(f3d, nan=0) needed in actual code to avoid model processing nans, but in this case we want to visualise them when plotting
    fig_multiview, ax_multiview = render_multiview(positions, f3d)
    combine_images_horizontally(fig_voxmt, fig_multiview, os.path.join(save_dir, 'voxel_multiview.png'))
    plt.close(fig_multiview)
    plt.close(fig_voxmt)


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

def render_multiview(positions, f3d_rgb, size=80, alpha=0.4, nan_color=(255, 51, 153)):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(projection='3d')
    # Make axes equal/orthonormal
    ax.set_box_aspect([1, 1, 1])
    colors = []
    for color in f3d_rgb:
        if all([np.isnan(c) for c in color]):
            color = nan_color
        colors.append(tuple([color[0]/255, color[1]/255, color[2]/255]))
    positions = positions.cpu().numpy()
    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
    img = ax.scatter(xs,ys,zs, c=colors, s=size, alpha=alpha)
    set_axes_equal(ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig, ax

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
    save_frames: bool = True
    """Save individual frames when rendering multiview"""
    overwrite: bool = False
    """Whether to overwrite files when they already exist"""
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
    save_assets(copy.deepcopy(metadata), max_workers=args.max_workers)