import os
import copy
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
import tyro
from tqdm import tqdm
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def _save_video(local_path, output_path):
    data_path = os.path.join(args.output_dir, local_path, 'data.npz')
    data = np.load(data_path)
    images = data["obs_rgb"]

    clip = ImageSequenceClip([im for im in images], fps=10)
    save_dir = os.path.dirname(output_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    clip.write_videofile(output_path, logger="bar")

def save_videos(metadata, max_workers=None, desc='Rendering recorded trajectories'):
    # filter out objects that are already processed
    output_paths = []
    for sha256 in copy.copy(metadata['sha256'].values):
        out_p = os.path.join(args.output_dir, f'assets/{sha256}', f'recorded_obs.mp4')
        if os.path.exists(out_p):
            metadata = metadata[metadata['sha256'] != sha256]
        else:
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