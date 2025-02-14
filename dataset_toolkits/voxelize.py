import json
import os
import copy
import sys
import importlib
import argparse
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from functools import partial
import numpy as np
import tyro
import utils3d
from tqdm import tqdm

def voxelize(metadata, output_dir, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    # load metadata
    metadata = metadata.to_dict('records')

    # load dataset params
    dataset_params = json.load(open(os.path.join(output_dir, 'dataset_params.json')))
    trellis_input_voxel_info = dataset_params['trellis_input_voxel_info']
    def check_trellis_input_voxel_info(trellis_input_voxel_info):
        supported_vox_grid_size = 64
        trellis_input_dims = trellis_input_voxel_info['dims']
        assert len(trellis_input_dims) == 4, "Trellis input voxel info must have 4 dimensions"
        assert trellis_input_dims[0] == trellis_input_dims[1] == trellis_input_dims[
            2] == supported_vox_grid_size, "Trellis input voxel must be 64x64x64"
        vox_preprocessing = trellis_input_voxel_info['active_voxel_preprocessing']
        ignore_nodes = vox_preprocessing['empty_space_node_ids']
        assert vox_preprocessing['crop'] and not vox_preprocessing['pad'] and not vox_preprocessing[
            'scale'], "Voxel preprocessing must crop and not resize"
        crop_args = vox_preprocessing['crop']
        return crop_args, ignore_nodes, supported_vox_grid_size

    crop_args, ignore_nodes, output_vox_grid = check_trellis_input_voxel_info(trellis_input_voxel_info)

    # processing objects
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
                tqdm(total=len(metadata), desc=desc) as pbar:
            def worker(metadatum):
                try:
                    local_path = metadatum['local_path']
                    sha256 = metadatum['sha256']
                    file = os.path.join(output_dir, local_path, 'data.npz')
                    record = _voxelize_mt(file, sha256, output_dir, ignore_nodes, crop_args, output_vox_grid)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
                    pbar.update()

            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")

    return pd.DataFrame.from_records(records)

def _voxelize_mt(file, sha256, output_dir, ignore_nodes, crop_args, output_vox_grid):

    # Only load 'obs_voxel_mt' from the .npz file
    mt_voxel_array = np.load(file)['obs_voxel_mt']
    assert mt_voxel_array.ndim == 5, "MT voxel data must have 4 dimensions (T, X, Y, Z, C)"
    assert mt_voxel_array.shape[-1] == 3, "MT voxel array must have 3 channels"
    mt_voxel_array = mt_voxel_array[-1]  # Only take the env representation at the last timestep
    mt_voxel_array = mt_voxel_array[..., 0]  # Only take the first channel of the MT voxel data

    mt_vox_shape = mt_voxel_array.shape
    assert all([d >= output_vox_grid for d in mt_vox_shape]), "MT voxel array must be as large or larger than the output voxel grid"
    # Crop the voxel grid around each edge to the desired size
    mt_voxel_array = mt_voxel_array[crop_args["left"][0]:crop_args["right"][0], crop_args["left"][1]:crop_args["right"][1],
                     crop_args["left"][2]:crop_args["right"][2]]
    node_mask = np.ones(mt_voxel_array.shape, dtype=bool)
    for node in ignore_nodes:
        node_mask &= mt_voxel_array != node
    voxel_grid = np.argwhere(node_mask)
    vertices = (voxel_grid + 0.5) / output_vox_grid - 0.5
    utils3d.io.write_ply(os.path.join(output_dir, 'voxels', f'{sha256}.ply'), vertices)
    return {'sha256': sha256, 'voxelized': True, 'num_voxels': len(vertices)}


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
        from util import MockThreadPoolExecutor as ThreadPoolExecutor
    else:
        from concurrent.futures import ThreadPoolExecutor

    os.makedirs(os.path.join(args.output_dir, 'voxels'), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(args.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(args.output_dir, 'metadata.csv'))
    if args.instances is None:
        if 'voxelized' in metadata.columns:
            metadata = metadata[metadata['voxelized'] == False]
    else:
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

    # filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(args.output_dir, 'voxels', f'{sha256}.ply')):
            pts = utils3d.io.read_ply(os.path.join(args.output_dir, 'voxels', f'{sha256}.ply'))[0]
            records.append({'sha256': sha256, 'voxelized': True, 'num_voxels': len(pts)})
            metadata = metadata[metadata['sha256'] != sha256]

    print(f'Processing {len(metadata)} objects...')

    # process objects
    voxelized = voxelize(metadata, args.output_dir, max_workers=args.max_workers, desc='Voxelizing')
    voxelized = pd.concat([voxelized, pd.DataFrame.from_records(records)])
    voxelized.to_csv(os.path.join(args.output_dir, f'voxelized_{args.rank}.csv'), index=False)