import os
import json
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from queue import Queue
from torchvision import transforms
from dataclasses import dataclass
from typing import Optional
import tyro

torch.set_grad_enabled(False)

def get_data(local_path, extrinsics_key):
    data_path = os.path.join(args.output_dir, local_path, 'data.npz')
    data = np.load(data_path)
    images = np.array(data["obs_rgb"]).astype(np.float32) / 255
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float()

    return {
        'images': images,
        'extrinsics': data[extrinsics_key],
        # need intrinsics to be tiled to match the number of views
        'intrinsics': np.tile(data['intrinsics'], (len(images), 1, 1)),
    }

@dataclass
class Args:
    """Command line arguments for the program."""
    output_dir: str
    """Directory to save the metadata"""
    model: str = "dinov2_vitl14_reg"
    """Feature extraction model"""
    instances: Optional[str] = None
    """Instances to process"""
    batch_size: int = 16
    """Batch size for processing"""
    rank: int = 0
    """Process rank"""
    world_size: int = 1
    """Total number of processes"""
    debug: bool = False
    """Enable debug mode. Will run the code in a single process."""

if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.debug:
        from util import MockThreadPoolExecutor as ThreadPoolExecutor
    else:
        from concurrent.futures import ThreadPoolExecutor

    dataset_params = json.load(open(os.path.join(args.output_dir, 'dataset_params.json')))
    feature_name = args.model
    os.makedirs(os.path.join(args.output_dir, 'features', feature_name), exist_ok=True)

    # load model
    dinov2_model = torch.hub.load('facebookresearch/dinov2', args.model)
    dinov2_model.eval().cuda()
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    n_patch = 518 // 14

    # get file list
    if os.path.exists(os.path.join(args.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(args.output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')
    if args.instances is not None:
        with open(args.instances, 'r') as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata['sha256'].isin(instances)]
    else:
        if f'feature_{feature_name}' in metadata.columns:
            metadata = metadata[metadata[f'feature_{feature_name}'] == False]
        metadata = metadata[metadata['voxelized'] == True]

    start = len(metadata) * args.rank // args.world_size
    end = len(metadata) * (args.rank + 1) // args.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    local_paths = list(metadata['local_path'].values)
    for sha256, local_path in zip(sha256s, local_paths):
        if os.path.exists(os.path.join(args.output_dir, 'features', feature_name, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'feature_{feature_name}': True})
            sha256s.remove(sha256)
            local_paths.remove(local_path)

    # extract features
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=8) as loader_executor, \
                ThreadPoolExecutor(max_workers=2) as saver_executor:
            def loader(local_path, sha256):
                try:
                    data = get_data(local_path, dataset_params['trellis_input_voxel_info']['extrinsics_key'])
                    data['images'] = transform(data['images'])
                    positions = utils3d.io.read_ply(os.path.join(args.output_dir, 'voxels', f'{sha256}.ply'))[0]
                    load_queue.put((sha256, data, positions))
                except Exception as e:
                    print(f"Error loading data for {sha256}: {e}")


            loader_executor.map(loader,local_paths, sha256s)

            def saver(sha256, pack, patchtokens, uv):
                patchtokens = F.grid_sample(
                    patchtokens.cpu(),
                    uv.unsqueeze(1).cpu(),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(2).permute(0, 2, 1).cpu().numpy().astype(np.float16)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    patchtokens = np.nanmean(patchtokens, axis=0)
                patchtokens = np.nan_to_num(patchtokens, nan=0)
                pack['patchtokens'] = patchtokens
                save_path = os.path.join(args.output_dir, 'features', feature_name, f'{sha256}.npz')
                np.savez_compressed(save_path, **pack)
                records.append({'sha256': sha256, f'feature_{feature_name}': True})

            for _ in tqdm(range(len(sha256s)), desc="Extracting features"):
                sha256, data, positions = load_queue.get()
                positions = torch.from_numpy(positions).float().cuda()
                indices = ((positions + 0.5) * 64).long()
                assert torch.all(indices >= 0) and torch.all(indices < 64), "Some vertices are out of bounds"
                n_views = len(data['images'])
                pack = {
                    'indices': indices.cpu().numpy().astype(np.uint8),
                }
                patchtokens_lst = []
                uv_lst = []
                for i in range(0, n_views, args.batch_size):
                    batch_data = {'images': data['images'][i:i + args.batch_size],
                                  'extrinsics': data['extrinsics'][i:i + args.batch_size],
                                  'intrinsics': data['intrinsics'][i:i + args.batch_size]}
                    bs = len(batch_data['images'])
                    batch_images = batch_data["images"].cuda()
                    batch_extrinsics = torch.tensor(batch_data["extrinsics"]).cuda()
                    batch_intrinsics = torch.tensor(batch_data["intrinsics"]).cuda()
                    features = dinov2_model(batch_images, is_training=True)
                    uv, depth = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)
                    uv = uv * 2 - 1
                    uv[depth < 0] = float('nan')
                    mask = ((uv[..., 0] < -1) + (uv[..., 1] < -1) + (uv[..., 0] > 1) + (uv[..., 1] > 1)) > 0
                    mask = mask.unsqueeze(-1).expand(-1, -1, 2)
                    uv[mask] = float('nan')
                    patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(
                        0, 2,1).reshape(bs, 1024, n_patch, n_patch)
                    patchtokens_lst.append(patchtokens)
                    uv_lst.append(uv)
                patchtokens = torch.cat(patchtokens_lst, dim=0)
                uv = torch.cat(uv_lst, dim=0)

                # save features
                saver_executor.submit(saver, sha256, pack, patchtokens, uv)

            saver_executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error happened during processing. Traceback: {e}")


    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(args.output_dir, f'feature_{feature_name}_{args.rank}.csv'), index=False)
