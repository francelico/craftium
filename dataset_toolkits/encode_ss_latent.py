import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import copy
import json
import argparse
import torch
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from easydict import EasyDict as edict
from queue import Queue
from dataclasses import dataclass
from typing import Optional
import tyro

import trellis.models as models

torch.set_grad_enabled(False)


def get_voxels(instance):
    position = utils3d.io.read_ply(os.path.join(args.output_dir, 'voxels', f'{instance}.ply'))[0]
    coords = ((torch.tensor(position) + 0.5) * args.resolution).int().contiguous()
    ss = torch.zeros(1, args.resolution, args.resolution, args.resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss


@dataclass
class Arguments:
    """Command line arguments for the program."""
    output_dir: str
    """Directory to save the metadata"""
    enc_pretrained: str = "JeffreyXiang/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
    """Pretrained encoder model"""
    model_root: str = "trellis/weights"
    """Root directory of models"""
    enc_model: Optional[str] = None
    """Encoder model. if specified, use this model instead of pretrained model"""
    ckpt: Optional[str] = None
    """Checkpoint to load"""
    resolution: int = 64
    """Resolution"""
    instances: Optional[str] = None
    """Instances to process"""
    rank: int = 0
    """Process rank"""
    world_size: int = 1
    """Total number of processes"""

if __name__ == '__main__':

    args = tyro.cli(Arguments)

    if args.enc_model is None:
        latent_name = f'{args.enc_pretrained.split("/")[-1]}'
        encoder = models.from_pretrained(args.enc_pretrained).eval().cuda()
    else:
        latent_name = f'{args.enc_model}_{args.ckpt}'
        cfg = edict(json.load(open(os.path.join(args.model_root, args.enc_model, 'config.json'), 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(args.model_root, args.enc_model, 'ckpts', f'encoder_{args.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')

    os.makedirs(os.path.join(args.output_dir, 'ss_latents', latent_name), exist_ok=True)

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
        metadata = metadata[metadata['voxelized'] == True]
        if f'ss_latent_{latent_name}' in metadata.columns:
            metadata = metadata[metadata[f'ss_latent_{latent_name}'] == False]

    start = len(metadata) * args.rank // args.world_size
    end = len(metadata) * (args.rank + 1) // args.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(args.output_dir, 'ss_latents', latent_name, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'ss_latent_{latent_name}': True})
            sha256s.remove(sha256)

    # encode latents
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=32) as loader_executor, \
                ThreadPoolExecutor(max_workers=32) as saver_executor:
            def loader(sha256):
                try:
                    ss = get_voxels(sha256)[None].float()
                    load_queue.put((sha256, ss))
                except Exception as e:
                    print(f"Error loading features for {sha256}: {e}")


            loader_executor.map(loader, sha256s)


            def saver(sha256, pack):
                save_path = os.path.join(args.output_dir, 'ss_latents', latent_name, f'{sha256}.npz')
                np.savez_compressed(save_path, **pack)
                records.append({'sha256': sha256, f'ss_latent_{latent_name}': True})


            for _ in tqdm(range(len(sha256s)), desc="Extracting latents"):
                sha256, ss = load_queue.get()
                ss = ss.cuda().float()
                latent = encoder(ss, sample_posterior=False)
                assert torch.isfinite(latent).all(), "Non-finite latent"
                pack = {
                    'mean': latent[0].cpu().numpy(),
                }
                saver_executor.submit(saver, sha256, pack)

            saver_executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")

    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(args.output_dir, f'ss_latent_{latent_name}_{args.rank}.csv'), index=False)
