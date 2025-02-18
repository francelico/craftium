import json
import os
from typing import *
import torch
import numpy as np
import pandas as pd
import imageio
import copy
from easydict import EasyDict as edict
from dataclasses import dataclass
from typing import Optional
import tyro
from tqdm import tqdm
from queue import Queue

from trellis.pipelines import TrellisImageTo3DPipeline, TrellisSlatTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils
from trellis.modules import sparse as sp

MAX_SEED = np.iinfo(np.int32).max

def get_rgb_obs(local_path) -> torch.Tensor:
    data_path = os.path.join(args.output_dir, local_path, 'data.npz')
    data = np.load(data_path)
    images = np.array(data["obs_rgb"]).astype(np.float32) / 255
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    return images

def get_trellis_latent(sha256) -> Tuple[torch.Tensor, torch.Tensor]:
    latent_path = os.path.join(args.output_dir, 'latents', args.latent_model, f'{sha256}.npz')
    latent = np.load(latent_path)
    slat = sp.SparseTensor(
        feats=torch.from_numpy(latent['feats']).float(),
        coords=torch.cat([
            torch.zeros(latent['feats'].shape[0], 1).int(),
            torch.from_numpy(latent['coords']).int(),
        ], dim=1),)
    return slat

def diffuse_and_decode_from_rgb_obs_cond(metadata):
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.decode_from_diffuser_rgbcond_pipeline)
    pipeline.cuda()
    seed = get_seed(args.diffusion_seed == -1, args.diffusion_seed)

    # filter out objects that are already processed
    output_paths = []
    for sha256 in copy.copy(metadata['sha256'].values):
        out_p = os.path.join(args.output_dir, f'assets/{sha256}', f'diff_decoded_gs_and_mesh.mp4')
        if os.path.exists(out_p):
            metadata = metadata[metadata['sha256'] != sha256]
        else:
            output_paths.append(out_p)

    # extract features
    load_queue = Queue()
    # try:
    with ThreadPoolExecutor(max_workers=8) as loader_executor:
        def loader(local_path, sha256):
            try:
                rgb = get_rgb_obs(local_path)
                load_queue.put((sha256, rgb))
            except Exception as e:
                print(f"Error loading data for {sha256}: {e}")

        sha256s = metadata['sha256'].values
        local_paths = metadata['local_path'].values
        loader_executor.map(loader, local_paths, sha256s)

        for _ in tqdm(range(len(sha256s)), desc="Running TrellisImageTo3DPipeline"):
            sha256, rgb = load_queue.get()
            outputs = _diffuse_and_decode_from_rgb_obs_cond(
                rgb_obs=rgb.cuda(),
                sha256=sha256,
                pipeline=pipeline,
                seed=seed,
                ss_guidance_strength=args.ss_guidance_strength,
                ss_sampling_steps=args.ss_sampling_steps,
                slat_guidance_strength=args.slat_guidance_strength,
                slat_sampling_steps=args.slat_sampling_steps,
                multiimage_algo=args.multiimage_algo,
            )

def _diffuse_and_decode_from_rgb_obs_cond(
        rgb_obs: torch.Tensor,
        sha256: str,
        pipeline: TrellisImageTo3DPipeline,
        seed: int,
        ss_guidance_strength: float,
        ss_sampling_steps: int,
        slat_guidance_strength: float,
        slat_sampling_steps: int,
        multiimage_algo: Literal["multidiffusion", "stochastic"],
):

    outputs = pipeline.run_multi_image(
        rgb_obs,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
        mode=multiimage_algo,
    )
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    if 'mesh' in outputs:
        video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
        video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(args.output_dir, 'assets', sha256, 'diff_decoded_gs_and_mesh.mp4')
    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path))
    imageio.mimsave(video_path, video, fps=15)
    diff_params = {
        "seed": seed,
        "ss_guidance_strength": ss_guidance_strength,
        "ss_sampling_steps": ss_sampling_steps,
        "slat_guidance_strength": slat_guidance_strength,
        "slat_sampling_steps": slat_sampling_steps,
        "multiimage_algo": multiimage_algo,
    }
    with open(os.path.join(args.output_dir, 'assets', sha256, 'diff_decoded_params.json'), 'w') as f:
        json.dump(diff_params, f)
    return outputs

def decode_from_latent(metadata):
    pipeline = TrellisSlatTo3DPipeline.from_pretrained(args.decode_from_latent_pipeline)
    pipeline.cuda()

    metadata = metadata[metadata[f'latent_{args.latent_model}'] == True]

    # filter out objects that are already processed
    output_paths = []
    for sha256 in copy.copy(metadata['sha256'].values):
        out_p = os.path.join(args.output_dir, f'assets/{sha256}', f'latent_decoded_gs.mp4')
        if os.path.exists(out_p):
            metadata = metadata[metadata['sha256'] != sha256]
        else:
            output_paths.append(out_p)

    # extract features
    load_queue = Queue()
    # try:
    with ThreadPoolExecutor(max_workers=8) as loader_executor:
        def loader(sha256):
            try:
                slat = get_trellis_latent(sha256)
                load_queue.put((sha256, slat))
            except Exception as e:
                print(f"Error loading data for {sha256}: {e}")

        sha256s = metadata['sha256'].values
        loader_executor.map(loader, sha256s)

        for _ in tqdm(range(len(sha256s)), desc="Running TrellisSlatTo3DPipeline"):
            sha256, slat = load_queue.get()
            outputs = _decode_from_latent(
                slat=slat.cuda(),
                sha256=sha256,
                pipeline=pipeline,
            )

def _decode_from_latent(
        slat: sp.SparseTensor,
        sha256: str,
        pipeline: TrellisSlatTo3DPipeline,
):
    outputs = pipeline.run(
        slat,
        formats=["gaussian"],
    )
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_path = os.path.join(args.output_dir, 'assets', sha256, 'latent_decoded_gs.mp4')
    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path))
    imageio.mimsave(video_path, video, fps=15)
    return outputs

def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed

@dataclass
class Args:
    """Command line arguments for the program."""
    output_dir: str
    """Directory to save the metadata"""
    decode_from_diffuser_rgbcond: bool = True
    """Whether to diffuse and decode the images"""
    decode_from_diffuser_rgbcond_pipeline: str = "JeffreyXiang/TRELLIS-image-large"
    diffusion_seed: int = 0
    """Seed for diffusion. Set to -1 to randomize seed."""
    ss_guidance_strength : float = 7.5
    ss_sampling_steps : int = 12
    slat_guidance_strength : float = 3.0
    slat_sampling_steps : int = 12
    multiimage_algo : Literal["multidiffusion", "stochastic"] = "stochastic"

    decode_from_latent: bool = False
    """Whether to decode from latent"""
    decode_from_latent_pipeline: str = "JeffreyXiang/TRELLIS-image-large"
    latent_model: str = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
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
        from dataset_toolkits.util import MockThreadPoolExecutor as ThreadPoolExecutor
    else:
        from concurrent.futures import ThreadPoolExecutor

    # get file list
    if os.path.exists(os.path.join(args.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(args.output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')
    if args.instances is not None:
        with open(args.instances, 'r') as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata['sha256'].isin(instances)]

    if args.decode_from_diffuser_rgbcond:
        diffuse_and_decode_from_rgb_obs_cond(copy.deepcopy(metadata))
        torch.cuda.empty_cache()
    if args.decode_from_latent:
        decode_from_latent(copy.deepcopy(metadata))
        torch.cuda.empty_cache()
