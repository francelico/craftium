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

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

MAX_SEED = np.iinfo(np.int32).max

def get_rgb_obs(local_path) -> torch.Tensor:
    data_path = os.path.join(args.output_dir, local_path, 'data.npz')
    data = np.load(data_path)
    images = np.array(data["obs_rgb"]).astype(np.float32) / 255
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float()

    return images

def get_trellis_latent(sha256) -> Tuple[torch.Tensor, torch.Tensor]:
    latent_path = os.path.join(args.output_dir, 'latent', args.latent_model, f'{sha256}.npz')
    latent = np.load(latent_path)

    return latent

def diffuse_and_decode_from_rgb_obs_cond(
        rgb_obs: torch.Tensor,
        sha256: str,
        image_3d_pipeline: TrellisImageTo3DPipeline,
        seed: int,
        ss_guidance_strength: float,
        ss_sampling_steps: int,
        slat_guidance_strength: float,
        slat_sampling_steps: int,
        multiimage_algo: Literal["multidiffusion", "stochastic"],
):

    outputs = image_3d_pipeline.run_multi_image(
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
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(args.output_dir, 'assets', sha256, 'diff_decoded_gs_and_mesh.mp4')
    imageio.mimsave(video_path, video, fps=15)
    return outputs

def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed

def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }


def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')

    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )

    return gs, mesh

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
        from util import MockThreadPoolExecutor as ThreadPoolExecutor
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
        image_3d_pipeline = TrellisImageTo3DPipeline.from_pretrained(args.decode_from_diffuser_rgbcond_pipeline)
        image_3d_pipeline.cuda()
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
        load_queue = Queue(maxsize=4)
        # try:
        with ThreadPoolExecutor(max_workers=8) as loader_executor, \
                ThreadPoolExecutor(max_workers=8) as saver_executor:
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
                outputs = diffuse_and_decode_from_rgb_obs_cond(
                        rgb_obs = rgb.cuda(),
                        sha256 = sha256,
                        image_3d_pipeline = image_3d_pipeline,
                        seed = seed,
                        ss_guidance_strength = args.ss_guidance_strength,
                        ss_sampling_steps = args.ss_sampling_steps,
                        slat_guidance_strength = args.slat_guidance_strength,
                        slat_sampling_steps = args.slat_sampling_steps,
                        multiimage_algo = args.multiimage_algo,
                )
