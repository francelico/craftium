from typing import *
import torch
import torch.nn as nn
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


class TrellisSlatTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        slat_normalization: dict = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None

    @staticmethod
    def from_pretrained(path: str) -> "TrellisSlatTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisSlatTo3DPipeline, TrellisSlatTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisSlatTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.slat_normalization = args['slat_normalization']

        return new_pipeline

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def unnormalize_slat(
        self,
        slat,
    ) -> sp.SparseTensor:

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    @torch.no_grad()
    def run(
        self,
        slat,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.
        """
        # slat = self.unnormalize_slat(slat) # almost certain this is not needed, the sampler is normalized but the stored latent shouldn't be
        return self.decode_slat(slat, formats)