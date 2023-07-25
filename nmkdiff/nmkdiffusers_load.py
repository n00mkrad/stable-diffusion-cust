import functools; print = functools.partial(print, flush = True)
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline # InstructPix2Pix
from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline, OnnxStableDiffusionInpaintPipeline, OnnxStableDiffusionInpaintPipelineLegacy # SD ONNX
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, DiffusionPipeline # SDXL
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
import os, sys, time

loaded_models = {}

class NoWatermark:
    def apply_watermark(self, img):
        return img

def load_sdxl(pipe_base, pipe_refiner, path_base, path_refiner, mode, optimize):
    if loaded_models.get("model") != path_base:
        print(f'SDXL: Loading base model ({path_base})')
        if path_base.endswith('.safetensors'):
            pipe_base = StableDiffusionXLPipeline.from_single_file(path_base, local_files_only = True, use_safetensors = True, torch_dtype = torch.float16)
        else:
            pipe_base = DiffusionPipeline.from_pretrained(path_base, torch_dtype = torch.float16, variant = "fp16", use_safetensors = True)
        if mode == "txt2img":
            pipe_base = StableDiffusionXLPipeline(**pipe_base.components)
        if mode == "img2img":
            pipe_base = StableDiffusionXLImg2ImgPipeline(**pipe_base.components)
        if mode == "inpaint":
            pipe_base = StableDiffusionXLInpaintPipeline(**pipe_base.components)
        if(optimize):
            pipe_base.enable_model_cpu_offload()
            pipe_base.enable_vae_slicing()
            pipe_base.enable_vae_tiling()
        else:
            pipe_base.to("cuda")
        pipe_base.watermark = NoWatermark()
    if loaded_models.get("model_refiner") != path_refiner and path_refiner is not None and os.path.exists(path_refiner):
        print(f'SDXL: Loading refiner model ({path_refiner})')
        if path_refiner.endswith('.safetensors'):
            pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(path_refiner, local_files_only = True, use_safetensors = True, torch_dtype = torch.float16)
        else:
            pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(path_refiner, text_encoder_2 = pipe_base.text_encoder_2, vae = pipe_base.vae, torch_dtype = torch.float16, variant = "fp16", use_safetensors = True)
        pipe_refiner.watermark = NoWatermark()
        if(optimize):
            pipe_refiner.enable_model_cpu_offload()
            pipe_refiner.enable_vae_slicing()
            pipe_refiner.enable_vae_tiling()
        else:
            pipe_refiner.to("cuda")
    if pipe_base is not None:
        loaded_models["model"] = path_base
    if pipe_refiner is not None:
        loaded_models["model_refiner"] = path_refiner
    return pipe_base, pipe_refiner
