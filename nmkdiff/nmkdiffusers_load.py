import functools; print = functools.partial(print, flush = True)
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline # InstructPix2Pix
from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline, OnnxStableDiffusionInpaintPipeline, OnnxStableDiffusionInpaintPipelineLegacy # SD ONNX
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, DiffusionPipeline # SDXL
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
import os, sys, time
from huggingface_hub import snapshot_download

loaded_models = {}

class NoWatermark:
    def apply_watermark(self, img):
        return img

def load_ip2p(pipe, path_mdl):
    # Load model if not yet loaded
    if loaded_models.get("model") != path_mdl:
        default_mdl_id = "timbrooks/instruct-pix2pix"
        if not path_mdl:
            ignore = ["*.ckpt", "*.safetensors", "safety_checker/*", "*.md", ".git*", "*.png", "*.pt"]
            try:
                path_mdl = snapshot_download(repo_id = default_mdl_id, revision = "fp16", ignore_patterns = ignore)
            except:
                path_mdl = snapshot_download(repo_id = default_mdl_id, revision = "fp16", ignore_patterns = ignore, local_files_only = True)
        print(f'IP2P: Loading model ({path_mdl})')
        try:
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(path_mdl, torch_dtype = torch.float16, safety_checker = None)
        except:
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(path_mdl, torch_dtype = torch.float16, safety_checker = None, local_files_only = True)
        pipe.enable_attention_slicing()
        pipe.to("cuda")
    # Save in loaded_models
    if pipe is not None:
        loaded_models["model"] = path_mdl
    return pipe

def load_sd_onnx(pipe, path_mdl, mode):
    prov = "DmlExecutionProvider"
    # Load model if not yet loaded
    if loaded_models.get("model") != path_mdl:
        print(f'ONNX: Loading model ({path_mdl})')
        pipe = DiffusionPipeline.from_pretrained(path_mdl, torch_dtype = torch.float16, provider=prov)
    # Cast to Text2Img/Img2Img/Inpaint Pipeline, save in loaded_models
    if pipe is not None:
        if mode == "txt2img":
            pipe = OnnxStableDiffusionPipeline(**pipe.components)
        if mode == "img2img":
            pipe = OnnxStableDiffusionImg2ImgPipeline(**pipe.components)
        if mode == "inpaint":
            pipe = OnnxStableDiffusionInpaintPipeline(**pipe.components)
        if mode == "inpaint-legacy":
            pipe = OnnxStableDiffusionInpaintPipelineLegacy(**pipe.components)
        loaded_models["model"] = path_mdl
    # Save in loaded_models
    if pipe is not None:
        loaded_models["model"] = path_mdl
    return pipe

def load_sdxl(pipe_base, pipe_refiner, path_base, path_refiner, mode, optimize):
    # Load base model if not yet loaded
    if loaded_models.get("model") != path_base:
        print(f'SDXL: Loading base model ({path_base}) [Optimize: {optimize}]')
        if path_base.endswith('.safetensors'):
            pipe_base = StableDiffusionXLPipeline.from_single_file(path_base, local_files_only = True, use_safetensors = True, torch_dtype = torch.float16)
        else:
            pipe_base = DiffusionPipeline.from_pretrained(path_base, torch_dtype = torch.float16, variant = "fp16", use_safetensors = True)
        if(optimize):
            pipe_base.enable_model_cpu_offload()
            pipe_base.enable_vae_slicing()
            pipe_base.enable_vae_tiling()
        else:
            pipe_base.to("cuda")
    # Load refiner if required and not yet loaded
    if loaded_models.get("model_refiner") != path_refiner and path_refiner is not None and os.path.exists(path_refiner):
        print(f'SDXL: Loading refiner model ({path_refiner} [Optimize: {optimize}])')
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
    # Cast to Text2Img/Img2Img/Inpaint Pipeline, remove watermark, save in loaded_models
    if pipe_base is not None:
        if mode == "txt2img":
            pipe_base = StableDiffusionXLPipeline(**pipe_base.components)
        if mode == "img2img":
            pipe_base = StableDiffusionXLImg2ImgPipeline(**pipe_base.components)
        if mode == "inpaint":
            pipe_base = StableDiffusionXLInpaintPipeline(**pipe_base.components)
        pipe_base.watermark = NoWatermark()
        loaded_models["model"] = path_base
    # Save in loaded_models
    if pipe_refiner is not None:
        loaded_models["model_refiner"] = path_refiner
    return pipe_base, pipe_refiner
