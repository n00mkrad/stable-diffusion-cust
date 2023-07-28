import functools; print = functools.partial(print, flush = True)
import torch
import diffusers
from diffusers import AutoencoderKL
from diffusers import StableDiffusionInstructPix2PixPipeline # InstructPix2Pix
from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline, OnnxStableDiffusionInpaintPipeline, OnnxStableDiffusionInpaintPipelineLegacy # SD ONNX
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, DiffusionPipeline # SDXL
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
import os, sys, time
from huggingface_hub import snapshot_download

pipe = None
refiner = None

loaded_models = {}

def unload(unload_base, unload_refiner):
    global pipe, refiner
    if unload_base and pipe is not None:
        pipe.to("cpu")
        pipe = None
    if unload_refiner and refiner is not None:
        refiner.to("cpu")
        refiner = None
    torch.cuda.empty_cache()

class NoWatermark:
    def apply_watermark(self, img):
        return img

def load_ip2p(path_mdl):
    global pipe
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

def load_sd_onnx(path_mdl, mode):
    global pipe
    prov = "DmlExecutionProvider"
    # Load model if not yet loaded
    if loaded_models.get("model") != path_mdl:
        print(f'ONNX: Loading model ({path_mdl})')
        pipe = DiffusionPipeline.from_pretrained(path_mdl, torch_dtype = torch.float16, provider=prov, safety_checker=None)
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
        # pipe.safety_checker = lambda images, **kwargs: (images, False)
        loaded_models["model"] = path_mdl


def load_sdxl(path_base, path_refiner, mode, optimize):
    global pipe, refiner
    # Load base model if not yet loaded
    if (pipe is None or loaded_models.get("model") != path_base) and path_base is not None and os.path.exists(path_base):
        print(f'SDXL: Loading base model ({path_base}) [Optimize: {optimize}]')
        if path_base.endswith('.safetensors'):
            fp16_vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            pipe = StableDiffusionXLPipeline.from_single_file(path_base, vae=fp16_vae, local_files_only = True, use_safetensors = True, torch_dtype = torch.float16)
        else:
            pipe = DiffusionPipeline.from_pretrained(path_base, torch_dtype = torch.float16, variant = "fp16", use_safetensors = True)
        if(optimize):
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
        else:
            pipe.to("cuda")
    # Load refiner if required and not yet loaded
    if (refiner is None or loaded_models.get("model_refiner") != path_refiner) and path_refiner is not None and os.path.exists(path_refiner):
        print(f'SDXL: Loading refiner model ({path_refiner} [Optimize: {optimize}])')
        if path_refiner.endswith('.safetensors'):
            if pipe is None:
                refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(path_refiner, vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16), local_files_only = True, use_safetensors = True, torch_dtype = torch.float16)
            else:
                refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(path_refiner, text_encoder_2 = pipe.text_encoder_2, vae = pipe.vae, local_files_only = True, use_safetensors = True, torch_dtype = torch.float16)
        else:
            if pipe is None:
                refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(path_refiner, torch_dtype = torch.float16, variant = "fp16", use_safetensors = True)
            else:
                refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(path_refiner, text_encoder_2 = pipe.text_encoder_2, vae = pipe.vae, torch_dtype = torch.float16, variant = "fp16", use_safetensors = True)
        refiner.watermark = NoWatermark()
        if(optimize):
            refiner.enable_model_cpu_offload()
            refiner.enable_vae_slicing()
            refiner.enable_vae_tiling()
        else:
            refiner.to("cuda")
    # Cast to Text2Img/Img2Img/Inpaint Pipeline, remove watermark, save in loaded_models
    if pipe is not None:
        if mode == "txt2img":
            pipe = StableDiffusionXLPipeline(**pipe.components)
        if mode == "img2img":
            pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
        if mode == "inpaint":
            pipe = StableDiffusionXLInpaintPipeline(**pipe.components)
        pipe.watermark = NoWatermark()
        loaded_models["model"] = path_base
    # Save in loaded_models
    if refiner is not None:
        loaded_models["model_refiner"] = path_refiner
        
def set_scheduler(sampler_name):
    global pipe, refiner
    if pipe is None and refiner is None:
        return
    conf = pipe.scheduler.config if pipe is not None else refiner.scheduler.config
    if sampler_name == "ddim": sched = diffusers.DDIMScheduler.from_config(conf, use_karras_sigmas = False)
    if sampler_name == "plms": sched = diffusers.PNDMScheduler.from_config(conf, use_karras_sigmas = False)
    if sampler_name == "lms": sched = diffusers.LMSDiscreteScheduler.from_config(conf, use_karras_sigmas = False)
    if sampler_name == "heun": sched = diffusers.HeunDiscreteScheduler.from_config(conf, use_karras_sigmas = False)
    if sampler_name == "euler": sched = diffusers.EulerDiscreteScheduler.from_config(conf, use_karras_sigmas = False)
    if sampler_name == "k_euler": sched = diffusers.EulerDiscreteScheduler.from_config(conf, use_karras_sigmas = True)
    if sampler_name == "euler_a": sched = diffusers.EulerAncestralDiscreteScheduler.from_config(conf, use_karras_sigmas = False)
    if sampler_name == "dpm_2": sched = diffusers.KDPM2DiscreteScheduler.from_config(conf, use_karras_sigmas = False)
    if sampler_name == "dpm_2_a": sched = diffusers.KDPM2AncestralDiscreteScheduler.from_config(conf, use_karras_sigmas = False)
    if sampler_name == "dpmpp_2s": sched = diffusers.DPMSolverSinglestepScheduler.from_config(conf, use_karras_sigmas = False)
    if sampler_name == "dpmpp_2m": sched = diffusers.DPMSolverMultistepScheduler.from_config(conf, use_karras_sigmas = False)
    if sampler_name == "k_dpmpp_2m": sched = diffusers.DPMSolverMultistepScheduler.from_config(conf, use_karras_sigmas = True)
    if sampler_name == "dpmpp_2m_sde": sched = diffusers.DPMSolverMultistepScheduler.from_config(conf, use_karras_sigmas = False, algorithm_type="sde-dpmsolver++")
    if sampler_name == "k_dpmpp_2m_sde": sched = diffusers.DPMSolverMultistepScheduler.from_config(conf, use_karras_sigmas = True, algorithm_type="sde-dpmsolver++")
    if sampler_name == "unipc": sched = diffusers.UniPCMultistepScheduler.from_config(conf, use_karras_sigmas = False)
    if pipe is not None:
        print(f"Set base scheduler to {sampler_name}")
        pipe.scheduler = sched
    if refiner is not None:
        print(f"Set refiner scheduler to {sampler_name}")
        refiner.scheduler = sched