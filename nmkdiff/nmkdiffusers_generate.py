import functools; print = functools.partial(print, flush = True)
import PIL
from PIL import PngImagePlugin, Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline # InstructPix2Pix
from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline, OnnxStableDiffusionInpaintPipeline, OnnxStableDiffusionInpaintPipelineLegacy # SD ONNX
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, DiffusionPipeline # SDXL
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
import os, sys, time
import json
import nmkdiffusers_load

def get_pnginfo(argdict):
    argdict_copy = dict(argdict)
    argdict_copy["model"] = os.path.basename(argdict_copy.get("model")) # For privacy, only store model name instead of full path
    info = PngImagePlugin.PngInfo()
    info.add_text('Nmkdiffusers', json.dumps(argdict_copy, separators = (',', ':')))
    return info

def generate_ip2p(argdict, outpath):
    model = argdict.get("model")
    nmkdiffusers_load.load_ip2p(model)
    sampler = argdict.get("sampler")
    nmkdiffusers_load.set_scheduler(sampler)
    prompt = argdict.get("prompt")
    prompt_neg = argdict.get("promptNeg")
    init_img_path = argdict.get("initImg")
    steps = int(argdict.get("steps") or 15)
    seed = int(argdict.get("seed") or 0)
    cfg_txt = float(argdict.get("scaleTxt") or 7.0)
    cfg_img = float(argdict.get("scaleImg") or 1.5)
    print(f'Generating (IP2P) - Prompt: {prompt} - Neg Prompt: {prompt_neg} - Steps: {steps} - Seed: {seed} - Text Scale {cfg_txt} - Image Scale {cfg_img}')
    start_time = time.time()
    rng = torch.manual_seed(seed)
    image = Image.open(init_img_path)
    image = PIL.ImageOps.exif_transpose(image).convert("RGB")
    image = nmkdiffusers_load.pipe(prompt, negative_prompt = prompt_neg, image = image, num_inference_steps = steps, guidance_scale = cfg_txt, image_guidance_scale = cfg_img, generator = rng).images[0]
    image.save(os.path.join(outpath, f"{time.time_ns()}.png"), 'PNG', pnginfo = get_pnginfo(argdict))
    print(f'Image generated in {(time.time() - start_time):.2f}s')
    image = None

def generate_sd_onnx(argdict, outpath):
    mode = argdict.get("mode")
    model = argdict.get("model")
    nmkdiffusers_load.load_sd_onnx(model, mode)
    sampler = argdict.get("sampler")
    nmkdiffusers_load.set_scheduler(sampler)
    prompt = argdict.get("prompt")
    prompt_neg = argdict.get("promptNeg")
    init_img_path = argdict.get("initImg")
    mask_img_path = argdict.get("inpaintMask")
    steps = int(argdict.get("steps") or 15)
    seed = int(argdict.get("seed") or 0)
    scale = float(argdict.get("scaleTxt") or 7.0)
    width = int(argdict.get("w") or 512)
    height = int(argdict.get("h") or 512)
    init_strength = float(argdict.get("initStrength") or 0.0)
    print(f'Generating (ONNX {mode}) - Prompt: {prompt} - Neg Prompt: {prompt_neg} - Steps: {steps} - Seed: {seed} - Scale {scale} - Res {width}x{height}')
    start_time = time.time()
    seed = int(seed)
    rng = np.random.RandomState(seed)
    eta = 0.0
    if mode == "txt2img":
        image = nmkdiffusers_load.pipe(prompt = prompt, height = height, width = width, num_inference_steps = steps, guidance_scale = scale, negative_prompt = prompt_neg, generator = rng).images[0]
    if mode == "img2img":
        img = Image.open(init_img_path).convert('RGB')
        image = nmkdiffusers_load.pipe(prompt = prompt, image = img, num_inference_steps = steps, guidance_scale = scale, negative_prompt = prompt_neg, eta = eta, strength = init_strength, generator = rng).images[0]
    if mode == "inpaint":
        img = Image.open(init_img_path).convert('RGB')
        mask = Image.open(mask_img_path)
        image = nmkdiffusers_load.pipe(prompt = prompt, image = img, mask_image = mask, height = height, width = width, num_inference_steps = steps, guidance_scale = scale, negative_prompt = prompt_neg, eta = eta, generator = rng).images[0]
    if mode == "inpaint-legacy":
        img = Image.open(init_img_path).convert('RGB')
        mask = Image.open(mask_img_path)
        image = nmkdiffusers_load.pipe(prompt = prompt, image = img, mask_image = mask, num_inference_steps = steps, guidance_scale = scale, negative_prompt = prompt_neg, eta = eta, strength = init_strength, generator = rng).images[0]

    image.save(os.path.join(outpath, f"{time.time_ns()}.png"), 'PNG', pnginfo = get_pnginfo(argdict))
    print(f'Image generated in {(time.time() - start_time):.2f}s')
    image = None

def generate_sd_xl(argdict, outpath, opt_load, seq_load):
    model = argdict.get("model")
    refiner = argdict.get("modelRefiner")
    mode = argdict.get("mode")
    nmkdiffusers_load.load_sdxl(model, refiner if not seq_load else "", mode, opt_load) # Load base+refiner (or only base, if seq_load is True)
    sampler = argdict.get("sampler")
    nmkdiffusers_load.set_scheduler(sampler)
    prompt = argdict.get("prompt")
    prompt_neg = argdict.get("promptNeg")
    init_img_path = argdict.get("initImg")
    mask_img_path = argdict.get("inpaintMask")
    steps = int(argdict.get("steps") or 20)
    seed = int(argdict.get("seed") or 0)
    scale = float(argdict.get("scaleTxt") or 7.0)
    width = int(argdict.get("w") or 1024)
    height = int(argdict.get("h") or 1024)
    init_strength = float(argdict.get("initStrength") or 0.0)
    print(f'Generating (SDXL {mode}) - Prompt: {prompt} - Neg Prompt: {prompt_neg} - Steps: {steps} - Seed: {seed} - Scale {scale} - Res {width}x{height}')
    start_time = time.time()
    seed = int(seed)
    refine_frac = float(argdict.get("refineFrac") or 1.0)
    g = torch.Generator()
    g.manual_seed(seed)
    do_refine = refiner and refine_frac < 0.999
    refine_frac = refine_frac if do_refine else 1.0
    print(f'SDXL: Using refine_frac = {refine_frac}')
    base_img_type = "latent" if do_refine else "pil"
    print(f'SDXL: Running base model [{mode}]')
    # Generate
    if mode == "txt2img":
        image = nmkdiffusers_load.pipe(prompt = prompt, height = height, width = width, num_inference_steps = steps, guidance_scale = scale, negative_prompt = prompt_neg, generator = g, output_type = base_img_type, denoising_end = refine_frac).images[0]
    if mode == "img2img":
        img = Image.open(init_img_path).convert('RGB')
        image = nmkdiffusers_load.pipe(prompt = prompt, image = img, strength = init_strength, num_inference_steps = steps, guidance_scale = scale, negative_prompt = prompt_neg, generator = g, output_type = base_img_type).images[0]
    if mode == "inpaint":
        img = Image.open(init_img_path).convert('RGB')
        mask = Image.open(mask_img_path)
        image = nmkdiffusers_load.pipe(prompt = prompt, image = img, mask_image = mask, strength = init_strength, height = img.height, width = img.width, num_inference_steps = steps, guidance_scale = scale, negative_prompt = prompt_neg, generator = g, output_type = base_img_type).images[0]
    
    if seq_load:
        nmkdiffusers_load.unload(True, False) # Unload base model
        nmkdiffusers_load.load_sdxl("", refiner, mode, opt_load) # Load refiner
        nmkdiffusers_load.set_scheduler(sampler)
    
    # Refine (optional)
    if do_refine:
        print(f'SDXL: Running refine model @ {refine_frac}')
        image = nmkdiffusers_load.refiner(prompt = prompt, num_inference_steps = steps, denoising_start = refine_frac, guidance_scale = scale, negative_prompt = prompt_neg, generator = g, image = image[None, :]).images[0]
    # Add metadata and save
    image.save(os.path.join(outpath, f"{time.time_ns()}.png"), 'PNG', pnginfo = get_pnginfo(argdict))
    print(f'Image generated in {(time.time() - start_time):.2f}s')
    image = None
    