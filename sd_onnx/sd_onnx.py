import os
import sys
import time
import torch
import numpy as np
from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline, OnnxStableDiffusionInpaintPipeline, OnnxStableDiffusionInpaintPipelineLegacy, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler

import argparse
import json
from PIL import PngImagePlugin, Image

os.chdir(sys.path[0])

parser = argparse.ArgumentParser()

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="Path to model checkpoint file",
    dest='mdlpath',
)
parser.add_argument(
    "-j",
    "--json",
    type=str,
    help="Path to command list JSON",
    dest='jsonpath',
)
parser.add_argument(
    "-s",
    "--sampler",
    type=str,
    help="Output path",
    dest='outpath',
)
parser.add_argument(
    "-o",
    "--outpath",
    type=str,
    help="Output path",
    dest='outpath',
)
parser.add_argument(
    "-mode",
    "--mode",
    choices=['txt2img', 'img2img', 'inpaint', 'inpaint-legacy'],
    default="txt2img",
    help="Specify generation mode",
    dest='mode',
)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

opt = parser.parse_args()


eta = 0.0
prov = "DmlExecutionProvider"

if opt.mode == "txt2img":
    pipe = OnnxStableDiffusionPipeline.from_pretrained(opt.mdlpath, provider=prov, safety_checker=None)
if opt.mode == "img2img":
    pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(opt.mdlpath, provider=prov, safety_checker=None)
if opt.mode == "inpaint":
    pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(opt.mdlpath, provider=prov, safety_checker=None)
if opt.mode == "inpaint-legacy":
    pipe = OnnxStableDiffusionInpaintPipelineLegacy.from_pretrained(opt.mdlpath, provider=prov, safety_checker=None)


def generate(prompt, prompt_neg, steps, width, height, seed, scale, init_img_path = None, init_strength = 0.75, mask_img_path = None):
    start_time = time.time()
    
    seed = int(seed)
    rng = np.random.RandomState(seed)
    print(f"Set seed to {seed}", flush=True)
    
    info = PngImagePlugin.PngInfo()
    neg_prompt_meta_text = "" if prompt_neg == "" else f' [{prompt_neg}]'
        
    if opt.mode == "txt2img":
        print("txt2img", flush=True)
        image=pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, generator=rng).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale}')
    if opt.mode == "img2img":
        print("img2img", flush=True)
        img=Image.open(init_img_path)
        image=pipe(prompt=prompt, image=img, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, strength=init_strength, generator=rng).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale} -I {init_img_path} -f {init_strength}')
    if opt.mode == "inpaint":
        print("inpaint", flush=True)
        img=Image.open(init_img_path)
        mask=Image.open(mask_img_path)
        image=pipe(prompt=prompt, image=img, mask_image = mask, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, generator=rng).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale} -I {init_img_path} -f 0.0 -M {mask_img_path}')
    if opt.mode == "inpaint-legacy":
        print("inpaint legacy", flush=True)
        img=Image.open(init_img_path).convert('RGB')
        mask=Image.open(mask_img_path)
        image=pipe(prompt=prompt, image=img, mask_image = mask, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, strength=init_strength, generator=rng).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale} -I {init_img_path} -f {init_strength} -M {mask_img_path}')

    
    image.save(os.path.join(opt.outpath, f"{time.time_ns()}.png"), 'PNG', pnginfo=info)
    
    print(f'Image generated in {(time.time() - start_time):.2f}s')
    image = None


print(f'Model loaded.')

f = open(opt.jsonpath)
data = json.load(f)

for i in range(len(data)):
    argdict = data[i]
    print(f'Generating {i+1}/{len(data)}: "{argdict["prompt"]}" - {argdict["steps"]} Steps - Scale {argdict["scale"]} - {argdict["w"]}x{argdict["h"]}')
    generate(argdict["prompt"], argdict["negprompt"], int(argdict["steps"]), int(argdict["w"]), int(argdict["h"]), argdict["seed"], float(argdict["scale"]), argdict["initImg"], float(argdict["initStrength"]), argdict["inpaintMask"])

pipe = None