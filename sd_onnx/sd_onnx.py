import os
import sys
import time
import torch
import numpy as np
from diffusers import OnnxStableDiffusionPipeline
from diffusers import OnnxStableDiffusionImg2ImgPipeline
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
    "-o",
    "--outpath",
    type=str,
    help="Output path",
    dest='outpath',
)
parser.add_argument(
    "-i",
    "--img2img",
    action="store_true",
    help="Enable img2img code",
    dest='img2img',
)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

opt = parser.parse_args()


eta = 0.0
prov = "DmlExecutionProvider"
# pipe = None

if opt.img2img:
    pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(opt.mdlpath, provider=prov, revision="fp16", torch_dtype=torch.float16)
else:
    pipe = OnnxStableDiffusionPipeline.from_pretrained(opt.mdlpath, provider=prov, revision="fp16", torch_dtype=torch.float16)


def generate(prompt, prompt_neg, steps, width, height, seed, scale, init_img_path = None, init_strength = 0.75):
    start_time = time.time()
    
    generator = torch.Generator()
    seed = int(seed)
    generator = generator.manual_seed(seed)
    latents = torch.randn(
        (1, 4, height // 8, width // 8),
        generator = generator
    )
    
    info = PngImagePlugin.PngInfo()
    neg_prompt_meta_text = "" if prompt_neg == "" else f' [{prompt_neg}]'
    
    if opt.img2img:
        img=Image.open(init_img_path)
        image=pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, prompt_neg=prompt_neg, eta=eta, latents=latents, execution_provider=prov, init_image=img, strength=init_strength).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale} -I {init_img_path} -f {init_strength}')
    else:
        image=pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, prompt_neg=prompt_neg, eta=eta, latents=latents, execution_provider=prov).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale}')
    
    image.save(os.path.join(opt.outpath, f"{time.time_ns()}.png"), 'PNG', pnginfo=info)
    
    print(f'Image generated in {(time.time() - start_time):.2f}s')
    image = None


print(f'Model loaded.')

f = open(opt.jsonpath)
data = json.load(f)

for i in range(len(data)):
    argdict = data[i]
    print(f'Generating {i+1}/{len(data)}: "{argdict["prompt"]}" - {argdict["steps"]} Steps - Scale {argdict["scale"]} - {argdict["w"]}x{argdict["h"]}')
    generate(argdict["prompt"], argdict["negprompt"], int(argdict["steps"]), int(argdict["w"]), int(argdict["h"]), argdict["seed"], float(argdict["scale"]), argdict["initImg"], float(argdict["initStrength"]))

pipe = None