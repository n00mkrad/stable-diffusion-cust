import os
import sys
import time
import torch
import numpy as np
from diffusers import OnnxStableDiffusionPipeline
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
    type=str,
    help="Enable img2img code",
    dest='img2img',
)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

opt = parser.parse_args()

eta=0.0

if opt.img2img:
    pipe = OnnxStableDiffusionPipeline.from_pretrained(opt.mdlpath, provider="DmlExecutionProvider", revision="fp16", torch_dtype=torch.float16)
else:
    pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(opt.mdlpath, provider="DmlExecutionProvider", revision="fp16", torch_dtype=torch.float16)

def txt_to_img(prompt, negative_prompt, steps, width, height, seed, scale):
    start_time = time.time()
    
    generator = torch.Generator()
    seed = int(seed)
    generator = generator.manual_seed(seed)
    latents = torch.randn(
        (1, 4, height // 8, width // 8),
        generator = generator
    )
    
    image = pipe(prompt, height, width, steps, scale, negative_prompt, eta, latents = latents, execution_provider="DmlExecutionProvider").images[0]
    
    info = PngImagePlugin.PngInfo()
    neg_prompt_meta_text = "" if negative_prompt == "" else f' [{negative_prompt}]'
    info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale}')
    image.save(os.path.join(opt.outpath, f"{time.time_ns()}.png"), 'PNG', pnginfo=info)
    
    image = None
    print(f'Image generated in {(time.time() - start_time):.2f}s')

print(f'Model loaded.')

f = open(opt.jsonpath)
data = json.load(f)

for i in range(len(data)):
    argdict = data[i]
    print(f'Generating {i+1}/{len(data)}: "{argdict["prompt"]}" - {argdict["steps"]} Steps - Scale {argdict["scale"]} - {argdict["w"]}x{argdict["h"]}')
    txt_to_img(argdict["prompt"], argdict["negprompt"], int(argdict["steps"]), int(argdict["w"]), int(argdict["h"]), argdict["seed"], float(argdict["scale"]))

pipe = None