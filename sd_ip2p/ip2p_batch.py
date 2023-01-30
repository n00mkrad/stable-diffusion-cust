import PIL
from PIL import PngImagePlugin, Image
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

import argparse
import os, sys, time
import random
import numpy as np
import string
import json

os.chdir(sys.path[0])

parser = argparse.ArgumentParser()

parser.add_argument(
    "-j",
    "--json",
    type=str,
    help="Path to command list JSON",
    dest='jsonpath',
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    help="Custom model folder path",
    dest='modeldir',
)
parser.add_argument(
    "-o",
    "--outpath",
    type=str,
    help="Output path",
    dest='outpath',
)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

args = parser.parse_args()


model_id = "timbrooks/instruct-pix2pix"
from huggingface_hub import snapshot_download

if not args.modeldir:
    ignore = ["*.ckpt", "*.safetensors", "safety_checker/*", ".md", ".git*"]
    rev = "fp16"
    try:
        args.modeldir = snapshot_download(repo_id=model_id, revision=rev, ignore_patterns=ignore)
    except:
        args.modeldir = snapshot_download(repo_id=model_id, revision=rev, ignore_patterns=ignore, local_files_only=True)

print(f"Trying to load model from '{args.modeldir}'", flush=True)

try:
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.modeldir, torch_dtype=torch.float16, safety_checker=None)
except:
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.modeldir, torch_dtype=torch.float16, safety_checker=None, local_files_only=True)

pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()


def generate(inpath, prompt, prompt_neg, seed, cfg_txt, cfg_img):
    start_time = time.time()
    
    print(f"Using seed {seed}", flush=True)
    rng = torch.manual_seed(seed)
    
    info = PngImagePlugin.PngInfo()
    
    image = Image.open(inpath)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image
    
    image = pipe(prompt, negative_prompt = prompt_neg, image=image, num_inference_steps=steps, guidance_scale=cfg_txt, image_guidance_scale=cfg_img, generator=rng).images[0]
    metadataDict = {"prompt": prompt, "image": inpath, "prompt_neg": prompt_neg, "steps": steps, "seed": seed, "cfg_txt": cfg_txt, "cfg_img": cfg_img}
    info.add_text('NmkdInstructPixToPix',  json.dumps(metadataDict, separators=(',', ':')))
    image.save(os.path.join(args.outpath, f"{time.time_ns()}.png"), 'PNG', pnginfo=info)
    print(f'Image generated in {(time.time() - start_time):.2f}s')
    image = None

print(f'Model loaded.')

data = json.load(open(args.jsonpath, encoding='utf-8'))

for i in range(len(data)):
    argdict = data[i]
    inpath = argdict["initImg"]
    prompt = argdict["prompt"]
    prompt_neg = argdict["prompt_neg"]
    steps = int(argdict["steps"])
    seed = int(argdict["seed"])
    cfg_txt = float(argdict["scale_txt"])
    cfg_img = float(argdict["scale_img"])
    print(f'Source Image: {inpath}')
    print(f'Generating {i+1}/{len(data)}: Prompt: {prompt} - Neg Prompt: {prompt_neg} - Steps: {steps} - Seed: {seed} - Text Scale {cfg_txt} - Image Scale {cfg_img}')
    generate(inpath, prompt, prompt_neg, seed, cfg_txt, cfg_img)

pipe = None


