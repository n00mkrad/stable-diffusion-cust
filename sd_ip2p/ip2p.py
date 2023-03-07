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
import threading
import queue

os.chdir(sys.path[0])

parser = argparse.ArgumentParser()

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

stdin_queue = queue.Queue()

# Read messages from stdin and add them to the queue
def read_stdin():
    while True: # TODO: Check if this loop causes performance issues as it has no wait times
        message = sys.stdin.readline().strip()
        if not message:
            time.sleep(0.01)
        if message == "kill":
            sys.exit()
        stdin_queue.put(message)
        # print(f"Queueing message ({stdin_queue.qsize()})", flush=True)

# Start the thread to read from stdin
stdin_thread = threading.Thread(target=read_stdin)
stdin_thread.start()


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

print(f'Model loaded.')

def generate(inpath, outpath, prompt, prompt_neg, steps, seed, cfg_txt, cfg_img):
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
    image.save(os.path.join(outpath, f"{time.time_ns()}.png"), 'PNG', pnginfo=info)
    print(f'Image generated in {(time.time() - start_time):.2f}s', flush=True)
    image = None

def generate_from_json(argdict):
    inpath = argdict["initImg"]
    prompt = argdict["prompt"]
    prompt_neg = argdict["prompt_neg"]
    steps = int(argdict["steps"])
    seed = int(argdict["seed"])
    cfg_txt = float(argdict["scale_txt"])
    cfg_img = float(argdict["scale_img"])
    # print(f'Using Image: {inpath}')
    print(f'Generating - Prompt: {prompt} - Neg Prompt: {prompt_neg} - Steps: {steps} - Seed: {seed} - Text Scale {cfg_txt} - Image Scale {cfg_img}')
    generate(inpath, args.outpath, prompt, prompt_neg, steps, seed, cfg_txt, cfg_img)

# Process messages from the queue
while True:
    try:
        # if stdin_queue.empty:
        #     time.sleep(0.01)
        #     continue

        message = stdin_queue.get(block=True, timeout=1)
        split = message.split()
        cmd = split[0]
        cmd_args = ' '.join(split[1:])
        
        if cmd == "generate":
            data = json.loads(cmd_args)
            generate_from_json(data)
            
        if cmd == "exit":
            os._exit(0)
            
    except queue.Empty:
        if not stdin_thread.is_alive():
            print(f"Breaking because queue empty and stdin thread not alive", flush=True)
            break

    except Exception as ex:
        print(f"Exception: {str(ex)}", flush=True)
        break

pipe = None

print(f"Exiting...", flush=True)
os._exit(0)