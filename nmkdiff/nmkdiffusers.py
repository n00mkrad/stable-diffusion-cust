import PIL
from PIL import PngImagePlugin, Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler # InstructPix2Pix
from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline, OnnxStableDiffusionInpaintPipeline, OnnxStableDiffusionInpaintPipelineLegacy # SD ONNX
import numpy as np
import argparse
import os, sys, time
import json
import threading
import queue
import traceback

os.chdir(sys.path[0])


parser = argparse.ArgumentParser()

parser.add_argument(
    "-p",
    "--pipeline",
    type=str,
    help="Diffusers Pipeline to run",
    choices=["InstructPix2Pix", "SdOnnx"],
    dest="pipeline",
)
parser.add_argument(
    "-g",
    "--generation_mode",
    type=str,
    help="Image generation mode",
    choices=["txt2img", "img2img", "inpaint", "inpaint-legacy"],
    dest="generation_mode",
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    help="Custom model folder path",
    dest="model_path",
)
parser.add_argument(
    "-o",
    "--outpath",
    type=str,
    help="Output path",
    dest="outpath",
)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

args = parser.parse_args()


stdin_queue = queue.Queue()

def read_stdin():
    while True:
        message = sys.stdin.readline().strip()

        if not message:
            time.sleep(0.01)

        if message == "stop":
            stdin_queue.queue.clear()

        if message == "kill":
            os._exit(0)

        stdin_queue.put(message)

stdin_thread = threading.Thread(target=read_stdin)
stdin_thread.start()

pipe = None

def load_ip2p():
    global pipe
    model_id = "timbrooks/instruct-pix2pix"
    from huggingface_hub import snapshot_download
    
    if not args.model_path:
        ignore = ["*.ckpt", "*.safetensors", "safety_checker/*", "*.md", ".git*", "*.png", "*.pt"]
        rev = "fp16"
        try:
            args.model_path = snapshot_download(repo_id=model_id, revision=rev, ignore_patterns=ignore)
        except:
            args.model_path = snapshot_download(repo_id=model_id, revision=rev, ignore_patterns=ignore, local_files_only=True)
    
    print(f"Trying to load model from '{args.model_path}'", flush=True)
    
    try:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, safety_checker=None)
    except:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, safety_checker=None, local_files_only=True)
    
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()

def load_sd_onnx():
    global pipe
    prov = "DmlExecutionProvider"
    if args.generation_mode == "txt2img":
        pipe = OnnxStableDiffusionPipeline.from_pretrained(args.model_path, provider=prov, safety_checker=None)
    if args.generation_mode == "img2img":
        pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(args.model_path, provider=prov, safety_checker=None)
    if args.generation_mode == "inpaint":
        pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(args.model_path, provider=prov, safety_checker=None)
    if args.mode == "inpaint-legacy":
        pipe = OnnxStableDiffusionInpaintPipelineLegacy.from_pretrained(args.model_path, provider=prov, safety_checker=None)

def generate_ip2p(inpath, outpath, prompt, prompt_neg, steps, seed, cfg_txt, cfg_img):
    global pipe
    print(f'Generating ({args.pipeline}) - Prompt: {prompt} - Neg Prompt: {prompt_neg} - Steps: {steps} - Seed: {seed} - Text Scale {cfg_txt} - Image Scale {cfg_img}')
    start_time = time.time()
    rng = torch.manual_seed(seed)
    info = PngImagePlugin.PngInfo()
    image = Image.open(inpath)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image
    image = pipe(prompt, negative_prompt = prompt_neg, image=image, num_inference_steps=steps, guidance_scale=cfg_txt, image_guidance_scale=cfg_img, generator=rng).images[0]
    metadataDict = {"mode": args.generation_mode, "prompt": prompt, "promptNeg": prompt_neg, "initImg": inpath, "steps": steps, "seed": seed, "scaleTxt": cfg_txt, "scaleImg": cfg_img}
    info.add_text('Nmkdiffusers',  json.dumps(metadataDict, separators=(',', ':')))
    image.save(os.path.join(outpath, f"{time.time_ns()}.png"), 'PNG', pnginfo=info)
    print(f'Image generated in {(time.time() - start_time):.2f}s', flush=True)
    image = None

def generate_sd_onnx(prompt, prompt_neg, outpath, steps, width, height, seed, scale, init_img_path = None, init_strength = 0.75, mask_img_path = None):
    global pipe
    print(f'Generating ({args.pipeline}- {args.generation_mode}) - Prompt: {prompt} - Neg Prompt: {prompt_neg} - Steps: {steps} - Seed: {seed} - Scale {scale} - Res {width}x{height}')
    start_time = time.time()
    seed = int(seed)
    rng = np.random.RandomState(seed)
    info = PngImagePlugin.PngInfo()
    metadataDict = {"mode": args.generation_mode, "prompt": prompt, "promptNeg": prompt_neg, "initImg": init_img_path, "initStrength": init_strength, "w": width, "h": height, "steps": steps, "seed": seed, "scaleTxt": scale, "inpaintMask": mask_img_path}
    info.add_text('NmkdInstructPixToPix',  json.dumps(metadataDict, separators=(',', ':')))
    eta = 0.0
    if args.generation_mode == "txt2img":
        image=pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, generator=rng).images[0]
    if args.generation_mode == "img2img":
        img=Image.open(init_img_path).convert('RGB')
        image=pipe(prompt=prompt, image=img, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, strength=init_strength, generator=rng).images[0]
    if args.generation_mode == "inpaint":
        img=Image.open(init_img_path).convert('RGB')
        mask=Image.open(mask_img_path)
        image=pipe(prompt=prompt, image=img, mask_image = mask, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, generator=rng).images[0]
    if args.mode == "inpaint-legacy":
        img=Image.open(init_img_path).convert('RGB')
        mask=Image.open(mask_img_path)
        image=pipe(prompt=prompt, image=img, mask_image = mask, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, strength=init_strength, generator=rng).images[0]

    info.add_text('Nmkdiffusers',  json.dumps(metadataDict, separators=(',', ':')))
    image.save(os.path.join(outpath, f"{time.time_ns()}.png"), 'PNG', pnginfo=info)
    print(f'Image generated in {(time.time() - start_time):.2f}s', flush=True)
    image = None

def generate_from_json(argdict):
    prompt = argdict.get("prompt")
    prompt_neg = argdict.get("promptNeg")
    inpath = argdict.get("initImg")
    steps = int(argdict.get("steps") or 0)
    seed = int(argdict.get("seed") or 0)
    cfg_txt = float(argdict.get("scaleTxt") or 0.0)
    cfg_img = float(argdict.get("scaleImg") or 0.0)
    w = int(argdict.get("w") or 0)
    h = int(argdict.get("h") or 0)
    init_strength = float(argdict.get("initStrength") or 0.0)
    mask_path = argdict.get("inpaintMask")
    if args.pipeline == "InstructPix2Pix":
        generate_ip2p(inpath, args.outpath, prompt, prompt_neg, steps, seed, cfg_txt, cfg_img)
    if args.pipeline == "SdOnnx":
        generate_sd_onnx(prompt, prompt_neg, args.outpath, steps, w, h, seed, cfg_txt, inpath, init_strength, mask_path)

print(f"Loading...", flush=True)
if args.pipeline == "InstructPix2Pix":
    load_ip2p()
if args.pipeline == "SdOnnx":
    load_sd_onnx()
print(f"Model loaded.", flush=True)

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
            
        if cmd == "stop":
            print(f"Stopped.", flush=True)

        if cmd == "exit":
            os._exit(0)
            
    except queue.Empty:
        if not stdin_thread.is_alive():
            print(f"Breaking because queue empty and stdin thread not alive", flush=True)
            break

    except Exception as ex:
        print(f"Exception: {str(ex)}", flush=True)
        traceback.print_stack()
        break

pipe = None

print(f"Exiting...", flush=True)
os._exit(0)