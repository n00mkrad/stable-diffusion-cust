import functools; print = functools.partial(print, flush = True)
import PIL
from PIL import PngImagePlugin, Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline # InstructPix2Pix
from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline, OnnxStableDiffusionInpaintPipeline, OnnxStableDiffusionInpaintPipelineLegacy # SD ONNX
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, DiffusionPipeline # SDXL
from diffusers import DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, HeunDiscreteScheduler, EulerDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler, DPMSolverMultistepScheduler # Samplers
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
import argparse
import os, sys, time
import json
import threading
import queue
import traceback

from nmkdiffusers_load import load_sdxl

os.chdir(sys.path[0])


parser = argparse.ArgumentParser()

parser.add_argument(
    "-p",
    "--pipeline",
    type = str,
    help = "Diffusers Pipeline to run",
    choices = ["InstructPix2Pix", "SdOnnx", "SdXl"],
    dest = "pipeline",
)
parser.add_argument(
    "-g",
    "--generation_mode",
    type = str,
    help = "Image generation mode",
    choices = ["txt2img", "img2img", "inpaint", "inpaint-legacy"],
    dest = "generation_mode",
)
parser.add_argument(
    "-m",
    "--model_path",
    type = str,
    help = "Custom model path",
    dest = "model_path",
)
parser.add_argument(
    "-m2",
    "--model_path_refiner",
    type = str,
    help = "Refiner model path",
    dest = "model_path_refiner",
)
parser.add_argument(
    "--sdxl_optimize",
    action = "store_true",
    help = "Reduce SD XL VRAM usage at the cost of some inference speed",
    dest = "sdxl_opt",
)
parser.add_argument(
    "-o",
    "--outpath",
    type = str,
    help = "Output path",
    dest = "outpath",
)

if len(sys.argv) == 1:
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

stdin_thread = threading.Thread(target = read_stdin)
stdin_thread.start()

# do_refine = args.model_path_refiner is not None and os.path.exists(args.model_path_refiner)
pipe = None
refiner = None

def set_scheduler(sampler_name):
    if sampler_name == "ddim": sched = DDIMScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "plms": sched = PNDMScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "lms": sched = LMSDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "heun": sched = HeunDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "euler": sched = EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "k_euler": sched = EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = True)
    if sampler_name == "euler_a": sched = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "dpm_2": sched = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "dpm_2_a": sched = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "dpmpp_2s": sched = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "dpmpp_2m": sched = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "k_dpmpp_2m": sched = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = True)
    if pipe is not None:
        print(f"Set base scheduler to {sampler_name}")
        pipe.scheduler = sched
    if refiner is not None:
        print(f"Set refiner scheduler to {sampler_name}")
        refiner.scheduler = sched

def load_ip2p():
    global pipe
    model_id = "timbrooks/instruct-pix2pix"
    from huggingface_hub import snapshot_download
    
    if not args.model_path:
        ignore = ["*.ckpt", "*.safetensors", "safety_checker/*", "*.md", ".git*", "*.png", "*.pt"]
        rev = "fp16"
        try:
            args.model_path = snapshot_download(repo_id = model_id, revision = rev, ignore_patterns = ignore)
        except:
            args.model_path = snapshot_download(repo_id = model_id, revision = rev, ignore_patterns = ignore, local_files_only = True)
    
    print(f"Trying to load model from '{args.model_path}'")
    
    try:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model_path, torch_dtype = torch.float16, safety_checker = None)
    except:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model_path, torch_dtype = torch.float16, safety_checker = None, local_files_only = True)
    
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()

def load_sd_onnx():
    global pipe
    prov = "DmlExecutionProvider"
    if args.generation_mode == "txt2img":
        pipe = OnnxStableDiffusionPipeline.from_pretrained(args.model_path, provider = prov, safety_checker = None)
    if args.generation_mode == "img2img":
        pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(args.model_path, provider = prov, safety_checker = None)
    if args.generation_mode == "inpaint":
        pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(args.model_path, provider = prov, safety_checker = None)
    if args.generation_mode == "inpaint-legacy":
        pipe = OnnxStableDiffusionInpaintPipelineLegacy.from_pretrained(args.model_path, provider = prov, safety_checker = None)

def generate_ip2p(inpath, outpath, prompt, prompt_neg, steps, sampler, seed, cfg_txt, cfg_img):
    global pipe
    set_scheduler(sampler)
    print(f'Generating ({args.pipeline}) - Prompt: {prompt} - Neg Prompt: {prompt_neg} - Steps: {steps} - Seed: {seed} - Text Scale {cfg_txt} - Image Scale {cfg_img}')
    start_time = time.time()
    rng = torch.manual_seed(seed)
    info = PngImagePlugin.PngInfo()
    image = Image.open(inpath)
    image = PIL.ImageOps.exif_transpose(image).convert("RGB")
    image = pipe(prompt, negative_prompt = prompt_neg, image = image, num_inference_steps = steps, guidance_scale = cfg_txt, image_guidance_scale = cfg_img, generator = rng).images[0]
    metadataDict = {"mode": args.generation_mode, "prompt": prompt, "promptNeg": prompt_neg, "initImg": inpath, "steps": steps, "seed": seed, "scaleTxt": cfg_txt, "scaleImg": cfg_img}
    info.add_text('Nmkdiffusers',  json.dumps(metadataDict, separators = (',', ':')))
    image.save(os.path.join(outpath, f"{time.time_ns()}.png"), 'PNG', pnginfo = info)
    print(f'Image generated in {(time.time() - start_time):.2f}s')
    image = None

def generate_sd_onnx(prompt, prompt_neg, outpath, steps, width, height, sampler, seed, scale, init_img_path = None, init_strength = 0.75, mask_img_path = None):
    global pipe
    set_scheduler(sampler)
    print(f'Generating ({args.pipeline} {args.generation_mode}) - Prompt: {prompt} - Neg Prompt: {prompt_neg} - Steps: {steps} - Seed: {seed} - Scale {scale} - Res {width}x{height}')
    start_time = time.time()
    seed = int(seed)
    rng = np.random.RandomState(seed)
    info = PngImagePlugin.PngInfo()
    metadataDict = {"mode": args.generation_mode, "prompt": prompt, "promptNeg": prompt_neg, "initImg": init_img_path, "initStrength": init_strength, "w": width, "h": height, "steps": steps, "seed": seed, "scaleTxt": scale, "inpaintMask": mask_img_path}
    eta = 0.0
    if args.generation_mode == "txt2img":
        image = pipe(prompt = prompt, height = height, width = width, num_inference_steps = steps, guidance_scale = scale, negative_prompt = prompt_neg, generator = rng).images[0]
    if args.generation_mode == "img2img":
        img = Image.open(init_img_path).convert('RGB')
        image = pipe(prompt = prompt, image = img, num_inference_steps = steps, guidance_scale = scale, negative_prompt = prompt_neg, eta = eta, strength = init_strength, generator = rng).images[0]
    if args.generation_mode == "inpaint":
        img = Image.open(init_img_path).convert('RGB')
        mask = Image.open(mask_img_path)
        image = pipe(prompt = prompt, image = img, mask_image = mask, height = height, width = width, num_inference_steps = steps, guidance_scale = scale, negative_prompt = prompt_neg, eta = eta, generator = rng).images[0]
    if args.generation_mode == "inpaint-legacy":
        img = Image.open(init_img_path).convert('RGB')
        mask = Image.open(mask_img_path)
        image = pipe(prompt = prompt, image = img, mask_image = mask, num_inference_steps = steps, guidance_scale = scale, negative_prompt = prompt_neg, eta = eta, strength = init_strength, generator = rng).images[0]

    info.add_text('Nmkdiffusers',  json.dumps(metadataDict, separators = (',', ':')))
    image.save(os.path.join(outpath, f"{time.time_ns()}.png"), 'PNG', pnginfo = info)
    print(f'Image generated in {(time.time() - start_time):.2f}s')
    image = None
    
def generate_sd_xl(mode, prompt, prompt_neg, outpath, steps, width, height, sampler, seed, scale, init_img_path = None, init_strength = 0.75, mask_img_path = None, refine_frac = 0.7):
    global pipe
    global refiner
    set_scheduler(sampler)
    print(f'Generating ({args.pipeline} {mode}) - Prompt: {prompt} - Neg Prompt: {prompt_neg} - Steps: {steps} - Seed: {seed} - Scale {scale} - Res {width}x{height}')
    start_time = time.time()
    seed = int(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    info = PngImagePlugin.PngInfo()
    metadataDict = {"mode": mode, "prompt": prompt, "promptNeg": prompt_neg, "initImg": init_img_path, "initStrength": init_strength, "w": width, "h": height, "steps": steps, "seed": seed, "scaleTxt": scale, "inpaintMask": mask_img_path, "sampler": sampler, "refineFrac": refine_frac}
    info.add_text('Nmkdiffusers',  json.dumps(metadataDict, separators = (',', ':')))
    do_refine = refiner is not None and refine_frac < 0.999
    refine_frac = refine_frac if do_refine else 1.0
    print(f'SDXL: Using refine_frac = {refine_frac}')
    base_img_type = "latent" if do_refine else "pil"
    print(f'SDXL: Running base model [{mode}]')
    # Generate
    if mode == "txt2img":
        image = pipe(prompt = prompt, height = height, width = width, num_inference_steps = steps, guidance_scale = scale, negative_prompt = prompt_neg, generator = g, output_type = base_img_type, denoising_end = refine_frac).images[0]
    if mode == "img2img":
        img = Image.open(init_img_path).convert('RGB')
        image = pipe(prompt = prompt, image = img, strength = init_strength, num_inference_steps = steps, guidance_scale = scale, negative_prompt = prompt_neg, generator = g, output_type = base_img_type).images[0]
    if mode == "inpaint":
        img = Image.open(init_img_path).convert('RGB')
        mask = Image.open(mask_img_path)
        image = pipe(prompt = prompt, image = img, mask_image = mask, strength = init_strength, height = img.height, width = img.width, num_inference_steps = steps, guidance_scale = scale, negative_prompt = prompt_neg, generator = g, output_type = base_img_type).images[0]
    # Refine (optional)
    if do_refine:
        print(f'SDXL: Running refine model @ {refine_frac}')
        image = refiner(prompt = prompt, num_inference_steps = steps, denoising_start = refine_frac, guidance_scale = scale, negative_prompt = prompt_neg, generator = g, image = image[None, :]).images[0]
    # Add metadata and save
    info.add_text('Nmkdiffusers',  json.dumps(metadataDict, separators = (',', ':')))
    image.save(os.path.join(outpath, f"{time.time_ns()}.png"), 'PNG', pnginfo = info)
    print(f'Image generated in {(time.time() - start_time):.2f}s')
    
    image = None

def generate_from_json(argdict):
    global pipe, refiner
    mode = argdict.get("mode")
    prompt = argdict.get("prompt")
    prompt_neg = argdict.get("promptNeg")
    inpath = argdict.get("initImg")
    steps = int(argdict.get("steps") or 0)
    steps_refine = int(argdict.get("stepsRefine") or 0)
    seed = int(argdict.get("seed") or 0)
    cfg_txt = float(argdict.get("scaleTxt") or 0.0)
    cfg_img = float(argdict.get("scaleImg") or 0.0)
    w = int(argdict.get("w") or 0)
    h = int(argdict.get("h") or 0)
    init_strength = float(argdict.get("initStrength") or 0.0)
    mask_path = argdict.get("inpaintMask")
    sampler = argdict.get("sampler")
    if args.pipeline == "InstructPix2Pix":
        generate_ip2p(inpath, args.outpath, prompt, prompt_neg, steps, sampler, seed, cfg_txt, cfg_img)
    if args.pipeline == "SdOnnx":
        generate_sd_onnx(prompt, prompt_neg, args.outpath, steps, w, h, sampler, seed, cfg_txt, inpath, init_strength, mask_path)
    if args.pipeline == "SdXl":
        pipe, refiner = load_sdxl(pipe, refiner, argdict.get("model"), argdict.get("modelRefiner"), mode, args.sdxl_opt)
        refine_frac = float(argdict.get("refineFrac") or 0.0)
        generate_sd_xl(mode, prompt, prompt_neg, args.outpath, steps, w, h, sampler, seed, cfg_txt, inpath, init_strength, mask_path, refine_frac)

def main():
    # print(f"Loading...")
    # if args.pipeline == "InstructPix2Pix":
    #     load_ip2p()
    # if args.pipeline == "SdOnnx":
    #     load_sd_onnx()
    # if args.pipeline == "SdXl":
    #     load_sdxl()
    # print(f"Model loaded.")
    
    # Process messages from the queue
    while True:
        try:
            message = stdin_queue.get(block = True, timeout = 1)
            split = message.split()
            cmd = split[0]
            cmd_args = ' '.join(split[1:])
            
            if cmd == "generate":
                data = json.loads(cmd_args)
                generate_from_json(data)
                
            if cmd == "stop":
                print(f"Stopped.")
    
            if cmd == "exit":
                os._exit(0)
                
        except queue.Empty:
            if not stdin_thread.is_alive():
                print(f"Breaking because queue empty and stdin thread not alive")
                break
    
        except Exception as ex:
            print(f"Exception: {str(ex)}")
            traceback.print_stack()
            break
    
    pipe = None
    
    print(f"Exiting...")
    os._exit(0)

if __name__ == "__main__":
    main()