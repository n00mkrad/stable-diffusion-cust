import functools; print = functools.partial(print, flush = True)
import PIL
from PIL import PngImagePlugin, Image
import torch
import diffusers
from diffusers import StableDiffusionInstructPix2PixPipeline # InstructPix2Pix
from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline, OnnxStableDiffusionInpaintPipeline, OnnxStableDiffusionInpaintPipelineLegacy # SD ONNX
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, DiffusionPipeline # SDXL
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
import argparse
import os, sys, time
import json
import threading
import queue
import traceback

from nmkdiffusers_load import load_sd_onnx, load_ip2p, load_sdxl
from nmkdiffusers_generate import generate_ip2p, generate_sd_onnx, generate_sd_xl

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

pipe = None
refiner = None

def set_scheduler(sampler_name):
    if pipe is None:
        return
    if sampler_name == "ddim": sched = diffusers.DDIMScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "plms": sched = diffusers.PNDMScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "lms": sched = diffusers.LMSDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "heun": sched = diffusers.HeunDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "euler": sched = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "k_euler": sched = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = True)
    if sampler_name == "euler_a": sched = diffusers.EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "dpm_2": sched = diffusers.KDPM2DiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "dpm_2_a": sched = diffusers.KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "dpmpp_2s": sched = diffusers.DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "dpmpp_2m": sched = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    if sampler_name == "k_dpmpp_2m": sched = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = True)
    if sampler_name == "dpmpp_2m_sde": sched = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False, algorithm_type="sde-dpmsolver++")
    if sampler_name == "k_dpmpp_2m_sde": sched = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = True, algorithm_type="sde-dpmsolver++")
    if sampler_name == "unipc": sched = diffusers.UniPCMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas = False)
    print(f"Set base scheduler to {sampler_name}")
    pipe.scheduler = sched
    if refiner is not None:
        print(f"Set refiner scheduler to {sampler_name}")
        refiner.scheduler = sched

def generate_from_json(argdict):
    global pipe, refiner
    mode = argdict.get("mode")
    model = argdict.get("model")
    sampler = argdict.get("sampler")
    # Load and run pipeline
    if args.pipeline == "InstructPix2Pix":
        pipe = load_ip2p(pipe, model)
        set_scheduler(sampler)
        generate_ip2p(pipe, argdict, args.outpath)
    if args.pipeline == "SdOnnx":
        pipe = load_sd_onnx(pipe, model, mode)
        set_scheduler(sampler)
        generate_sd_onnx(pipe, argdict, args.outpath)
    if args.pipeline == "SdXl":
        pipe, refiner = load_sdxl(pipe, refiner, model, argdict.get("modelRefiner"), mode, args.sdxl_opt)
        set_scheduler(sampler)
        generate_sd_xl(pipe, refiner, argdict, args.outpath)

def main():
    print(f"Ready.")
    
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
            print(traceback.format_exc())
            break
    
    print(f"Exiting...")
    os._exit(0)

if __name__ == "__main__":
    main()