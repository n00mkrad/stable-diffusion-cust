import torch
import diffusers
import argparse
from diffusers import StableDiffusionXLPipeline

parser = argparse.ArgumentParser()

parser.add_argument(
    "--checkpoint_path",
    default=None,
    type=str,
    required=True,
    help="Path to the checkpoint to convert.",
)
parser.add_argument(
    "--to_safetensors",
    action="store_true",
    help="Whether to store pipeline in safetensors format or not.",
)
parser.add_argument(
    "--dump_path",
    default=None,
    type=str,
    required=True,
    help="Path to the output model.",
)
parser.add_argument(
    "--half",
    action="store_true",
    help="Save weights in half precision."
)

args = parser.parse_args()

pipe = StableDiffusionXLPipeline.from_single_file(args.checkpoint_path, local_files_only = True, use_safetensors = True, torch_dtype = torch.float32)

if args.half:
    pipe.to(torch_dtype=torch.float16)

pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)