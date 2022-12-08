from safetensors import safe_open
from safetensors.torch import save_file
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import argparse
import os
import sys

os.chdir(sys.path[0])

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--input",
    type=str,
    help="Path to model file",
    dest='inpath',
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="Output path including filename (set automatically if omitted)",
    dest='outpath',
)
parser.add_argument(
    "-half",
    "--half-precision",
    action="store_true",
    help="Use fp16 (half-precision) to further reduce file size (if applicable to model format)",
    dest='fp16',
)
parser.add_argument(
    "-if",
    "--input-format",
    choices=["pickle", "diffusers", "safetensors"]
    help="Specify input model format",
    dest='informat',
)
parser.add_argument(
    "-of",
    "--output-format",
    choices=["pickle", "diffusers", "safetensors"]
    help="Specify output model format",
    dest='outformat',
)


if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

opt = parser.parse_args()

with torch.no_grad():
    weights = torch.load("sd-v1.4.ckpt")["state_dict"]
    print(f'Saving...')
    save_file(weights, "st.safetensors")
    
tensors = {}
with safe_open("st.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)
    torch.save({'state_dict': tensors}, "pt.ckpt")


print('Done!')