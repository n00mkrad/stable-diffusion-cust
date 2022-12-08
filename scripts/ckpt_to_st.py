from safetensors import safe_open
from safetensors.torch import save_file
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import argparse
import os
import sys

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--inpath",
    type=str,
    help="Path to ckpt file",
    dest='inpath',
)
parser.add_argument(
    "-o",
    "--outpath",
    type=str,
    help="Output path for safetensors file",
    dest='outpath',
)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

opt = parser.parse_args()

os.chdir(sys.path[0])

print('Saving...', flush=True)

with torch.no_grad():
    weights = torch.load(opt.inpath)["state_dict"]
    print(f'Saving...')
    save_file(weights, opt.outpath)

print('Done!', flush=True)