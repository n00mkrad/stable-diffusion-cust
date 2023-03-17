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
    help="Path to model checkpoint file",
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
    help="Use fp16 (half-precision) to further reduce file size",
    dest='fp16',
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    help="Path to model config file (usually v1-inference.yaml)",
    dest='configpath',
)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

opt = parser.parse_args()

if not opt.configpath:
    opt.configpath = '../configs/stable-diffusion/v1-inference.yaml'

if not opt.outpath:
    fname = os.path.splitext(opt.inpath)[0]
    if opt.fp16:
        opt.outpath = f"{fname}-pruned-fp16.ckpt"
    else:
        opt.outpath = f"{fname}-pruned.ckpt"

def prune(config_path, in_path, out_path, fp16=False):
    if fp16:
        print("Saving model as fp16 (half-precision). This cuts the file size in half but may slightly reduce quality.", flush=True)
        torch.set_default_tensor_type(torch.HalfTensor)
        torch.set_default_dtype(torch.float16)
    pl_sd = torch.load(in_path, map_location='cpu')
    sd = pl_sd['state_dict']
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    torch.save({'state_dict': model.state_dict()}, out_path)
    if fp16:
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_dtype(torch.float32)

print("Pruning...", flush=True)

prune(opt.configpath, opt.inpath, opt.outpath, opt.fp16)

print("Done.", flush=True)