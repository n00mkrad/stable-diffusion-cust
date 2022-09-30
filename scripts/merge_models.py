import os
import argparse
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "-1",
    "--first",
    type=str,
    dest='model1',
)
parser.add_argument(
    "-2",
    "--second",
    type=str,
    dest='model2',
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    dest='outpath',
)
parser.add_argument(
    "-w",
    "--weight",
    type=float,
    default=0.5,
    help="weight of second model",
    dest='alpha',
)

opt = parser.parse_args()

model_0 = torch.load(opt.model1)
model_1 = torch.load(opt.model2)
theta_0 = model_0["state_dict"]
theta_1 = model_1["state_dict"]

print("Merging models, please wait...")

for key in tqdm(theta_0.keys(), desc="Stage 1/2"):
    if "model" in key and key in theta_1:
        theta_0[key] = (1 - opt.alpha) * theta_0[key] + opt.alpha * theta_1[key]

for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
    if "model" in key and key not in theta_0:
        theta_0[key] = theta_1[key]

print("Saving...")

torch.save({"state_dict": theta_0}, opt.outpath)

print("Done!")