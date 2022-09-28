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


# parser = argparse.ArgumentParser(description="Merge two models")
# user_input_1 = input("Enter the first model name: ")
# user_input_2 = input("Enter the second model name: ")
# alpha = float(input("Enter the second model weight(default:0.5): ") or 0.5)
# parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)

# args = parser.parse_args()

model_0 = torch.load(opt.model1)
model_1 = torch.load(opt.model2)
theta_0 = model_0["state_dict"]
theta_1 = model_1["state_dict"]

print("Merging models, please wait...")

# output_file = f'{user_input_1.strip(".ckpt")}-{str(round((1-alpha)*100,2))}-{user_input_2.strip(".ckpt")}-{str(round((alpha*100),2))}.ckpt'

for key in tqdm(theta_0.keys(), desc="Stage 1/2"):
    if "model" in key and key in theta_1:
        theta_0[key] = (1 - opt.alpha) * theta_0[key] + opt.alpha * theta_1[key]

for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
    if "model" in key and key not in theta_0:
        theta_0[key] = theta_1[key]

print("Saving...")

torch.save({"state_dict": theta_0}, opt.outpath)

print("Done!")