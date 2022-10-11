import os
import sys
import stat
import time
import glob
import argparse


os.chdir(sys.path[0])

parser = argparse.ArgumentParser()

parser.add_argument(
    "-dir",
    type=str,
    help="Path to checkpoints dir",
    dest='dir',
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
    dest='cfg_path',
)

opt = parser.parse_args()


if not opt.cfg_path:
    opt.cfg_path = '../configs/stable-diffusion/v1-inference.yaml'

min_age_seconds = 30
prune_counter = 0

def file_age_in_seconds(path):
    return time.time() - os.stat(path)[stat.ST_MTIME]
    
def file_is_older_than(path, min_age_secs):
    return file_age_in_seconds(path) > min_age_secs

def checkpoint_is_big(path):
    return os.path.getsize(path) > 5368709120 # 5 GB

def try_remove(path):
    try:
        os.remove(path)
        print(f"Deleted {path}.")
    except:
        print(f"Error while deleting file: {path}")

last_ckpt_path = os.path.join(opt.dir, "last.ckpt")

print("Waiting...")

while True:
    time.sleep(20)
        
    epochFiles = glob.glob(os.path.join(opt.dir, "epoch*.ckpt"))
        
    for epochFile in epochFiles:
        if file_is_older_than(epochFile, min_age_seconds):
            try_remove(epochFile)
    
    if os.path.exists(last_ckpt_path) and file_is_older_than(last_ckpt_path, min_age_seconds):
        print(f"Modified {file_age_in_seconds(last_ckpt_path)} seconds ago, will prune.")
        prune_counter += 1
        ckpt_name = os.path.join(opt.dir, f"checkpoint{prune_counter}.ckpt")
        os.rename(last_ckpt_path, ckpt_name)
        os.system(f"python prune_model.py -i \"{ckpt_name}\" -c \"{opt.cfg_path}\" {'--half' if opt.fp16 else ''}")
        prunedFiles = glob.glob(os.path.join(opt.dir, f"checkpoint{prune_counter}*pruned*.ckpt"))
        success = len(prunedFiles) > 0
        print(f"Success: {success}")
        if success:
            try_remove(ckpt_name)
        print("Waiting...")