import argparse
import os
import shutil
import sys
from huggingface_hub import snapshot_download

os.chdir(sys.path[0])


parser = argparse.ArgumentParser()

parser.add_argument(
    "-r",
    "--repo",
    type=str,
    help="Diffusers repository to clone",
    dest="repo",
)
parser.add_argument(
    "-c",
    "--cache_path",
    type=str,
    help="Save path",
    dest="cache_path",
)
parser.add_argument(
    "-s",
    "--save_path",
    type=str,
    help="Save path",
    dest="save_path",
)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

args = parser.parse_args()


ignore = ["*.ckpt", "*.safetensors", "safety_checker/*", ".md", ".git*"]
# rev = "fp16"
snapshot_dir = snapshot_download(repo_id=args.repo, ignore_patterns=ignore, cache_dir=args.cache_path)
print(f"Moving {snapshot_dir} to {args.save_path}", flush=True)
os.rename(snapshot_dir, args.save_path)
shutil.rmtree(os.path.join(snapshot_dir, "..", ".."))

print(f"Done.", flush=True)
os._exit(0)