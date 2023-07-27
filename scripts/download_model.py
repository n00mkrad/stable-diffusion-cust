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
    "-v",
    "--revision",
    type=str,
    help="Revision/branch to download",
    default="main",
    dest="rev",
)
parser.add_argument(
    "-c",
    "--cache_path",
    type=str,
    help="Cache path",
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


whitelist = [
    "feature_extractor/*",
    "scheduler/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "unet/*",
    "vae/*",
    "*.json",
    "*.txt",
]

snapshot_dir = snapshot_download(repo_id=args.repo, revision=args.rev, allow_patterns=whitelist, cache_dir=args.cache_path, local_dir_use_symlinks=False, etag_timeout=30, max_workers=1)
print(f"Moving {snapshot_dir} to {args.save_path}", flush=True)
os.rename(snapshot_dir, args.save_path)
shutil.rmtree(os.path.join(snapshot_dir, "..", ".."))
shutil.rmtree(args.cache_path)

print(f"Done.", flush=True)
os._exit(0)