import functools; print = functools.partial(print, flush=True)
import sys
import os
import shutil
import traceback
import warnings
from pathlib import Path
from urllib import request
from tqdm import tqdm

os.chdir(sys.path[0])

def download_realesrgan():
    print("Installing models from RealESRGAN...", file=sys.stderr)
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    wdn_model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"

    model_dest = "../../invoke/models/realesrgan/realesr-general-x4v3.pth"
    wdn_model_dest = "../../invoke/models/realesrgan/realesr-general-wdn-x4v3.pth"

    download_with_progress_bar(model_url, model_dest, "RealESRGAN")
    download_with_progress_bar(wdn_model_url, wdn_model_dest, "RealESRGANwdn")


def download_gfpgan():
    print("Installing GFPGAN models...", file=sys.stderr)
    for model in (
        [
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            "../../invoke/models/gfpgan/GFPGANv1.4.pth",
        ],
        [
            "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
            "../../invoke/models/gfpgan/weights/detection_Resnet50_Final.pth",
        ],
        [
            "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
            "../../invoke/models/gfpgan/weights/parsing_parsenet.pth",
        ],
    ):
        model_url, model_dest = model[0], model[1]
        download_with_progress_bar(model_url, model_dest, "GFPGAN weights")


# ---------------------------------------------
def download_codeformer():
    print("Installing CodeFormer model file...", file=sys.stderr)
    model_url = (
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    )
    model_dest = "../../invoke/models/codeformer/codeformer.pth"
    download_with_progress_bar(model_url, model_dest, "CodeFormer")#
    
class ProgressBar:
    def __init__(self, model_name="file"):
        self.pbar = None
        self.name = model_name

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(
                desc=self.name,
                initial=0,
                unit="iB",
                unit_scale=True,
                unit_divisor=1000,
                total=total_size,
            )
        self.pbar.update(block_size)
    
def download_with_progress_bar(model_url: str, model_dest: str, label: str = "the"):
    try:
        print(f"Installing {label} model file {model_url}...", end="", file=sys.stderr)
        if not os.path.exists(model_dest):
            os.makedirs(os.path.dirname(model_dest), exist_ok=True)
            request.urlretrieve(
                model_url, model_dest, ProgressBar(os.path.basename(model_dest))
            )
            print("...downloaded successfully", file=sys.stderr)
        else:
            print("...exists", file=sys.stderr)
    except Exception:
        print("...download failed", file=sys.stderr)
        print(f"Error downloading {label} model", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        
download_realesrgan()
download_gfpgan()
download_codeformer()