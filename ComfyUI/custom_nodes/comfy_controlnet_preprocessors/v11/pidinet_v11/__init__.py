# Pidinet
# https://github.com/hellozhuo/pidinet

import os
import torch
import numpy as np
import comfy.model_management
from einops import rearrange
from .model import pidinet
from custom_nodes.comfy_controlnet_preprocessors.util import annotator_ckpts_path, safe_step, load_file_from_url

class PidiNetDetector:
    def __init__(self):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth"
        modelpath = os.path.join(annotator_ckpts_path, "table5_pidinet.pth")
        if not os.path.exists(modelpath):
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        self.netNetwork = pidinet()
        self.netNetwork.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(
            modelpath, map_location=comfy.model_management.get_torch_device())['state_dict'].items()})
        self.netNetwork.eval()
        self.netNetwork = self.netNetwork.to(comfy.model_management.get_torch_device())

    def __call__(self, input_image, safe=False):
        assert input_image.ndim == 3
        input_image = input_image[:, :, ::-1].copy()
        with torch.no_grad():
            image_pidi = torch.from_numpy(input_image).float().to(comfy.model_management.get_torch_device())
            image_pidi = image_pidi / 255.0
            image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')
            edge = self.netNetwork(image_pidi)[-1]
            edge = edge.cpu().numpy()
            if safe:
                edge = safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
            return edge[0][0]
