# Uniformer
# From https://github.com/Sense-X/UniFormer
# # Apache-2.0 license
import os

from comfy_controlnet_preprocessors.v1.uniformer.mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from custom_nodes.comfy_controlnet_preprocessors.v1.uniformer.mmseg.core.evaluation import get_palette
from custom_nodes.comfy_controlnet_preprocessors.util import annotator_ckpts_path
import comfy.model_management as model_management


checkpoint_file = "https://huggingface.co/lllyasviel/Annotators/resolve/main/upernet_global_small.pth"
#Change to a new URL but the file is still the same


class UniformerDetector:
    def __init__(self):
        modelpath = os.path.join(annotator_ckpts_path, "upernet_global_small.pth")
        if not os.path.exists(modelpath):
            from custom_nodes.comfy_controlnet_preprocessors.util import load_file_from_url
            load_file_from_url(checkpoint_file, model_dir=annotator_ckpts_path)
        config_file = os.path.join(os.path.dirname(__file__), "exp", "upernet_global_small", "config.py")
        self.model = init_segmentor(config_file, modelpath).to(model_management.get_torch_device())

    def __call__(self, img):
        result = inference_segmentor(self.model, img)
        res_img = show_result_pyplot(self.model, img, result, get_palette('ade'), opacity=1)
        return res_img
