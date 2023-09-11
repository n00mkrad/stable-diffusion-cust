from .util import common_annotator_call, img_np_to_tensor, skip_v1
if not skip_v1:
    from ..v1 import midas, leres
from ..v11 import zoe, normalbae
import numpy as np

if not skip_v1:
    class MIDAS_Depth_Map_Preprocessor:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": {"image": ("IMAGE",),
                                 "a": ("FLOAT", {"default": np.pi * 2.0, "min": 0.0, "max": np.pi * 5.0, "step": 0.05}),
                                 "bg_threshold": ("FLOAT", {"default": 0.05, "min": 0, "max": 1, "step": 0.05})
                                 }}

        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "estimate_depth"

        CATEGORY = "preprocessors/normal_depth_map"

        def estimate_depth(self, image, a, bg_threshold):
            # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_depth2image.py
            depth_map_np, normal_map_np = common_annotator_call(midas.MidasDetector(), image, a, bg_threshold)
            return (img_np_to_tensor(depth_map_np),)


class MIDAS_Normal_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "a": ("FLOAT", {"default": np.pi * 2.0, "min": 0.0, "max": np.pi * 5.0, "step": 0.05}),
                             "bg_threshold": ("FLOAT", {"default": 0.05, "min": 0, "max": 1, "step": 0.05})
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_normal"

    CATEGORY = "preprocessors/normal_depth_map"

    def estimate_normal(self, image, a, bg_threshold):
        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_depth2image.py
        depth_map_np, normal_map_np = common_annotator_call(midas.MidasDetector(), image, a, bg_threshold)
        return (img_np_to_tensor(normal_map_np),)


if not skip_v1:
    class LERES_Depth_Map_Preprocessor:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": {"image": ("IMAGE",),
                                 "rm_nearest": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1, "step": 0.05}),
                                 "rm_background": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05})
                                 }}

        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "estimate_depth"

        CATEGORY = "preprocessors/normal_depth_map"

        def estimate_depth(self, image, rm_nearest, rm_background):
            # Ref: https://github.com/Mikubill/sd-webui-controlnet/blob/main/scripts/processor.py#L105
            depth_map_np = common_annotator_call(leres.apply_leres, image, rm_nearest, rm_background)
            return (img_np_to_tensor(depth_map_np),)


class Zoe_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_depth"

    CATEGORY = "preprocessors/normal_depth_map"

    def estimate_depth(self, image):
        # Ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/blob/main/gradio_depth.py
        np_detected_map = common_annotator_call(zoe.ZoeDetector(), image)
        return (img_np_to_tensor(np_detected_map),)


class BAE_Normal_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_normal"

    CATEGORY = "preprocessors/normal_depth_map"

    def estimate_normal(self, image):
        # Ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/blob/main/gradio_normalbae.py
        np_detected_map = common_annotator_call(normalbae.NormalBaeDetector(), image)
        return (img_np_to_tensor(np_detected_map),)


NODE_CLASS_MAPPINGS = {
    "MiDaS-NormalMapPreprocessor": MIDAS_Normal_Map_Preprocessor,
    "Zoe-DepthMapPreprocessor": Zoe_Depth_Map_Preprocessor,
    "BAE-NormalMapPreprocessor": BAE_Normal_Map_Preprocessor
}
if not skip_v1:
    NODE_CLASS_MAPPINGS.update({
        "MiDaS-DepthMapPreprocessor": MIDAS_Depth_Map_Preprocessor,
        "LeReS-DepthMapPreprocessor": LERES_Depth_Map_Preprocessor,
    })
