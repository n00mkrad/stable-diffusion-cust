from .util import common_annotator_call, img_np_to_tensor, skip_v1

if not skip_v1:
    from ..v1 import uniformer
from ..v11 import oneformer

if not skip_v1:
    class Uniformer_SemSegPreprocessor:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": {"image": ("IMAGE",)}}

        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "semantic_segmentate"

        CATEGORY = "preprocessors/semseg"

        def semantic_segmentate(self, image):
            # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_seg2image.py
            np_detected_map = common_annotator_call(uniformer.UniformerDetector(), image)
            return (img_np_to_tensor(np_detected_map),)


class OneFormer_COCO_SemSegPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "semantic_segmentate"

    CATEGORY = "preprocessors/semseg"

    def semantic_segmentate(self, image):
        np_detected_map = common_annotator_call(oneformer.OneformerDetector({
            "name": "150_16_swin_l_oneformer_coco_100ep.pth",
            "config": 'configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml'
        }), image)
        return (img_np_to_tensor(np_detected_map),)


class OneFormer_ADE20K_SemSegPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "semantic_segmentate"

    CATEGORY = "preprocessors/semseg"

    def semantic_segmentate(self, image):
        np_detected_map = common_annotator_call(oneformer.OneformerDetector({
            "name": "250_16_swin_l_oneformer_ade20k_160k.pth",
            "config": 'configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml'
        }), image)
        return (img_np_to_tensor(np_detected_map),)


NODE_CLASS_MAPPINGS = {

    "OneFormer-COCO-SemSegPreprocessor": OneFormer_COCO_SemSegPreprocessor,
    "OneFormer-ADE20K-SemSegPreprocessor": OneFormer_ADE20K_SemSegPreprocessor
}
if not skip_v1:
    NODE_CLASS_MAPPINGS.update({
        "UniFormer-SemSegPreprocessor": Uniformer_SemSegPreprocessor,
        "SemSegPreprocessor": Uniformer_SemSegPreprocessor,
    })
