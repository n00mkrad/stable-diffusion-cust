from .util import common_annotator_call, img_np_to_tensor, img_tensor_to_np, skip_v1
if not skip_v1:
    from ..v1 import canny, hed_v1, mlsd
from ..v11 import hed_v11, pidinet_v11, lineart, lineart_anime
from .. import binary, manga_line_extraction
import numpy as np
import cv2

if not skip_v1:
    class Canny_Edge_Preprocessor:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": {"image": ("IMAGE",),
                                 "low_threshold": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                                 "high_threshold": ("INT", {"default": 200, "min": 0, "max": 255, "step": 1}),
                                 "l2gradient": (["disable", "enable"], {"default": "disable"})
                                 }}

        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "detect_edge"

        CATEGORY = "preprocessors/edge_line"

        def detect_edge(self, image, low_threshold, high_threshold, l2gradient):
            # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_canny2image.py
            np_detected_map = common_annotator_call(canny.CannyDetector(), image, low_threshold, high_threshold,
                                                    l2gradient == "enable")
            return (img_np_to_tensor(np_detected_map),)


class HED_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        ret = {"required": {"image": ("IMAGE",), "version": (["v1", "v1.1"], {"default": "v1.1"}),
                            "safe": (["enable", "disable"], {"default": "enable"})}}
        if not skip_v1:
            ret["required"]["version"] = (["v1.1"], {"default": "v1.1"})

        return ret

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_boundary"

    CATEGORY = "preprocessors/edge_line"

    def detect_boundary(self, image, version, safe):
        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_hed2image.py, https://github.com/lllyasviel/ControlNet-v1-1-nightly/blob/main/gradio_softedge.py
        safe = safe == "enable"
        if version == "v1.1":
            np_detected_map = common_annotator_call(hed_v11.HEDdetector(), image, safe)
        else:
            np_detected_map = common_annotator_call(hed_v1.HEDdetector(), image)
        return (img_np_to_tensor(np_detected_map),)


class Scribble_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_scribble"

    CATEGORY = "preprocessors/edge_line"

    def transform_scribble(self, image):
        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_scribble2image.py
        np_img_list = img_tensor_to_np(image)
        out_list = []
        for np_img in np_img_list:
            np_detected_map = np.zeros_like(np_img, dtype=np.uint8)
            np_detected_map[np.min(np_img, axis=2) < 127] = 255
            out_list.append(np_detected_map)
        return (img_np_to_tensor(out_list),)


class Fake_Scribble_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_scribble"

    CATEGORY = "preprocessors/edge_line"

    def transform_scribble(self, image):
        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_fake_scribble2image.py
        np_detected_map_list = common_annotator_call(hed_v1.HEDdetector(), image)
        out_list = []
        for np_detected_map in np_detected_map_list:
            np_detected_map = hed_v1.nms(np_detected_map, 127, 3.0)
            np_detected_map = cv2.GaussianBlur(np_detected_map, (0, 0), 3.0)
            np_detected_map[np_detected_map > 4] = 255
            np_detected_map[np_detected_map < 255] = 0
            out_list.append(np_detected_map)
        return (img_np_to_tensor(out_list),)

if not skip_v1:
    class MLSD_Preprocessor:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": {"image": ("IMAGE",),
                                 # Idk what should be the max value here since idk much about ML
                                 "score_threshold": (
                                 "FLOAT", {"default": np.pi * 2.0, "min": 0.0, "max": np.pi * 2.0, "step": 0.05}),
                                 "dist_threshold": ("FLOAT", {"default": 0.05, "min": 0, "max": 1, "step": 0.05})
                                 }}

        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "detect_edge"

        CATEGORY = "preprocessors/edge_line"

        def detect_edge(self, image, score_threshold, dist_threshold):
            # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_hough2image.py
            np_detected_map = common_annotator_call(mlsd.MLSDdetector(), image, score_threshold, dist_threshold)
            return (img_np_to_tensor(np_detected_map),)


class Binary_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image": ("IMAGE",), "threshold": ("INT", {"min": 0, "max": 255, "step": 1, "default": 0})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_binary"

    CATEGORY = "preprocessors/edge_line"

    def transform_binary(self, image, threshold):
        np_detected_map = common_annotator_call(binary.apply_binary, image, threshold)
        return (img_np_to_tensor(np_detected_map),)


class PIDINET_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "safe": (["enable", "disable"], {"default": "enable"})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_boundary"

    CATEGORY = "preprocessors/edge_line"

    def detect_boundary(self, image, safe):
        np_detected_map = common_annotator_call(pidinet_v11.PidiNetDetector(), image, safe == "enable")
        # The only diff between PiDiNet v1.1 and v1 is safe mode
        return (img_np_to_tensor(np_detected_map),)


class LineArt_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "coarse": (["disable", "enable"], {"default": "disable"})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_lineart"

    CATEGORY = "preprocessors/edge_line"

    def transform_lineart(self, image, coarse):
        np_detected_map = common_annotator_call(lineart.LineartDetector(), image, coarse == "enable")
        reversed_np_detected_map = map(lambda np_img: 255 - np_img, np_detected_map)
        # https://github.com/lllyasviel/ControlNet-v1-1-nightly/blob/main/gradio_lineart.py#L48
        return (img_np_to_tensor(reversed_np_detected_map),)


class AnimeLineArt_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_lineart"

    CATEGORY = "preprocessors/edge_line"

    def transform_lineart(self, image):
        np_detected_map = common_annotator_call(lineart_anime.LineartAnimeDetector(), image)
        reversed_np_detected_map = map(lambda np_img: 255 - np_img, np_detected_map)
        # https://github.com/lllyasviel/ControlNet-v1-1-nightly/blob/main/gradio_lineart_anime.py#L54
        return (img_np_to_tensor(reversed_np_detected_map),)


class Manga2Anime_LineArt_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_lineart"

    CATEGORY = "preprocessors/edge_line"

    def transform_lineart(self, image):
        np_detected_map = common_annotator_call(manga_line_extraction.MangaLineExtractor(), image)
        reversed_np_detected_map = map(lambda np_img: 255 - np_img, np_detected_map)
        # Based on AnimeLineArt_Preprocessor
        return (img_np_to_tensor(reversed_np_detected_map),)


NODE_CLASS_MAPPINGS = {
    "HEDPreprocessor": HED_Preprocessor,
    "ScribblePreprocessor": Scribble_Preprocessor,
    "FakeScribblePreprocessor": Fake_Scribble_Preprocessor,
    "BinaryPreprocessor": Binary_Preprocessor,
    "PiDiNetPreprocessor": PIDINET_Preprocessor,
    "LineArtPreprocessor": LineArt_Preprocessor,
    "AnimeLineArtPreprocessor": AnimeLineArt_Preprocessor,
    "Manga2Anime-LineArtPreprocessor": Manga2Anime_LineArt_Preprocessor
}
if not skip_v1:
    NODE_CLASS_MAPPINGS.update({
        "CannyEdgePreprocessor": Canny_Edge_Preprocessor,
        "M-LSDPreprocessor": MLSD_Preprocessor,
    })
