from custom_nodes.comfy_controlnet_preprocessors.util import HWC3
from .shuffle import ContentShuffleDetector

preprocessor = ContentShuffleDetector()

def preprocess(image):
    H, W, C = image.shape
    image = HWC3(image)
    detected_map = preprocessor(image, w=W, h=H, f=256)

    return detected_map
