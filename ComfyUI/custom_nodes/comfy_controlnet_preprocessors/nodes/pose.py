from .util import common_annotator_call, img_np_to_tensor, skip_v1
if not skip_v1:
    from ..v1 import openpose_v1
from ..v11 import openpose_v11
from .. import mp_pose_hand

class OpenPose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        ret = {"required": { "image": ("IMAGE", ),
                              "detect_hand": (["enable", "disable"], {"default": "enable"}),
                              "detect_body": (["enable", "disable"], {"default": "enable"}),
                              "detect_face": (["enable", "disable"], {"default": "enable"}),
                              "version": (["v1", "v1.1"], {"default": "v1.1"})
                              }}
        if not skip_v1:
            ret["required"]["version"] = (["v1.1"], {"default": "v1.1"})
        return ret
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_pose"

    CATEGORY = "preprocessors/pose"

    def estimate_pose(self, image, detect_hand, detect_body, detect_face, version):
        detect_hand = detect_hand == "enable"
        detect_body = detect_body == "enable"
        detect_face = detect_face == "enable"
        if version == "v1.1":
            np_detected_map = common_annotator_call(openpose_v11.OpenposeDetector(), image, detect_hand, detect_body, detect_face)
        else:
            #Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_pose2image.py
            np_detected_map, pose_info = common_annotator_call(openpose_v1.OpenposeDetector(), image, detect_hand, detect_body)
        return (img_np_to_tensor(np_detected_map),)


class Media_Pipe_Hand_Pose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE", ),
                              "detect_pose": (["enable", "disable"], {"default": "enable"}),
                              "detect_hands": (["enable", "disable"], {"default": "enable"}),
                            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect"

    CATEGORY = "preprocessors/pose"

    def detect(self, image, detect_pose, detect_hands):
        np_detected_map = common_annotator_call(mp_pose_hand.apply_mediapipe, image, detect_pose == "enable", detect_hands == "enable")
        return (img_np_to_tensor(np_detected_map),)

NODE_CLASS_MAPPINGS = {
    "OpenposePreprocessor": OpenPose_Preprocessor,
    "MediaPipe-HandPosePreprocessor": Media_Pipe_Hand_Pose_Preprocessor
}
