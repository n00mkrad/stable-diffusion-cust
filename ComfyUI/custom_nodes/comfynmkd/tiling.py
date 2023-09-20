import torch

class Conv2dSettings:
    PADDING_MODES = ['zeros', 'circular']
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
            },
            "required": {
                "padding_mode": (s.PADDING_MODES, ),
            },
        }

    RETURN_TYPES = ("MODEL", "VAE", "CLIP")

    FUNCTION = "apply_settings"

    CATEGORY = "_for_testing"

    def apply_settings(self, model=None, vae=None, clip=None, padding_mode=None):
        if model:
            self._apply_settings(model.model, padding_mode)
        if vae:
            self._apply_settings(vae.first_stage_model, padding_mode)

        return (model, vae, clip)

    def _apply_settings(self, model, padding_mode):
        def flatten(el):
            flattened = [flatten(children) for children in el.children()]
            res = [el]
            for c in flattened:
                res += c
            return res

        layers = flatten(model)

        for layer in [layer for layer in layers if isinstance(layer, torch.nn.Conv2d)]:
            layer.padding_mode = padding_mode if padding_mode else 'zeros'

        return model

NODE_CLASS_MAPPINGS = {
    "Conv2dSettings": Conv2dSettings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Conv2dSettings": "Conv2d Settings"
}