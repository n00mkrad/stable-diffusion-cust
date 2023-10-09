import nodes
import folder_paths
import comfy.sd
import latent_preview
import torch
import json
from PIL import Image, ImageOps
import numpy as np
import comfy.utils
from comfy_extras.chainner_models import model_loading
from comfy import model_management
import time
import math
import base64
from io import BytesIO
import os
from comfy_extras.nodes_hypernetwork import load_hypernetwork_patch
from PIL.PngImagePlugin import PngInfo


# Int Constant
class NmkdIntegerConstant:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    },
                }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("value", )
    FUNCTION = "get_value"

    CATEGORY = "Nmkd/Basic"

    def get_value(self, value, ):
        return (value,)


# Float Constant
class NmkdFloatConstant:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "value": ("FLOAT", {"default": 0.0, "step": 0.01}),
                    },
                }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("value", )
    FUNCTION = "get_value"

    CATEGORY = "Nmkd/Basic"

    def get_value(self, value, ):
        return (value,)


# String Constant
class NmkdStringConstant:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "value": ("STRING", {"multiline": True}),
                    },
                }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("value", )
    FUNCTION = "get_value"

    CATEGORY = "Nmkd/Basic"

    def get_value(self, value, ):
        return (value,)


# Checkpoint Loader that supports absolute paths
class NmkdCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "mdl_path": ("STRING", {"multiline": False}),
                    "load_vae": (["disable", "enable"], ),
                    },
                "optional": {
                    "vae_path": ("STRING", {"multiline": False}),
                    "embeddings_dir": ("STRING", {"multiline": False}),
                    "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                    },
                }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "Nmkd/Loaders"

    def load_checkpoint(self, mdl_path, load_vae, vae_path, embeddings_dir, clip_skip):
        print(f"Loading checkpoint: {mdl_path}")
        enable_vae = load_vae == "enable"
        external_vae = enable_vae and vae_path and vae_path is not mdl_path # Only load VAE separately if it has a value and is not identical to the main model
        diffusers = os.path.isdir(mdl_path)
        if not embeddings_dir:
            embeddings_dir = folder_paths.get_folder_paths("embeddings")
        try:
            if(diffusers): # Assume path is Diffusers model
                mdl = comfy.diffusers_load.load_diffusers(mdl_path, fp16=comfy.model_management.should_use_fp16(), output_vae=(enable_vae and not external_vae), output_clip=True, embedding_directory=embeddings_dir)
            else:
                mdl = comfy.sd.load_checkpoint_guess_config(mdl_path, output_vae=(enable_vae and not external_vae), output_clip=True, embedding_directory=embeddings_dir)
            print(f"MODEL INFO: {os.path.basename(mdl_path)} | {mdl[0].model.model_type.name} | {mdl[0].model.model_config.unet_config}{' | Diffusers' if diffusers else ''}")
            if external_vae:
                print(f"Loading external VAE: {vae_path}")
                ext_vae = comfy.sd.VAE(ckpt_path=vae_path)
                mdl = (mdl[0], mdl[1], ext_vae, mdl[3]) if not diffusers else (mdl[0], mdl[1], ext_vae)
            if clip_skip < -1:
                clip = mdl[1]
                clip = clip.clone()
                clip.clip_layer(clip_skip)
                print(f"Applied CLIP skip: Stop at layer {clip_skip}")
            return mdl
        except:
            print(f"Failed to load model: {os.path.basename(mdl_path)}")
            return None


# KSamplerAdvanced but with denoising control
class NmkdKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "Nmkd/Sampling"

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        print(f"Sampling: Steps: {steps} - Start: {start_at_step} - End: {end_at_step} - Add Noise: {add_noise} - Return with noise: {return_with_leftover_noise} - Denoise: {denoise} - Sampler: {sampler_name} - Scheduler: {scheduler}")
        return nodes.common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)


# Image loader that accepts an absolute path
class NmkdImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": { "image_path": ("STRING", {"multiline": False}), },
               }

    RETURN_TYPES = ("IMAGE", "MASK", )
    FUNCTION = "load_image"
    CATEGORY = "Nmkd/Loaders"

    def load_image(self, image_path):
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask)


# ESRGAN Image Upscaler, accepts absolute model path, is skipped if no model is provided
class NmkdImageUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_path": ("STRING", {"multiline": False}),
                              "image": ("IMAGE",),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "Nmkd/Postprocessing"

    def upscale(self, model_path, image):
        if not model_path:
            print("NmkdImageUpscale: No model specified, skipping node.")
            return (image,)
        
        try:
            sd = comfy.utils.load_torch_file(model_path, safe_load=True)
            out = model_loading.load_state_dict(sd).eval()
            upscale_model = out
            device = model_management.get_torch_device()
            upscale_model.to(device)
            in_img = image.movedim(-1,-3).to(device)
            free_memory = model_management.get_free_memory(device)
    
            tile = 1024
            overlap = 32
    
            oom = True
            while oom:
                try:
                    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                    pbar = comfy.utils.ProgressBar(steps)
                    s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                    oom = False
                except model_management.OOM_EXCEPTION as e:
                    tile //= 2
                    if tile < 128:
                        raise e
    
            upscale_model.cpu()
            s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
            return (s,)
        except:
            print("NmkdImageUpscale: Upscaling failed! Returning original image.")
            return (image,)


# Upscale model loader that accepts an absolute path
class NmkdUpscaleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_path": ("STRING", {"multiline": False} ), }}
    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "Nmkd/Loaders"

    def load_model(self, model_path):
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        out = model_loading.load_state_dict(sd).eval()
        return (out, )


# LoRA loader that can load any amount of LoRAs without needing separate nodes
class NmkdMultiLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_paths": ("STRING", {"multiline": True}),
                              "lora_strengths": ("STRING", {"multiline": False}),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_loras"

    CATEGORY = "Nmkd/Loaders"

    def load_loras(self, model, clip, lora_paths, lora_strengths):
        lora_list = lora_paths.split(',')
        strengths_list = lora_strengths.split(',')
    
        if len(lora_list) < 1 or len(strengths_list) < 1:
            print(f"NmkdMultiLoraLoader: Skipping because lora_list has no entries or strengths_list has no entries ({len(lora_list)}/{len(strengths_list)}).")
            return (model, clip)
    
        def apply_lora(model, clip, lora_path, strength):
            if not lora_path or strength < 0.001:
                print(f"NmkdMultiLoraLoader: Skipping because lora_path is falsy or strength is zero ({strength}).")
                return (model, clip)
        
            loaded_lora = None
        
            print(f"Loading LoRA [{strength}]: {lora_path}")
    
            lora = None
            if loaded_lora is not None:
                if loaded_lora[0] == lora_path:
                    lora = loaded_lora[1]
                else:
                    temp = loaded_lora
                    loaded_lora = None
                    del temp
        
            if lora is None:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                loaded_lora = (lora_path, lora)
        
            lora_model, lora_clip = comfy.sd.load_lora_for_models(model, clip, lora, strength, strength)
            return (lora_model, lora_clip)

        for i in range(len(lora_list)):
            lora_path = lora_list[i].strip().strip(',')
            try:
                lora_weight = float(strengths_list[i].strip().strip(','))
            except:
                lora_weight = 1.0
            model, clip = apply_lora(model, clip, lora_path, lora_weight)
        
        print("All LoRAs applied.")
        return (model, clip)


# VAE Encoder with optional mask, can be used for img2img or inpainting
class NmkdVaeEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": { "pixels": ("IMAGE", ), "vae": ("VAE", ), "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),},
        "optional": { "mask": ("MASK", ), },
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "latent/inpaint"

    def encode(self, vae, pixels, mask = None, grow_mask_by=6):    
        if mask is None:
            mask = torch.zeros(pixels.shape[1], pixels.shape[2])
                    
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        pixels = pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        #grow mask by a few pixels to keep things seamless in latent space
        if grow_mask_by == 0:
            mask_erosion = mask
        else:
            kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
            padding = math.ceil((grow_mask_by - 1) / 2)

            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

        print(f"{mask.shape}")
        m = (1.0 - mask.round()).squeeze(1)
        print(f"{m.shape}")
        for i in range(3):
            pixels[:,:,:,i] -= 0.5
            pixels[:,:,:,i] *= m
            pixels[:,:,:,i] += 0.5
        t = vae.encode(pixels)

        return ({"samples":t, "noise_mask": (mask_erosion[:,:,:x,:y].round())}, )


# Mix two images based on a mask
class NmkdImageMaskComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image_to": ("IMAGE",),
                              "image_from": ("IMAGE",),
                              "mask": ("MASK",),
                              }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "Nmkd/Images"

    def composite(self, image_to, image_from, mask):
        print(f"NmkdImageMaskComposite: image_to = {image_to.shape}, image_from = {image_from.shape}, mask = {mask.shape}")
        
        if not (image_to.shape == image_from.shape):
            print(f"NmkdImageMaskComposite: Shape mismatch, returning 'image_to'. ({image_to.shape} vs {image_from.shape})")
            return (image_to,)

        try:
            image_out = image_to.clone()    
            image_out[0,:,:,0] = image_to[0,:,:,0] * mask + image_from[0,:,:,0] * (1 - mask)
            image_out[0,:,:,1] = image_to[0,:,:,1] * mask + image_from[0,:,:,1] * (1 - mask)
            image_out[0,:,:,2] = image_to[0,:,:,2] * mask + image_from[0,:,:,2] * (1 - mask)
            return (image_out,)
        except:
            return (image_to,)


# ControlNet Load & Apply
class NmkdControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": {
                "controlnet_path": ("STRING", {"multiline": False} ),
                "conditioning": ("CONDITIONING", ),
                "image": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
            },
            "optional": {
                "model": ("MODEL",),
                "send_preview": (["disable", "enable"], ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_controlnet"

    CATEGORY = "Nmkd/Loaders"

    def apply_controlnet(self, controlnet_path, conditioning, image, strength, model = None, send_preview = "disable"):
        if not controlnet_path:
            print("NmkdControlNet: No model specified, skipping node.")
            return (conditioning,)

        if strength == 0:
            return (conditioning, )
        
        try:
            if send_preview == "enable":
                img = 255. * image[0].cpu().numpy()
                img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
                buffer = BytesIO()
                img.save(buffer, format="WEBP", subsampling=2, quality=5)
                print(f"PREVIEW_WEBP:{base64.b64encode(buffer.getvalue())}")
            
            control_net = comfy.controlnet.load_controlnet(controlnet_path) if model is None else comfy.controlnet.load_controlnet(controlnet_path, model)
            
            c = []
            control_hint = image.movedim(-1,1)
            for t in conditioning:
                n = [t[0], t[1].copy()]
                c_net = control_net.copy().set_cond_hint(control_hint, strength)
                if 'control' in t[1]:
                    c_net.set_previous_controlnet(t[1]['control'])
                n[1]['control'] = c_net
                n[1]['control_apply_to_uncond'] = True
                c.append(n)
            print(f"NmkdControlNet: Applied '{controlnet_path}' ({strength})")
            return (c, )
        except Exception as e:
            print(f"NmkdControlNet: Failed to apply '{controlnet_path}', returning original conditioning.")
            print(e)
            return (conditioning, )

# Hypernetwork Loader that can take absolute paths
class NmkdHypernetworkLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "hypernetwork_path": ("STRING", {"multiline": False} ),
                    "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                }}
                
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_hypernetwork"
    CATEGORY = "Nmkd/Loaders"

    def load_hypernetwork(self, model, hypernetwork_path, strength):
        model_hypernetwork = model.clone()
        patch = load_hypernetwork_patch(hypernetwork_path, strength)
        if patch is not None:
            model_hypernetwork.set_model_attn1_patch(patch)
            model_hypernetwork.set_model_attn2_patch(patch)
        print(f"Applied Hypernetwork: {os.path.basename(hypernetwork_path)}");
        return (model_hypernetwork,)


class NmkdColorPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
        "image": ("IMAGE",),
        "divisor": ("INT", {"default": 64, "min": 8, "max": 128, "step": 8}),
        }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_palette"

    CATEGORY = "Nmkd/Preprocessors"

    def get_palette(self, image, divisor):
        img = tensor_to_image(image)
        w = img.width
        h = img.height
        img = img.resize((w // divisor, h // divisor), Image.BICUBIC)
        img = img.resize((w, h), Image.NEAREST)
        img = image_to_tensor(img)
        return (img,)


# Encode two texts with CLIP (for pos/neg prompts, etc)
class NmkdDualTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
        "text1": ("STRING", {"multiline": False}),
        "text2": ("STRING", {"multiline": False}),
        "clip": ("CLIP", )
        }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "Nmkd/Conditioning"

    def encode(self, clip, text1, text2):
        cond1, pooled1 = clip.encode_from_tokens(clip.tokenize(text1), return_pooled=True)
        cond2, pooled2 = clip.encode_from_tokens(clip.tokenize(text2), return_pooled=True)
        return ([[cond1, {"pooled_output": pooled1}]], [[cond2, {"pooled_output": pooled2}]])


# Save Image with logging
class NmkdSaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE", ),
                    "filename_prefix": ("STRING", {"default": "ComfyUI"})
                },
                "optional": {
                    "override_save_path": "STRING"
                },
                "hidden": {
                    "prompt": "PROMPT",
                    "extra_pnginfo": "EXTRA_PNGINFO"
                }}

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Nmkd/Images"

    def save_images(self, images, filename_prefix="ComfyUI", override_save_path=None, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}{counter:05}.png"
            
            if not override_save_path:
                abs_path = os.path.join(full_output_folder, file)
                img.save(abs_path, pnginfo=metadata, compress_level=4)
                print(f"Saved image {counter} with prefix '{filename_prefix}' to: {abs_path}")
            else:
                img.save(override_save_path, pnginfo=metadata, compress_level=4)
                print(f"Saved image {counter} with prefix '{filename_prefix}' to override path: {override_save_path}")
            counter += 1

        return ""


# Register nodes

NODE_CLASS_MAPPINGS = {
    "NmkdIntegerConstant": NmkdIntegerConstant,
    "NmkdFloatConstant": NmkdFloatConstant,
    "NmkdStringConstant": NmkdStringConstant,
    "NmkdCheckpointLoader": NmkdCheckpointLoader,
    "NmkdKSampler": NmkdKSampler,
    "NmkdImageLoader": NmkdImageLoader,
    "NmkdImageUpscale": NmkdImageUpscale,
    "NmkdUpscaleModelLoader": NmkdUpscaleModelLoader,
    "NmkdMultiLoraLoader": NmkdMultiLoraLoader,
    "NmkdVaeEncode": NmkdVaeEncode,
    "NmkdImageMaskComposite": NmkdImageMaskComposite,
    "NmkdControlNet": NmkdControlNet,
    "NmkdHypernetworkLoader": NmkdHypernetworkLoader,
    "NmkdColorPreprocessor": NmkdColorPreprocessor,
    "NmkdDualTextEncode": NmkdDualTextEncode,
    "NmkdSaveImage": NmkdSaveImage,
}

def tensor_to_image(tensor):
    img = tensor[0]
    img = 255. * img.cpu().numpy()
    img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    return img
    
def image_to_tensor(image):
    img = np.array(image).astype(np.float32) / 255.0
    img = torch.from_numpy(img)[None,]
    return img