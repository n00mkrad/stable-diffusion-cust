import functools; print = functools.partial(print, flush=True)
import nodes
import comfy.sd
import numpy as np
import comfy.utils
from .Fooocus import core
from .Fooocus import patch


# Sampler with Refiner
class NmkdHybridSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "refiner_model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                    "refiner_switch_step": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m_sde_gpu", }),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras", }),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "refiner_positive": ("CONDITIONING", ),
                    "refiner_negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "sharpness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "Nmkd/Sampling"

    def sample(self, model, refiner_model, add_noise, noise_seed, steps, refiner_switch_step, cfg, sampler_name, scheduler, positive, negative, refiner_positive, refiner_negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, sharpness, denoise):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        patch.sharpness = sharpness
        print(f"Sampling: Steps: {steps} - Switch at: {refiner_switch_step} - Add Noise: {add_noise} - Return with noise: {return_with_leftover_noise} - Denoise: {denoise} - Sampler: {sampler_name} - Scheduler: {scheduler}")
        return (core.ksampler_with_refiner(model, positive, negative, refiner_model, refiner_positive, refiner_negative, latent_image, noise_seed, steps, refiner_switch_step, cfg, sampler_name, scheduler, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise), )


# Register nodes

NODE_CLASS_MAPPINGS = {
    "NmkdHybridSampler": NmkdHybridSampler,
}