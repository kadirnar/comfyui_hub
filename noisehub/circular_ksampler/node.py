# https://github.com/cobanov/cobanov_nodes/blob/main/cobanovnodes.py

import comfy
import torch
import nodes

MAX_RESOLUTION = 8192


class CustomKSamplerAdvancedTile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"],),
                "noise_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "tileX": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
                "tileY": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "ComfyUI-Helpers/Ksampler/CustomTiledKsampler"

    def set_layer_padding(self, layer, tileX, tileY):
        """Set padding mode and values for a convolutional layer based on tiling parameters."""
        layer.padding_modeX = "circular" if tileX else "constant"
        layer.padding_modeY = "circular" if tileY else "constant"
        layer.paddingX = (
            layer._reversed_padding_repeated_twice[0],
            layer._reversed_padding_repeated_twice[1],
            0,
            0,
        )
        layer.paddingY = (
            0,
            0,
            layer._reversed_padding_repeated_twice[2],
            layer._reversed_padding_repeated_twice[3],
        )

    def apply_asymmetric_tiling(self, model, tileX, tileY):
        """Apply asymmetric tiling to all convolutional layers in the model."""
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                self.set_layer_padding(layer, tileX, tileY)
                print(layer.paddingX, layer.paddingY)

    def hijack_conv2d_methods(self, model, tileX: bool, tileY: bool):
        """Override the convolution methods of Conv2d layers in the model."""
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                self.set_layer_padding(layer, tileX, tileY)

                def make_bound_method(method, current_layer):
                    def bound_method(self, *args, **kwargs):
                        return method(current_layer, *args, **kwargs)

                    return bound_method

                bound_method = make_bound_method(self.replacement_conv2d_forward, layer)
                layer._conv_forward = bound_method.__get__(layer, type(layer))

    def replacement_conv2d_forward(self, layer, input: torch.Tensor, weight: torch.Tensor, bias: [torch.Tensor]):
        """Replacement method for convolutional forward pass."""
        working = torch.nn.functional.pad(input, layer.paddingX, mode=layer.padding_modeX)
        working = torch.nn.functional.pad(working, layer.paddingY, mode=layer.padding_modeY)
        return torch.nn.functional.conv2d(working, weight, bias, layer.stride, (0, 0), layer.dilation, layer.groups)

    def restore_conv2d_methods(self, model):
        """Restore the original convolution methods to Conv2d layers in the model."""
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer._conv_forward = torch.nn.Conv2d._conv_forward.__get__(layer, torch.nn.Conv2d)

    def sample(
        self,
        model,
        add_noise,
        noise_seed,
        tileX,
        tileY,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        denoise=1.0,
    ):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        self.hijack_conv2d_methods(model.model, tileX == 1, tileY == 1)
        result = nodes.common_ksampler(
            model,
            noise_seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
            disable_noise=disable_noise,
            start_step=start_at_step,
            last_step=end_at_step,
            force_full_denoise=force_full_denoise,
        )
        self.restore_conv2d_methods(model.model)
        return result


class CircularVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "ComfyUI-Helpers/Ksampler/CircularVAEDecode"

    def decode(self, vae, samples):
        for layer in [layer for layer in vae.first_stage_model.modules() if isinstance(layer, torch.nn.Conv2d)]:
            layer.padding_mode = "circular"
        return (vae.decode(samples["samples"]),)
