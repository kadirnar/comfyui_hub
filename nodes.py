import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ComfyUI.custom_nodes.comfyui_helpers.noisehub.advanced_noise.node import MathEncode, LatentGaussianNoise
from ComfyUI.custom_nodes.comfyui_helpers.noisehub.perlin_noise.node import NoisyLatentPerlin
from ComfyUI.custom_nodes.comfyui_helpers.preprocessing.node import ImageLoaderAndProcessor


NODE_CLASS_MAPPINGS = {
	"MathEncode": MathEncode,
	"LatentGaussianNoise": LatentGaussianNoise,
    "NoisyLatentPerlin": NoisyLatentPerlin,
    "ImageLoaderAndProcessor": ImageLoaderAndProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MathEncode": MathEncode.TITLE,
    "LatentGaussianNoise": LatentGaussianNoise.TITLE,
    "NoisyLatentPerlin": NoisyLatentPerlin.TITLE,
    "ImageLoaderAndProcessor": ImageLoaderAndProcessor.TITLE,
}
