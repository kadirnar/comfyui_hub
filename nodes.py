import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ComfyUI.custom_nodes.comfyui_helpers.noisehub.plasma_noise.node import (
    PlasmaNoise, RandNoise, GreyNoise, PinkNoise, BrownNoise, PlasmaSampler,
    PowerImage, ImageContrast, GreyScale, Equalize, AutoContrast, ResizeFactor,
    BlendImages, GaussianBlur, LoadImagePath, LoadImagePathWithMetadata
)
                                                                        
from ComfyUI.custom_nodes.comfyui_helpers.noisehub.advanced_noise.node import MathEncode, LatentGaussianNoise
from ComfyUI.custom_nodes.comfyui_helpers.noisehub.perlin_noise.node import NoisyLatentPerlin
from ComfyUI.custom_nodes.comfyui_helpers.preprocessing.node import ImageLoaderAndProcessor




NODE_CLASS_MAPPINGS = {
	"MathEncode": MathEncode,
	"LatentGaussianNoise": LatentGaussianNoise,
    "NoisyLatentPerlin": NoisyLatentPerlin,
    "ImageLoaderAndProcessor": ImageLoaderAndProcessor,
	"JDC_Plasma": PlasmaNoise,
	"JDC_RandNoise": RandNoise,
	"JDC_GreyNoise": GreyNoise,
	"JDC_PinkNoise": PinkNoise,
	"JDC_BrownNoise": BrownNoise,
	"JDC_PlasmaSampler": PlasmaSampler,
	"JDC_PowerImage": PowerImage,
	"JDC_Contrast": ImageContrast,
	"JDC_Greyscale": GreyScale,
	"JDC_EqualizeGrey": Equalize,
	"JDC_AutoContrast": AutoContrast,
	"JDC_ResizeFactor": ResizeFactor,
	"JDC_BlendImages": BlendImages,
	"JDC_GaussianBlur": GaussianBlur,
	"JDC_ImageLoader": LoadImagePath,
	"JDC_ImageLoaderMeta": LoadImagePathWithMetadata
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MathEncode": MathEncode.TITLE,
    "LatentGaussianNoise": LatentGaussianNoise.TITLE,
    "NoisyLatentPerlin": NoisyLatentPerlin.TITLE,
    "ImageLoaderAndProcessor": ImageLoaderAndProcessor.TITLE,
	"JDC_Plasma": "Plasma Noise",
	"JDC_RandNoise": "Random Noise",
	"JDC_GreyNoise": "Greyscale Noise",
	"JDC_PinkNoise": "Pink Noise",
	"JDC_BrownNoise": "Brown Noise",
	"JDC_PlasmaSampler": "Plasma KSampler",
	"JDC_PowerImage": "Image To The Power Of",
	"JDC_Contrast": "Brightness & Contrast",
	"JDC_Greyscale": "RGB to Greyscale",
	"JDC_EqualizeGrey": "Equalize Histogram",
	"JDC_AutoContrast": "AutoContrast",
	"JDC_ResizeFactor": "Resize Image by Factor",
	"JDC_BlendImages": "Blend Images",
	"JDC_GaussianBlur": "Gaussian Blur",
	"JDC_ImageLoader": "Load Image From Path",
	"JDC_ImageLoaderMeta": "Load Image From Path With Meta"
}
