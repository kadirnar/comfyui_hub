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
from ComfyUI.custom_nodes.comfyui_helpers.noisehub.latent2rgb.node import LatentToRGB


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
	"JDC_ImageLoaderMeta": LoadImagePathWithMetadata,
	"LatentToRGB": LatentToRGB,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MathEncode": MathEncode.TITLE,
    "LatentGaussianNoise": LatentGaussianNoise.TITLE,
    "NoisyLatentPerlin": NoisyLatentPerlin.TITLE,
    "ImageLoaderAndProcessor": ImageLoaderAndProcessor.TITLE,
	"JDC_Plasma": "PlasmaNoise",
	"JDC_RandNoise": "RandomNoise",
	"JDC_GreyNoise": "GreyscaleNoise",
	"JDC_PinkNoise": "PinkNoise",
	"JDC_BrownNoise": "BrownNoise",
	"JDC_PlasmaSampler": "PlasmaKSampler",
	"JDC_PowerImage": "Image2Power",
	"JDC_Contrast": "Brightness_Contrast",
	"JDC_Greyscale": "RGB2Greyscale",
	"JDC_EqualizeGrey": "EqualizeHistogram",
	"JDC_AutoContrast": "AutoContrast",
	"JDC_ResizeFactor": "ResizeImageFactor",
	"JDC_BlendImages": "BlendImages",
	"JDC_GaussianBlur": "GaussianBlur",
	"JDC_ImageLoader": "LoadImagePath",
	"JDC_ImageLoaderMeta": "LoadImagePathMeta",
	"LatentToRGB": "Latent2RGB",
 
}
