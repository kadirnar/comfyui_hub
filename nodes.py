import os
from PIL import Image, ImageOps
import numpy as np
import torch
import folder_paths
import torch
import numpy as np
from torchvision import transforms
from noisehub.latent_math_encoder import linear_encoder
from noisehub.latent_noise_generator import gaussian_latent_noise

class MathEncode:
	"""
		Encode latents without using a NN.
	"""
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"pixels": ("IMAGE",),
				"latent_ver": (["v1", "xl"],),
				"mode": ([
					"linear_encoder",
				],),
			}
		}
	RETURN_TYPES = ("LATENT",)
	FUNCTION = "encode"
	CATEGORY = "comfyui_helpers"
	TITLE = "MathEncoder"

	def encode(self, pixels, latent_ver, mode):
		out = []
		for batch, img in enumerate(pixels.numpy()):
			# target latent size
			lat_size = (round(img.shape[0]/8), round(img.shape[1]/8))
			img = img.transpose((2, 0, 1)) # [W,H,3]=>[3,W,H]
			img = torch.from_numpy(img)
			img = transforms.Resize(lat_size, antialias=True)(img)
			# encode
			lat = linear_encoder(img, latent_ver)
			out.append(lat)
		return ({"samples":torch.stack(out)},)


class LatentGaussianNoise:
	"""
		Create Gaussian noise directly in latent space.
	"""
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"latent_ver": (["v1", "xl"],),
				"width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
				"height": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
				"factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
				"null": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
				"batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
				"scale": ("INT", {"default": 1, "min": 1, "max": 8}),
				"random": (["shared", "per channel"],),
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
			}
		}
	RETURN_TYPES = ("LATENT",)
	FUNCTION = "generate"
	CATEGORY = "comfyui_helpers"
	TITLE = "GaussianNoise(Latent)"

	def generate(self, latent_ver, width, height, factor, null, batch_size, scale, random, seed):
		out = []
		for b in range(batch_size):
			lat = gaussian_latent_noise(
				width = round(width/8/scale),
				height = round(height/8/scale),
				ver = latent_ver,
				seed = seed+b,
				fac = factor,
				nul = null,
				srnd = True if random == "shared" else False,
			)
			if scale > 1:
				target = (round(height/8),round(width/8))
				lat = transforms.Resize(target, antialias=True)(lat)
			out.append(lat)
		out = torch.stack(out)
		
		return ({"samples":out},)


class ImageLoaderAndProcessor:
    """
    A node for processing images with optional transparency handling.

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Specifies input parameters of the node.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tuple.
    FUNCTION (`str`):
        The name of the entry-point method.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    """

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "is_transparent": (["True", "False"], {"checkbox": True, "default": "False"})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_and_process_image"
    CATEGORY = "comfyui_helpers"
    TITLE = "LoadImageandProcess"

    def load_and_process_image(self, image, is_transparent="False"):
        """
        Loads and processes an image with optional transparency handling.

        Parameters
        ----------
        image : IMAGE
            The image to be processed.
        is_transparent : str, optional
            "True" if transparency processing is enabled, "False" otherwise.
            Default is "False".

        Returns
        -------
        tuple
            A tuple containing the processed image tensor.
        """
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        
        # Convert to RGBA if transparency processing is active and an alpha channel exists
        if is_transparent == "True" and 'A' in img.getbands():
            img = img.convert('RGBA')
            alpha = np.array(img.split()[-1]).astype(np.float32) / 255.0  # Extract alpha channel
            img = img.convert('RGB')  # Remove alpha channel
        else:
            img = img.convert('RGB')
            alpha = 1.0  # Full opacity

        image_np = np.array(img).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        # Apply transparency processing
        if is_transparent == "True":
            for c in range(3):  # For each RGB channel
                image_tensor[0, :, :, c] *= alpha

        return (image_tensor,)


NODE_CLASS_MAPPINGS = {
	"MathEncode": MathEncode,
	"LatentGaussianNoise": LatentGaussianNoise,
    "ImageLoaderAndProcessor": ImageLoaderAndProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MathEncode": MathEncode.TITLE,
    "LatentGaussianNoise": LatentGaussianNoise.TITLE,
    "ImageLoaderAndProcessor": ImageLoaderAndProcessor.TITLE,
}
