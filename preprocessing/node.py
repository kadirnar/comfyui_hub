import os
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
import folder_paths
import torch

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
    CATEGORY = "ComfyUI-Helpers/Preprocessing"
    TITLE = "LoadImagePlus"

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


class ImageToContrastMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_threshold": ("INT", {
                    "default": 1, 
                    "min": 1,
                    "max": 255,
                    "step": 1
                }),
                "high_threshold": ("INT", {
                    "default": 255, 
                    "min": 1,
                    "max": 255,
                    "step": 1
                }),
                "blur_radius": ("INT", {
                    "default": 1, 
                    "min": 1,
                    "max": 32768,
                    "step": 1
                })
            },
        }

    RETURN_TYPES = ("IMAGE","MASK",)
    FUNCTION = "image_to_contrast_mask"

    CATEGORY = "ComfyUI-Helpers/Preprocessing"

    def image_to_contrast_mask(self, image, low_threshold, high_threshold, blur_radius):
        image = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        image = ImageOps.grayscale(image)

        if blur_radius > 1:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        #high_filter = lambda x: 255 if x > high_threshold else x
        #image = image.convert("L").point(high_filter, mode="L")

        #low_filter = lambda x: 0 if x < low_threshold else x
        #image = image.convert("L").point(high_filter, mode="L")

        filter = lambda x: 255 if x > high_threshold else 0 if x < low_threshold else x
        image = image.convert("L").point(filter, mode="L")

        image = np.array(image).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(image)
        image = torch.from_numpy(image)[None,]
        
        return (image, mask,)
