import os
from PIL import Image, ImageOps
import numpy as np
import torch
import folder_paths

class TransparentImageProcessor:
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
    FUNCTION = "process_image"
    CATEGORY = "comfyui-image-utils"

    def process_image(self, image, is_transparent="False"):
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
    "ImageProcessingWithTransparency": TransparentImageProcessor
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageProcessingWithTransparency": "Transparent Image Processor"
}
