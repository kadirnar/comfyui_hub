class ImageProcessingWithTransparency:
    """
    Image processing node with transparency handling.

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Specify the input parameters of the node.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tuple.
    FUNCTION (`str`):
        The name of the entry-point method.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """
            Return a dictionary which contains config for all input fields.
        """
        return {
            "required": {
                "image_path": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "is_transparent": (["True", "False"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    FUNCTION = "process_image"
    CATEGORY = "Image Processing"

    def process_image(self, image_path, is_transparent):
        from PIL import Image, ImageOps
        import numpy as np
        import torch

        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        image = img.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        if is_transparent == "True" and 'A' in img.getbands():
            alpha = np.array(img.getchannel('A')).astype(np.float32) / 255.0
            alpha_tensor = torch.from_numpy(alpha)
            mask = 1.0 - alpha_tensor

            smooth_image = image_tensor.detach().clone()
            for c in range(3):  # RGB channels
                smooth_image[0, :, :, c] *= alpha_tensor
            return (image_tensor, mask, smooth_image)
        else:
            mask = torch.zeros((image_tensor.shape[2], image_tensor.shape[3]), dtype=torch.float32)
            return (image_tensor, mask, image_tensor)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ImageProcessingWithTransparency": ImageProcessingWithTransparency
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageProcessingWithTransparency": "Image Processing with Transparency"
}
