# https://github.com/Jordach/comfy-plasma/blob/main/nodes.py

from xml.dom import minidom
import json
import torch
import numpy as np
from PIL import Image
import comfy

def remap(val, min_val, max_val, min_map, max_map):
	return (val-min_val)/(max_val-min_val) * (max_map-min_map) + min_map

def conv_pil_tensor(img):
	return (torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0),)

def conv_tensor_pil(tsr):
	return Image.fromarray(np.clip(255. * tsr.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def clamp(val, min, max):
	if val < min:
		return min
	elif val > max:
		return max
	else:
		return val

def get_pil_resampler(resampler):
	if resampler == "nearest":
		return Image.Resampling.NEAREST
	elif resampler == "box":
		return Image.Resampling.BOX
	elif resampler == "bilinear":
		return Image.Resampling.BILINEAR
	elif resampler == "bicubic":
		return Image.Resampling.BICUBIC
	elif resampler == "hamming":
		return Image.Resampling.HAMMING
	elif resampler == "lanczos":
		return Image.Resampling.LANCZOS
	else:
		return Image.Resampling.NEAREST

# borrowed from https://github.com/receyuki/stable-diffusion-prompt-reader/blob/master/sd_prompt_reader/image_data_reader.py
EASYDIFFUSION_MAPPING_A = {
	"prompt": "Prompt",
	"negative_prompt": "Negative Prompt",
	"seed": "Seed",
	"use_stable_diffusion_model": "Stable Diffusion model",
	"clip_skip": "Clip Skip",
	"use_vae_model": "VAE model",
	"sampler_name": "Sampler",
	"width": "Width",
	"height": "Height",
	"num_inference_steps": "Steps",
	"guidance_scale": "Guidance Scale",
}

EASYDIFFUSION_MAPPING_B = {
	"prompt": "prompt",
	"negative_prompt": "negative_prompt",
	"seed": "seed",
	"use_stable_diffusion_model": "use_stable_diffusion_model",
	"clip_skip": "clip_skip",
	"use_vae_model": "use_vae_model",
	"sampler_name": "sampler_name",
	"width": "width",
	"height": "height",
	"num_inference_steps": "num_inference_steps",
	"guidance_scale": "guidance_scale",
}

def handle_auto1111(params):
	if params and "\nSteps:" in params:
		# has a negative:
		if "Negative prompt:" in params:
			prompt_index = [params.index("\nNegative prompt:"), params.index("\nSteps:")]
			neg = params[prompt_index[0] + 1 + len("Negative prompt: "):prompt_index[-1]]
		else:
			index = [params.index("\nSteps:")]
			neg = ""

		pos = params[:prompt_index[0]]
		return pos, neg
	elif params:
		# has a negative:
		if "Negative prompt:" in params:
			prompt_index = [params.index("\nNegative prompt:")]
			neg = params[prompt_index[0] + 1 + len("Negative prompt: "):]
		else:
			index = [len(params)]
			neg = ""
		
		pos = params[:prompt_index[0]]
		return pos, neg
	else:
		return "", ""

def handle_ezdiff(params):
	data = json.loads(params)
	if data.get("prompt"):
		ed = EASYDIFFUSION_MAPPING_B
	else:
		ed = EASYDIFFUSION_MAPPING_A

	pos = data.get(ed["prompt"])
	data.pop(ed["prompt"])
	neg = data.get(ed["negative_prompt"])
	return pos, neg

def handle_invoke_modern(params):
	meta = json.loads(params.get("sd-metadata"))
	img = meta.get("image")
	prompt = img.get("prompt")
	index = [prompt.rfind("["), prompt.rfind("]")]

	# negative
	if -1 not in index:
		pos = prompt[:index[0]]
		neg = prompt[index[0] + 1:index[1]]
		return pos, neg
	else:
		return prompt, ""

def handle_invoke_legacy(params):
	dream = params.get("Dream")
	pi = dream.rfind('"')
	ni = [dream.rfind("["), dream.rfind("]")]

	# has neg
	if -1 not in ni:
		pos = dream[1:ni[0]]
		neg = dream[ni[0] + 1:ni[1]]
		return pos, neg
	else:
		pos = dream[1:pi]
		return pos, ""

def handle_novelai(params):
	pos = params.get("Description")
	comment = params.get("Comment") or {}
	comment_json = json.loads(comment)
	neg = comment_json.get("uc")
	return pos, neg

def handle_qdiffusion(params):
	pass

def handle_drawthings(params):
	try:
		data = minidom.parseString(params.get("XML:com.adobe.xmp"))
		data_json = json.loads(data.getElementByTagName("exif:UserComment")[0].childNodes[1].childNodes[1].childNodes[0].data)
	except:
		return "", ""
	else:
		pos = data_json.get("c")
		neg = data_json.get("uc")
		return pos, neg


# Torch rand noise
def prepare_rand_noise(latent_image, seed, noise_inds=None):
	"""
	creates random noise given a latent image and a seed.
	optional arg skip can be used to skip and discard x number of noise generations for a given seed
	"""
	generator = torch.manual_seed(seed)
	if noise_inds is None:
		return (torch.rand(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu") - 0.5) * 2 * 1.73
	
	unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
	noises = []
	for i in range(unique_inds[-1]+1):
		noise = (torch.rand(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu") - 0.5) * 2 * 1.73
		if i in unique_inds:
			noises.append(noise)
	noises = [noises[i] for i in inverse]
	noises = torch.cat(noises, axis=0)
	return noises

# Modified ComfyUI sampler
def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, latent_noise, use_rand=False, start_step=None, last_step=None):
	device = comfy.model_management.get_torch_device()
	latent_image = latent["samples"]

	noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")

	if latent_noise > 0:
		batch_inds = latent["batch_index"] if "batch_index" in latent else None
		if use_rand:
			noise = noise + (prepare_rand_noise(latent_image, seed, batch_inds) * latent_noise)
		else:
			noise = noise + (comfy.sample.prepare_noise(latent_image, seed, batch_inds) * latent_noise)

	noise_mask = None
	if "noise_mask" in latent:
		noise_mask = latent["noise_mask"]

	pbar = comfy.utils.ProgressBar(steps)
	def callback(step, x0, x, total_steps):
		pbar.update_absolute(step + 1, total_steps)

	samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
								  denoise=denoise, disable_noise=True, start_step=start_step, last_step=last_step,
								  force_full_denoise=False, noise_mask=noise_mask, callback=callback)
	out = latent.copy()
	out["samples"] = samples
	return (out, )