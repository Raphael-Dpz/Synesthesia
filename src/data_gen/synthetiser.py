import torch
from diffusers import StableDiffusionPipeline
import os

class DataSynthesizer:
    """
    Wrapper class for Stable Diffusion.
    It handles model loading, memory optimization, and image generation.
    """

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda"):
        """
        Initializes the Stable Diffusion pipeline.

        Args:
            model_id (str): The Hugging Face model ID (default: SD v1.5).
            device (str): Computation device ('cuda' for GPU or 'cpu').
        """
        self.device = device
        print(f"[INFO] Loading Text-to-Image model: {model_id}...")

        # We use float16 (half precision) if on GPU to save ~50% VRAM and speed up generation.
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=dtype,
                use_safetensors=True
            )
            self.pipe.to(self.device)
            
            # Disable the Safety Checker
            if hasattr(self.pipe, 'safety_checker') and self.pipe.safety_checker is not None:
                self.pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
            
            print(f"Model loaded successfully on {self.device}.")
            
        except Exception as e:
            print(f"Failed to load Stable Diffusion: {e}")
            raise e

    def generate_image(self, prompt: str, output_path: str) -> None:
        """
        Generates an image from a text prompt and saves it to disk.

        Args:
            prompt (str): The raw caption from AudioCaps (e.g., "Wind blowing in trees").
            output_path (str): Full path where the .png will be saved.
        """
        # Prompt Engineering
        enhanced_prompt = f"{prompt}, realistic, 4k, high detailed, cinematic lighting, atmospheric"
        
        # Negative prompt
        negative_prompt = "low quality, blurry, distorted, text, watermark, signature, bad anatomy"

        # Inference
        with torch.no_grad():
            image = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=30, # Balance between speed (20) and quality (50)
                guidance_scale=7.5      # How strictly to follow the prompt
            ).images[0]

        image.save(output_path)