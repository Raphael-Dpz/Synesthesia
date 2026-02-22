import os
import yaml
import torch
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor
from diffusers import AudioLDMPipeline

# Model import
from src.models.synesthesia_model import SynesthesiaModel

class SynesthesiaPipeline:
    """
    End-to-end pipeline for generating audio from images using a trained SynesthesiaModel and AudioLDM.
    It handles model loading, image processing, and audio generation in a seamless manner.
    """
    def __init__(self, checkpoint_path, config_path="configs/config.yaml"):
        # config loading
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Synesthesia Pipeline on {self.device}...")

        # A. Image processor and Synesthesia Model
        print("[INFO] Loading Synesthesia Model...")
        self.model = SynesthesiaModel(self.config).to(self.device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True), 
            strict=False
        )
        self.model.eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(self.config['model']['image_encoder'])

        # B. Audio generator from latent vector: AudioLDM 
        print("[INFO] Loading Audio Synthesizer (AudioLDM)...")
        self.audio_pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2", torch_dtype=torch.float32)
        self.audio_pipe = self.audio_pipe.to(self.device)
        
        if self.device == "cpu":
            self.audio_pipe.enable_attention_slicing()
            
        print("Pipeline ready!")

    def generate(self, image_input, duration=5.0, steps=20):
        """
        Generates an audio clip from a given image input.
        
        ==========
        Args:
            image_input (str or PIL.Image): Path to the input image or a PIL Image object.
            duration (float): Desired duration of the output audio in seconds (default: 5.0s).
            steps (int): Number of inference steps for AudioLDM (default: 20).
            
        ==========
        Returns:
            audio_output (numpy array): Generated audio waveform as a 1D numpy array.
            sample_rate (int): Sample rate of the generated audio (AudioLDM generates at 16000 Hz).
        """
        # A. Image loading and processing
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')

        # B. Latent vector prediction with SynesthesiaModel
        with torch.no_grad():
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
            predicted_embedding = self.model(inputs['pixel_values'])
            
            # Normalize the predicted embedding to unit length (AudioLDM expects normalized embeddings)
            predicted_embedding = predicted_embedding / predicted_embedding.norm(dim=-1, keepdim=True)
            prompt_embeds = predicted_embedding
            negative_embeds = torch.zeros_like(prompt_embeds)

        # C. Generates audio
        audio_output = self.audio_pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            num_inference_steps=steps, 
            audio_length_in_s=duration
        ).audios[0]

        # AudioLDM always generates at 16000 Hz
        return audio_output, 16000