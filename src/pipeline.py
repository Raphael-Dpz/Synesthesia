import os
import yaml
import torch
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor
from diffusers import AudioLDMPipeline

# Import de ton modèle
from src.models.synesthesia_model import SynesthesiaModel

class SynesthesiaPipeline:
    """
    Pipeline End-to-End pour générer du son à partir d'une image.
    À instancier une seule fois dans un Notebook.
    """
    def __init__(self, checkpoint_path, config_path="configs/config.yaml"):
        # 1. Configuration & Device
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️ Initializing Synesthesia Pipeline on {self.device}...")

        # 2. Charger le Cerveau Visuel (Ton modèle)
        print("👁️ Loading Vision Model...")
        self.model = SynesthesiaModel(self.config).to(self.device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True), 
            strict=False
        )
        self.model.eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(self.config['model']['image_encoder'])

        # 3. Charger le Générateur Audio (AudioLDM)
        print("🎹 Loading Audio Synthesizer (AudioLDM)...")
        self.audio_pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2", torch_dtype=torch.float32)
        self.audio_pipe = self.audio_pipe.to(self.device)
        
        if self.device == "cpu":
            self.audio_pipe.enable_attention_slicing()
            
        print("✅ Pipeline ready!")

    def generate(self, image_input, duration=5.0, steps=20):
        """
        Génère l'audio.
        image_input: peut être un chemin (str) ou directement un objet PIL Image.
        Retourne: un tuple (audio_array, sample_rate) prêt à être lu.
        """
        # A. Préparer l'image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')

        # B. Prédire le vecteur latent
        with torch.no_grad():
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
            predicted_embedding = self.model(inputs['pixel_values'])
            
            # Normalisation et formatage pour AudioLDM
            predicted_embedding = predicted_embedding / predicted_embedding.norm(dim=-1, keepdim=True)
            prompt_embeds = predicted_embedding
            negative_embeds = torch.zeros_like(prompt_embeds)

        # C. Générer le son
        audio_output = self.audio_pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            num_inference_steps=steps, 
            audio_length_in_s=duration
        ).audios[0]

        # AudioLDM génère toujours en 16000 Hz
        return audio_output, 16000