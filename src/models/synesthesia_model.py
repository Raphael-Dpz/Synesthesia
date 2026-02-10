import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class SynesthesiaModel(nn.Module):
    def __init__(self, config):
        """
        Model of Synesthesia: Image -> Audio features
        Uses a frozen CLIP encoder and a MLP trainable head
        """
        super(SynesthesiaModel, self).__init__()

        # Load Image encoder
        model_name = config['model']['image_encoder'] # e.g. "openai/clip-vit-base-patch32"
        print(f"[INFO] Loading Image Encoder: {model_name}...")
        self.image_encoder = CLIPVisionModel.from_pretrained(model_name)
        
        # Freeze Image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
            
        input_dim = config['model']['input_dim']   # 768 for CLIP ViT-Base
        hidden_dim = config['model']['hidden_dim']
        output_dim = config['model']['output_dim'] # 512 for CLAP
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),              
            nn.Dropout(0.1),  
            nn.Linear(hidden_dim, output_dim) 
        )
        
    def forward(self, images):
        img_features = self.image_encoder(images) # [Batch, 768]
        audio_features = self.projection(img_features) # [Batch, 512]
        
        return audio_features