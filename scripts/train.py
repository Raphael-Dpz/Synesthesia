import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor, CLIPImageProcessor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.dataset import SynesthesiaDataset
from src.models.synesthesia_model import SynesthesiaModel

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    # Create checkpoint directory if it doesn't exist
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)

    # Prepare Dataset and Dataloader
    print("[INFO] Loading Dataset...")
    train_dataset = SynesthesiaDataset(
        metadata_path=os.path.join(config['paths']['synthetic_dir'], config['paths']['metadata_file']),
        target_sample_rate=48000, # CLAP requires 48kHz
        max_audio_length=30 
    )
    
    def collate_fn(batch):
        return {
            'image': [item['image'] for item in batch],
            'audio': torch.stack([item['audio'] for item in batch]),
            'caption': [item['caption'] for item in batch]
        }
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Initialize Models
    
    # A. Synesthesia model (Image -> Audio Embedding)
    print("[INFO] Initializing Synesthesia Model...")
    synesthesia_model = SynesthesiaModel(config).to(device)
    synesthesia_model.train() # Set to training mode

    image_processor = CLIPImageProcessor.from_pretrained(config['model']['image_encoder'])


    # B. Audio encoder model (Audio -> Audio Embedding)
    print("[INFO] Initializing Audio encoder model...")
    clap_model_name = config['model']['audio_encoder'] # e.g. "laion/clap-htsat-unfused"
    audio_encoder = ClapModel.from_pretrained(clap_model_name).to(device)
    audio_encoder.eval()
    for param in audio_encoder.parameters(): # freeze the audio encoder
        param.requires_grad = False
    
    audio_processor = ClapProcessor.from_pretrained(clap_model_name)



    optimizer = optim.AdamW(synesthesia_model.parameters(), lr=float(config['training']['learning_rate']))
    criterion = nn.MSELoss() 


    # Training Loop
    epochs = config['training']['epochs']
    print(f"[INFO] Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            # Process dataset
            
            # A. Prepare Images for Synesthesia model (CLIP)
            image_inputs = image_processor(images=batch['image'], return_tensors="pt")
            pixel_values = image_inputs['pixel_values'].to(device)
            
            # B. Prepare Audio for the encoder (CLAP)
            raw_audio = [x.numpy() for x in batch['audio']]
            audio_inputs = audio_processor(audio=raw_audio, sampling_rate=48000, return_tensors="pt", padding=True)

            if 'input_features' in audio_inputs:
                audio_features = audio_inputs['input_features'].to(device)
            else:
                audio_features = audio_inputs['input_values'].to(device)

            # Forward pass
            
            with torch.no_grad():
                outputs = audio_encoder.audio_model(input_features=audio_features)
                audio_embeddings = audio_encoder.audio_projection(outputs.pooler_output).detach() # [Batch, 512]

            pred_embeddings = synesthesia_model(pixel_values)

            # Loss & Backpropagation
            
            loss = criterion(audio_embeddings, pred_embeddings)

            optimizer.zero_grad() # Reset gradients
            loss.backward()       # Calculate gradients
            optimizer.step()      # Update weights

            # Logging loss
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})


        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.6f}")

        # Save Checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], f"synesthesia_epoch_{epoch+1}.pt")
            torch.save(synesthesia_model.state_dict(), checkpoint_path)
            print(f"[SAVE] Model saved to {checkpoint_path}")

    print("[SUCCESS] Training completed")

if __name__ == "__main__":
    main()
            

