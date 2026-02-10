import os
import torch
import pandas as pd
import torchaudio
from PIL import Image
from torch.utils.data import Dataset

class SynesthesiaDataset(Dataset):
    """
    Dataset loader for Image-to-Audio.
    - Images: Returns raw PIL images (RGB).
    - Audio: Returns raw waveforms (Tensor), resampled to 48kHz, mono, fixed length.
    - Captions: Returns text captions (not used during training).
    """
    def __init__(self, metadata_path: str, transform=None, target_sample_rate=48000, max_audio_length=5):
        self.metadata = pd.read_csv(metadata_path)
        self.root_dir = os.path.dirname(metadata_path)
        self.transform = transform # Usually not used since CLIPProcessor handles it during training
        self.target_sample_rate = target_sample_rate
        self.max_audio_length = max_audio_length

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        try:
            row = self.metadata.iloc[idx]

            image_path = os.path.join(self.root_dir, row['image_path'])
            audio_path = os.path.join(self.root_dir, row['audio_path'])

            # Image
            image = Image.open(image_path).convert("RGB") # raw for more flexibility wrt the Image encoder model

            # Audio
            waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile")
            
            # A. Resample
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)

            # B. Mix to Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # C. Pad or Truncate
            num_samples = self.target_sample_rate * self.max_audio_length
            
            if waveform.shape[1] > num_samples:
                waveform = waveform[:, :num_samples] # too long: cut
            elif waveform.shape[1] < num_samples:
                padding = num_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding)) # too short: pad

            waveform = waveform.squeeze(0)

            return {
                "image": image,      # PIL Image
                "audio": waveform,   # Tensor 1D ([240000] if 5 seconds at 48kHz)
                "caption": row['caption']
            }

        except Exception as e:
            print(f"[WARN] Failed to load index {idx}: {e}. Skipping...")
            return self.__getitem__((idx + 1) % len(self))