import os
import torch
import pandas as pd
import librosa
from PIL import Image
from torch.utils.data import Dataset

class SynesthesiaDataset(Dataset):
    """
    Dataset loader robuste utilisant Librosa pour Windows.
    Gère automatiquement le resampling et le mixage mono.
    """
    def __init__(self, metadata_path: str, transform=None, target_sample_rate=48000, max_audio_length=5):
        self.metadata = pd.read_csv(metadata_path)
        self.root_dir = os.path.dirname(metadata_path)
        self.transform = transform
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
            image = Image.open(image_path).convert("RGB")

            # Audio
            # A. Resample and mix to mono using Librosa (more robust than torchcodec on Windows)
            audio_array, _ = librosa.load(audio_path, sr=self.target_sample_rate, mono=True)
            
            waveform = torch.from_numpy(audio_array)

            # B. Pad or Truncate
            target_length = self.target_sample_rate * self.max_audio_length
            current_length = waveform.shape[0]

            if current_length > target_length:
                waveform = waveform[:target_length] # too long: cut
            elif current_length < target_length:
                padding = target_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, padding)) # too short: pad

            return {
                "image": image,      # PIL Image
                "audio": waveform,   # Tensor 1D ([240000] if 5 seconds at 48kHz)
                "caption": row['caption']
            }

        except Exception as e:
            print(f"[ERROR] Impossible de lire {idx}: {e}")
            if idx != 0: 
                return self.__getitem__(0)
            else:
                raise e