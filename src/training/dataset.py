import os
import torch
import pandas as pd
import torchaudio
from PIL import Image
from torch.utils.data import Dataset

class SynesthesiaDataset(Dataset):
    """
    Dataset loader for the Synthetic Synesthesia Dataset (ESC-50 based).
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

        row = self.metadata.iloc[idx]
        
        # Paths
        image_path = os.path.join(self.root_dir, row['image_path'])
        audio_path = os.path.join(self.root_dir, row['audio_path'])

        # Load Image
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"[ERROR] Image load failed: {image_path}")
            return self.__getitem__((idx + 1) % len(self))

        # Load Audio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample to 48kHz (CLAP standard)
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)

            # Pad or Truncate to fixed length (5 seconds)
            num_samples = self.target_sample_rate * self.max_audio_length
            
            if waveform.shape[1] > num_samples:
                waveform = waveform[:, :num_samples]
            elif waveform.shape[1] < num_samples:
                padding = num_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            # Mix to Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

        except Exception as e:
            print(f"[ERROR] Audio load failed: {audio_path}")
            return self.__getitem__((idx + 1) % len(self))

        return {
            "image": image,
            "audio": waveform,
            "caption": row['caption']
        }