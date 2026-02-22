import os
import torch
import pandas as pd
import librosa
from PIL import Image
from torch.utils.data import Dataset

class SynesthesiaDataset(Dataset):
    """
    Dataset loader for Clotho (Image + Audio 15-30s).
    Takes care of loading, preprocessing, and unifying the length of audio samples.
    
    ==========
    Returns:
        A dictionary with:
            - "image": PIL Image object
            - "audio": 1D torch.Tensor of shape (target_sample_rate * max_audio_length,)
            - "caption": Raw caption string from the dataset
    
    """
    def __init__(self, metadata_path: str, transform=None, target_sample_rate=48000, max_audio_length=30):
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

            # A. Image
            image = Image.open(image_path).convert("RGB")

            # B. Audio
            # Librosa loads the audio in 48kHz Mono
            audio_array, _ = librosa.load(audio_path, sr=self.target_sample_rate, mono=True)
            waveform = torch.from_numpy(audio_array)

            # length unification at max_audio_length (=30s -> 1 440 000 points at 48kHz)
            target_length = self.target_sample_rate * self.max_audio_length 
            current_length = waveform.shape[0]

            if current_length > target_length:
                waveform = waveform[:target_length]                          # cut if it exceeds 30s
            elif current_length < target_length:
                padding = target_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, padding))   # pad if it is shorter than 30s

            return {
                "image": image,
                "audio": waveform,
                "caption": row['caption']
            }

        except Exception as e:
            print(f"[ERROR] failed to load element n°{idx}: {e}")
            if idx != 0: 
                return self.__getitem__(0)
            else:
                raise e