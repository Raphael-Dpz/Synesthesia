import os
import torch
import pandas as pd
import librosa
from PIL import Image
from torch.utils.data import Dataset

class SynesthesiaDataset(Dataset):
    """
    Dataset loader pour Clotho (Image + Audio 15-30s).
    Prend l'audio entier et applique un padding (silence) pour unifier la taille.
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
            
            # Chemins Clotho
            image_path = os.path.join(self.root_dir, row['image_path'])
            audio_path = os.path.join(self.root_dir, row['audio_path'])

            # --- 1. IMAGE ---
            image = Image.open(image_path).convert("RGB")

            # --- 2. AUDIO (Chargement complet) ---
            # Librosa charge tout le fichier en 48kHz Mono
            audio_array, _ = librosa.load(audio_path, sr=self.target_sample_rate, mono=True)
            waveform = torch.from_numpy(audio_array)

            # --- 3. UNIFICATION DE LA TAILLE (30 SECONDES) ---
            # 30 secondes * 48000 Hz = 1 440 000 points
            target_length = self.target_sample_rate * self.max_audio_length 
            current_length = waveform.shape[0]

            if current_length > target_length:
                # Si par hasard un fichier dépasse 30s, on le coupe pile à 30s
                waveform = waveform[:target_length]
            elif current_length < target_length:
                # S'il fait moins de 30s, on ajoute du silence (zéros) à la fin
                padding = target_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            return {
                "image": image,
                "audio": waveform,
                "caption": row['caption']
            }

        except Exception as e:
            print(f"[ERROR] Impossible de charger l'index {idx}: {e}")
            # Sécurité : on renvoie le premier élément si celui-ci est corrompu
            if idx != 0: 
                return self.__getitem__(0)
            else:
                raise e