import os
import sys
import yaml
import pandas as pd
import shutil
from tqdm import tqdm

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_gen.synthesizer import DataSynthesizer

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_metadata(metadata_list, csv_path):
    if not metadata_list:
        return
    df = pd.DataFrame(metadata_list)
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=header, index=False)

def main():
    config = load_config()
    output_dir = config['paths']['synthetic_dir']
    num_samples_to_generate = config['generation']['num_samples']
    steps = config['generation']['steps']
    guidance_scale = config['generation']['guidance_scale']
    
    img_dir = os.path.join(output_dir, "images")
    audio_dir = os.path.join(output_dir, "audio")
    csv_path = os.path.join(output_dir, config['paths']['metadata_file'])
    
    # --- CONFIGURATION CLOTHO ---
    clotho_csv = "data/raw/clotho_captions_development.csv"
    clotho_audio_dir = "data/raw/development"
    
    if not os.path.exists(clotho_csv):
        print(f"[ERROR] can't find clotho at {clotho_csv}. Must be donwloaded and extracted there")
        return

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    existing_files = len([name for name in os.listdir(img_dir) if name.endswith('.png')])
    start_index = existing_files
    
    print(f"Found {existing_files} existing pairs.")
    print(f"Target: Generate {num_samples_to_generate} pairs.")

    if existing_files >= num_samples_to_generate:
        print("Target reached.")
        return

    device = config['training']['device']
    synthesizer = DataSynthesizer(steps=steps, guidance_scale=guidance_scale, device=device)

    print("Loading Clotho dataset metadata...")
    df_clotho = pd.read_csv(clotho_csv)
    
    # Skipping already processed samples
    df_clotho = df_clotho.iloc[start_index:]

    metadata = []
    count = 0
    total_processed = start_index
    
    pbar = tqdm(total=num_samples_to_generate - start_index, desc="Generating Data")
    
    for index, row in df_clotho.iterrows():
        if total_processed >= num_samples_to_generate:
            break
            
        # Clotho uses 'file_name' for the audio currently 'caption_1' for the prompt
        orig_audio_name = row['file_name']
        prompt = row['caption_1'] # other captions for data augmentation could be used here
        
        orig_audio_path = os.path.join(clotho_audio_dir, orig_audio_name)
        
        if not os.path.exists(orig_audio_path):
            print(f"\n[WARN] Current audio can't be found at: {orig_audio_path}. Skipping...")
            continue
        
        file_id = f"sample_{total_processed:04d}"
        img_filename = f"{file_id}.png"
        audio_filename = f"{file_id}.wav"
        
        img_path = os.path.join(img_dir, img_filename)
        audio_path = os.path.join(audio_dir, audio_filename)

        shutil.copy(orig_audio_path, audio_path)

        # Generating image from prompt using synthesizer
        synthesizer.generate_image(prompt=prompt, output_path=img_path)

        # Creating metadata entry to be saved in csv in batches later
        metadata.append({
            "id": file_id,
            "image_path": os.path.join("images", img_filename),
            "audio_path": os.path.join("audio", audio_filename),
            "caption": prompt,
            "category": "clotho"
        })
        
        count += 1
        total_processed += 1
        pbar.update(1)

        if count % 5 == 0:
            save_metadata(metadata, csv_path)
            metadata = [] 
            
    pbar.close()

    if metadata:
        save_metadata(metadata, csv_path)
    
    print(f"Generation complete. Total pairs: {total_processed}")

if __name__ == "__main__":
    main()