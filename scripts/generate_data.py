import os
import sys
import yaml
import pandas as pd
import scipy.io.wavfile
from datasets import load_dataset
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
    
    img_dir = os.path.join(output_dir, "images")
    audio_dir = os.path.join(output_dir, "audio")
    csv_path = os.path.join(output_dir, config['paths']['metadata_file'])
    
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
    synthesizer = DataSynthesizer(device=device)

    print("Loading ESC-50 dataset...")
    dataset = load_dataset("ashraq/esc50", split="train", streaming=True)

    dataset_iter = iter(dataset)
    for _ in range(start_index):
        next(dataset_iter, None)

    metadata = []
    count = 0
    total_processed = start_index
    
    pbar = tqdm(total=num_samples_to_generate - start_index, desc="Generating Data")
    
    while total_processed < num_samples_to_generate:
        
        sample = next(dataset_iter)
        
        category = sample['category'] 
        prompt = category.replace("_", " ") # Clean category prompt
        
        audio_array = sample['audio']['array']
        sample_rate = sample['audio']['sampling_rate']
        
        file_id = f"sample_{total_processed:04d}"
        img_filename = f"{file_id}.png"
        audio_filename = f"{file_id}.wav"
        
        img_path = os.path.join(img_dir, img_filename)
        audio_path = os.path.join(audio_dir, audio_filename)

        scipy.io.wavfile.write(audio_path, sample_rate, audio_array)
        synthesizer.generate_image(prompt=prompt, output_path=img_path)

        metadata.append({
            "id": file_id,
            "image_path": os.path.join("images", img_filename),
            "audio_path": os.path.join("audio", audio_filename),
            "caption": prompt,
            "category": category
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