import os
import sys
import yaml
import pandas as pd
import scipy.io.wavfile
from datasets import load_dataset
from tqdm import tqdm

# Add project root to python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_gen.synthesizer import DataSynthesizer

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Load Configuration
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
    print(f"Starting generation from index {start_index} to {start_index + num_samples_to_generate}...")

    # Initialize Synthesizer
    synthesizer = DataSynthesizer(device=config['training']['device'])

    # Load AudioCaps and skip duplicates
    dataset = load_dataset("audiocaps", split="train", streaming=True)
    dataset_iter = iter(dataset)
    for _ in range(start_index):
        next(dataset_iter, None)

    metadata = []
    count = 0
    total_processed = start_index
    
    pbar = tqdm(total=num_samples_to_generate, desc="Generating New Data")
    
    while count < num_samples_to_generate:
        try:
            sample = next(dataset_iter)
            
            original_caption = sample['caption']
            audio_array = sample['audio']['array']
            sample_rate = sample['audio']['sampling_rate']
            
            # File Naming 
            file_id = f"sample_{total_processed:04d}"
            img_filename = f"{file_id}.png"
            audio_filename = f"{file_id}.wav"
            
            img_path = os.path.join(img_dir, img_filename)
            audio_path = os.path.join(audio_dir, audio_filename)

            # Save Audio & Generate Image
            scipy.io.wavfile.write(audio_path, sample_rate, audio_array)
            synthesizer.generate_image(prompt=original_caption, output_path=img_path)

            # Temporary list (saved and reset every 10 or so items)
            metadata.append({
                "id": file_id,
                "image_path": os.path.join("images", img_filename),
                "audio_path": os.path.join("audio", audio_filename),
                "caption": original_caption
            })
            
            count += 1
            total_processed += 1
            pbar.update(1)

            # Save to CSV every 10 items
            if count % 10 == 0:
                 save_metadata(metadata, csv_path)
                 metadata = []

        except StopIteration:
            print("End of dataset reached")
            break
        except Exception as e:
            print(f"Failed on {file_id}: {e}")
            continue
            
    pbar.close()

    # Save remaining metadata
    if metadata:
        save_metadata(metadata, csv_path)
    
    print(f"Generation complete. Total dataset size: {total_processed}")

def save_metadata(metadata_list, csv_path):
    """Helper function to add generated data to the csv file."""
    df = pd.DataFrame(metadata_list)
    
    # If file does not exist, write header. If it exists, append without header.
    header = not os.path.exists(csv_path)
    
    df.to_csv(csv_path, mode='a', header=header, index=False)

if __name__ == "__main__":
    main()