import os
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import fire
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def batch_image_to_embedding(image_paths, clip_model, clip_processor, device, batch_size):
    embeddings = []
    valid_paths = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        current_valid_paths = []
        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                images.append(image)
                current_valid_paths.append(path)
            except Exception as e:
                print(f"Error processing image file: {path}, error: {e}")
                continue

        if not images:
            continue

        try:
            inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                batch_embeddings = clip_model.get_image_features(**inputs)
            embeddings.extend(batch_embeddings.cpu().numpy().tolist())
            valid_paths.extend(current_valid_paths)
        except Exception as e:
            print(f"Error processing images in batch {batch_paths}: {e}")
            continue

    return embeddings, valid_paths

def extract_features(model_path, source_image_dir, output_file, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained(model_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_path)

    all_images = []
    for root, dirs, files in os.walk(source_image_dir):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                all_images.append(os.path.join(root, file))

    embeddings_list, image_paths = batch_image_to_embedding(all_images, clip_model, clip_processor, device, batch_size)
    np.savez(output_file, embeddings=embeddings_list, image_paths=image_paths)
    print(f"embeddings have been saved to {output_file}")

if __name__ == '__main__':
    fire.Fire(extract_features)
