import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import fire

def image_to_embedding(image_path, clip_model, clip_processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings.squeeze().cpu().numpy()

def extract_features(model_path, source_image_dir, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained(model_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_path)

    all_images = [os.path.join(source_image_dir, img) for img in os.listdir(source_image_dir) if
                  img.lower().endswith(('jpg', 'jpeg', 'png'))]

    embeddings_list = []
    image_paths = []
    for img_path in all_images:
        embedding = image_to_embedding(img_path, clip_model, clip_processor, device)
        embeddings_list.append(embedding)
        image_paths.append(img_path)

    np.savez(output_file, embeddings=embeddings_list, image_paths=image_paths)
    print(f"embeddings have been saved to {output_file}")

if __name__ == '__main__':
    fire.Fire(extract_features)
