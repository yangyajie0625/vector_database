import os
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import fire
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None


def batch_image_to_embedding(image_paths, clip_model, clip_processor, device, batch_size):
    """
        Processes a list of image file paths in batches, computes image embeddings using the CLIP model.
        Args:
            image_paths (list): A list of file paths to the images that need to be processed.
            clip_model (CLIPModel): A pre-trained CLIP model used to compute the image embeddings.
            clip_processor (CLIPProcessor): A processor to preprocess images before passing them to the CLIP model.
            device (torch.device): The device (CPU or GPU) on which the computations will be performed.
            batch_size (int): The number of images to process in each batch.

        Returns:
            tuple: A tuple containing:
                - embeddings (list of lists): A list of computed image embeddings, where each embedding
                  is a list of floats representing the image in vector space.
                - valid_paths (list of str): A list of file paths corresponding to the images
                  that were successfully processed and embedded.
        """

    embeddings = []  # List to store image embeddings
    valid_paths = []  # List to store valid image file paths, ensuring the lengths of embeddings and valid paths are consistent, since only successfully processed images are included.
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i:i + batch_size]
        images = []  # List to store loaded images
        current_valid_paths = []  # List to store valid paths in the current batch
        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                images.append(image)
                current_valid_paths.append(path)
            except Exception as e:
                print(f"Error processing image file: {path}, error: {e}")
                continue

        if not images:
            # Skip the batch if no valid images were found
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
    # Walk through the source directory to find all images with specified extensions
    for root, dirs, files in os.walk(source_image_dir):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                all_images.append(os.path.join(root, file))  # Add valid image paths


    embeddings_list, image_paths = batch_image_to_embedding(all_images, clip_model, clip_processor, device, batch_size)
    # Save the embeddings and valid image paths to a .npz file. The .npz file will contain two main arrays: embeddings and image_paths.
    # Each index in image_paths corresponds directly to the same index in embeddings.
    np.savez(output_file, embeddings=embeddings_list, image_paths=image_paths)
    print(f"Embeddings have been saved to {output_file}")

if __name__ == '__main__':
    fire.Fire(extract_features)
