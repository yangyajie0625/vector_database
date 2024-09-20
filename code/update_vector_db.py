import os
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel
from chromadb import PersistentClient
from tqdm import tqdm
import fire

Image.MAX_IMAGE_PIXELS = None
def batch_image_to_embedding(image_paths, clip_model, clip_processor, device, batch_size):
    embeddings = []
    valid_paths = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                images.append(image)
                valid_paths.append(path)
            except UnidentifiedImageError:
                print(f"Cannot identify image file: {path}")
                continue

        if not images:
            continue

        inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            batch_embeddings = clip_model.get_image_features(**inputs)
        embeddings.extend(batch_embeddings.cpu().numpy().tolist())

    return embeddings, valid_paths


def update_vector_db(model_path, dataset_path, vector_db_path, vector_db_name, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained(model_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_path)

    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, filename))

    image_embeddings, valid_image_paths = batch_image_to_embedding(image_paths, clip_model, clip_processor, device, batch_size)
    metadata = [{"path": img_path} for img_path in valid_image_paths]

    client = PersistentClient(path=vector_db_path)
    collection = client.get_collection(name=vector_db_name)

    existing_ids = collection.get()['ids']
    if existing_ids:
        max_existing_id = max(map(int, existing_ids))
    else:
        max_existing_id = -1

    max_batch_size = 41660
    for i in range(0, len(image_embeddings), max_batch_size):
        batch_embeddings = image_embeddings[i:i + max_batch_size]
        batch_metadata = metadata[i:i + max_batch_size]
        batch_ids = [str(j) for j in range(max_existing_id + 1 + i, max_existing_id + 1 + i + len(batch_embeddings))]

        collection.add(
            embeddings=batch_embeddings,
            metadatas=batch_metadata,
            ids=batch_ids
        )
    print(f"The vector database has been updated.")


if __name__ == '__main__':
    fire.Fire(update_vector_db)
