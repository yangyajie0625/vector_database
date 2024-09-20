import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from chromadb import PersistentClient
import fire


def image_to_embedding(image_path, clip_model, clip_processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings.squeeze().cpu().numpy()


def query_similar_images(model_path, query_image_path, vector_db_load_path, vector_db_name, k=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained(model_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_path)

    client = PersistentClient(path=vector_db_load_path)
    collection = client.get_collection(name=vector_db_name)
    print("The vector database has been loaded.")

    query_embedding = image_to_embedding(query_image_path, clip_model, clip_processor, device).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    for idx, result in enumerate(results['metadatas'][0]):
        print(f"Similar image {idx + 1}: Path is {result['path']}")


if __name__ == '__main__':
    fire.Fire(query_similar_images)

