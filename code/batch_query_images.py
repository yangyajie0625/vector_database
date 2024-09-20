import os
import random
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from chromadb import PersistentClient
import fire
import time

def image_to_embedding(image_path, clip_model, clip_processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings.squeeze().cpu().numpy()


def query_similar_images(model_path, vector_db_load_path, vector_db_name, source_image_dir, sample_num, k=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained(model_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_path)

    client = PersistentClient(path=vector_db_load_path)
    collection = client.get_collection(name=vector_db_name)
    print("The vector database has been loaded.")

    all_images = [os.path.join(source_image_dir, img) for img in os.listdir(source_image_dir) if
                  img.endswith(('jpg', 'jpeg', 'png'))]

    selected_images = random.sample(all_images, min(sample_num, len(all_images)))
    query_embeddings = [image_to_embedding(img_path, clip_model, clip_processor, device).tolist() for img_path in selected_images]
    with open(f"batch_query_result_{sample_num}.txt", "a", encoding="utf-8") as file:
        start_time = time.time()
        file.write(f"Time before query: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")

        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=k
        )

        end_time = time.time()
        file.write(f"Time after query: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        elapsed_time = end_time - start_time
        file.write(f"Time spent on query: {elapsed_time:.2f} sec\n")

        for i, img_path in enumerate(selected_images):
            file.write(f"\n====================================\nquery image: {img_path}\n")
            for idx, result in enumerate(results['metadatas'][i]):
                file.write(f"Similar image {idx + 1}: Path is {result['path']}\n")


if __name__ == '__main__':
    fire.Fire(query_similar_images)