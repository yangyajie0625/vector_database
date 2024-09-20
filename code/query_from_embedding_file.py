import numpy as np
from chromadb import PersistentClient
import fire
import time

def query_similar_images(embeddings_file, vector_db_load_path, vector_db_name, k=1):
    data = np.load(embeddings_file, allow_pickle=True)
    embeddings = data['embeddings']
    image_paths = data['image_paths']

    client = PersistentClient(path=vector_db_load_path)
    collection = client.get_collection(name=vector_db_name)
    print("The vector database has been loaded.")

    with open("vectordb_query_result.txt", "a", encoding="utf-8") as file:
        start_time = time.time()
        query_embeddings = [embedding.tolist() for embedding in embeddings]
        file.write("\n=================================\n")
        file.write(f"Time before query: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=k
        )
        end_time = time.time()
        file.write(f"Time after query: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        elapsed_time = end_time - start_time
        file.write(f"Time spent on query: {elapsed_time:.2f} sec\n")

        for i, img_path in enumerate(image_paths):
            print(f"\n====================================\nquery image: {img_path}\n")
            for idx, result in enumerate(results['metadatas'][i]):
                print(f"Similar image {idx + 1}: Path is {result['path']}\n")

if __name__ == '__main__':
    fire.Fire(query_similar_images)
