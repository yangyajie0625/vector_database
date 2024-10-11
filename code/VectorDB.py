import json
import os
import torch
from PIL import Image
from chromadb import PersistentClient
from utils import batch_image_to_embedding, perform_kmeans_clustering, caption_to_embedding
import random
Image.MAX_IMAGE_PIXELS = None

class VectorDB:
    def __init__(self, device=None):
        self.collection = None
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_vector_db(self, vector_db_load_path, vector_db_name):
        client = PersistentClient(path=vector_db_load_path)
        self.collection = client.get_collection(name=vector_db_name)

    def create_vector_db(self, clip_model, clip_processor, dataset_path, vector_db_save_path, vector_db_name,
                         distance_func, batch_size=32):
        image_paths = self._collect_image_paths(dataset_path)
        image_embeddings, valid_image_paths = batch_image_to_embedding(image_paths, clip_model, clip_processor,
                                                                       self.device, batch_size)
        metadata = [{"path": img_path} for img_path in valid_image_paths]
        self._save_embeddings_to_db(vector_db_save_path, vector_db_name, image_embeddings, metadata, distance_func)

    def update_vector_db(self, clip_model, clip_processor, dataset_path, batch_size=32):

        image_paths = self._collect_image_paths(dataset_path)
        image_embeddings, valid_image_paths = batch_image_to_embedding(image_paths, clip_model, clip_processor,
                                                                       self.device, batch_size)
        metadata = [{"path": img_path} for img_path in valid_image_paths]
        existing_ids = self.collection.get()['ids']
        if existing_ids:
            max_existing_id = max(map(int, existing_ids))
        else:
            max_existing_id = -1
        max_batch_size = 41660
        for i in range(0, len(image_embeddings), max_batch_size):
            batch_embeddings = image_embeddings[i:i + max_batch_size]
            batch_metadata = metadata[i:i + max_batch_size]
            batch_ids = [str(j) for j in
                         range(max_existing_id + 1 + i, max_existing_id + 1 + i + len(batch_embeddings))]

            self.collection.add(
                embeddings=batch_embeddings,
                metadatas=batch_metadata,
                ids=batch_ids
            )

    def query_vector_db(self, clip_model, clip_processor, dataset_path, batch_size=32, k=1):
        image_paths = self._collect_image_paths(dataset_path)
        query_embedding, valid_image_paths = batch_image_to_embedding(image_paths, clip_model, clip_processor,
                                                                      self.device, batch_size)
        results = self.collection.query(query_embeddings=query_embedding, n_results=k)
        return results, valid_image_paths

    def cluster_and_query_vector_db(self, clip_model, clip_processor, dataset_path, query_num, random_state, cluster_num=3, batch_size=32):
        image_paths = self._collect_image_paths(dataset_path)
        query_embedding, valid_image_paths = batch_image_to_embedding(image_paths, clip_model, clip_processor,
                                                                      self.device, batch_size)
        clustered_embeddings = perform_kmeans_clustering(query_embedding, cluster_num, random_state)
        result_list = []
        total_queries_made = 0
        for cluster_id in range(cluster_num):
            cluster_embeddings = clustered_embeddings[cluster_id]
            cluster_size = len(cluster_embeddings)
            if cluster_size == 0:
                continue
            n_results_per_data = max(1, query_num // (cluster_num * cluster_size))
            result = self.collection.query(
                query_embeddings=cluster_embeddings,
                n_results=n_results_per_data
            )
            result_list.append(result)
            total_queries_made += len(cluster_embeddings) * n_results_per_data
        remaining_queries = query_num - total_queries_made
        if remaining_queries > 0:
            remaining_indices = random.sample(range(len(image_paths)), remaining_queries)
            for idx in remaining_indices:
                remaining_result = self.collection.query(
                    query_embeddings=[query_embedding[idx]],
                    n_results=1
                )
                result_list.append(remaining_result)

        return result_list, valid_image_paths


    def query_with_fixed_total_queries(self, clip_model, clip_processor, dataset_path, query_num, batch_size=32):
        image_paths = self._collect_image_paths(dataset_path)
        total_images = len(image_paths)
        if total_images == 0:
            raise ValueError("No images found in the dataset path.")
        queries_per_image = query_num // total_images
        remaining_queries = query_num % total_images
        query_embedding, valid_image_paths = batch_image_to_embedding(image_paths, clip_model, clip_processor,
                                                                      self.device, batch_size)
        result_list = []
        if queries_per_image > 0:
            result = self.collection.query(
                query_embeddings=query_embedding,
                n_results=queries_per_image
            )
            result_list.append(result)
        if remaining_queries > 0:
            remaining_indices = random.sample(range(total_images), remaining_queries)
            for idx in remaining_indices:
                remaining_result = self.collection.query(
                    query_embeddings=[query_embedding[idx]],
                    n_results=1
                )
                result_list.append(remaining_result)

        return result_list, valid_image_paths

    def query_vector_db_by_caption(self, clip_model, clip_processor, caption_path, k=1):
        with open(caption_path, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
        captions = []
        result_list = []
        for video_id, caption_list in captions_data.items():
            captions.extend(caption_list)
            query_embeddings = caption_to_embedding(captions, clip_model, clip_processor, self.device)
            results = self.collection.query(query_embeddings=query_embeddings, n_results=k)
            result_list.append(results)
        return result_list

    def query_each_image_separately(self, clip_model, clip_processor, dataset_path, batch_size=32, k=1):
        image_paths = self._collect_image_paths(dataset_path)
        query_embedding, valid_image_paths = batch_image_to_embedding(image_paths, clip_model, clip_processor,
                                                                      self.device, batch_size)
        result_list = []
        for i, img_embedding in enumerate(query_embedding):
            results = self.collection.query(
                query_embeddings=query_embedding[i],
                n_results=k
            )
            result_list.append(results)
        return result_list

    def _collect_image_paths(self, dataset_path):
        image_paths = []
        for root, _, files in os.walk(dataset_path):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, filename))
        return image_paths

    def _save_embeddings_to_db(self, vector_db_save_path, vector_db_name, image_embeddings, metadata, distance_func):
        client = PersistentClient(path=vector_db_save_path)
        collection = client.create_collection(name=vector_db_name, metadata={"hnsw:space": distance_func})
        max_batch_size = 41660

        for i in range(0, len(image_embeddings), max_batch_size):
            batch_embeddings = image_embeddings[i:i + max_batch_size]
            batch_metadata = metadata[i:i + max_batch_size]
            batch_ids = [str(j) for j in range(i, i + len(batch_embeddings))]

            collection.add(embeddings=batch_embeddings, metadatas=batch_metadata, ids=batch_ids)
