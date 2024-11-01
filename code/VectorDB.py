import json
import torch
from chromadb import PersistentClient
from utils import batch_image_to_embedding, perform_kmeans_clustering, caption_to_embedding


class VectorDB:
    def __init__(self, device=None):
        self.collection = None
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_vector_db(self, vector_db_load_path, vector_db_name):
        client = PersistentClient(path=vector_db_load_path)
        self.collection = client.get_collection(name=vector_db_name)

    def create_vector_db(self, clip_model, clip_processor, image_paths, vector_db_save_path, vector_db_name,
                         distance_func, batch_size=32):

        image_embeddings, valid_image_paths = batch_image_to_embedding(image_paths, clip_model, clip_processor,
                                                                       self.device, batch_size)
        metadata = [{"path": img_path} for img_path in valid_image_paths]
        self._save_embeddings_to_db(vector_db_save_path, vector_db_name, image_embeddings, metadata, distance_func)

    def update_vector_db(self, clip_model, clip_processor, image_paths, batch_size=32):

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

    def query_vector_db(self, query_embedding, valid_image_paths, k=1):

        results = self.collection.query(query_embeddings=query_embedding, n_results=k)
        return results, valid_image_paths

    def cluster_and_query_vector_db(self, query_embedding, valid_image_paths, query_num, random_state, cluster_num=3):
        clustered_embeddings = perform_kmeans_clustering(query_embedding, cluster_num, random_state)
        result_list = []
        query_num_per_cluster = query_num // cluster_num
        global_result_set = set()
        print("query_num_per_cluster", query_num_per_cluster)

        for cluster_id in range(cluster_num):
            result_set = set()
            cluster_embeddings = clustered_embeddings[cluster_id]
            cluster_size = len(cluster_embeddings)
            print("cluster_id", cluster_id)
            print("cluster_size", cluster_size)

            if cluster_size == 0:
                continue

            n_results_per_data = max(1, query_num_per_cluster // cluster_size)
            result = self.collection.query(
                query_embeddings=cluster_embeddings,
                n_results=n_results_per_data
            )
            print("n_results_per_data", n_results_per_data)

            for res in result['metadatas']:
                for metadata in res:
                    if metadata['path'] not in global_result_set:
                        result_set.add(metadata['path'])
                        global_result_set.add(metadata['path'])

            remaining_queries = query_num_per_cluster - len(result_set)
            while remaining_queries > 0:
                n_results_per_data = n_results_per_data * 2
                for embedding in query_embedding:
                    if remaining_queries <= 0:
                        break

                    result = self.collection.query(
                        query_embeddings=[embedding],
                        n_results=n_results_per_data
                    )
                    for res in result['metadatas']:
                        for metadata in res:
                            if metadata['path'] not in global_result_set:
                                result_set.add(metadata['path'])
                                global_result_set.add(metadata['path'])
                    remaining_queries = query_num_per_cluster - len(result_set)

            result_list.append(list(result_set))

        return result_list, valid_image_paths

    # def cluster_and_query_vector_db(self, query_embedding, valid_image_paths, query_num, random_state, cluster_num=3):
    #     clustered_embeddings = perform_kmeans_clustering(query_embedding, cluster_num, random_state)
    #     result_list = []
    #     query_num_per_cluster = query_num // cluster_num
    #     print("query_num_per_cluster",query_num_per_cluster)
    #     for cluster_id in range(cluster_num):
    #         result_set = set()
    #         cluster_embeddings = clustered_embeddings[cluster_id]
    #         cluster_size = len(cluster_embeddings)
    #         print("cluster_id",cluster_id)
    #         print("cluster_size",cluster_size)
    #         if cluster_size == 0:
    #             continue
    #         n_results_per_data = max(1, query_num_per_cluster // cluster_size)
    #         result = self.collection.query(
    #             query_embeddings=cluster_embeddings,
    #             n_results=n_results_per_data
    #         )
    #         print("n_results_per_data",n_results_per_data)
    #         for res in result['metadatas']:
    #             for metadata in res:
    #                 result_set.add(metadata['path'])
    #         remaining_queries = query_num_per_cluster - len(result_set)
    #         while remaining_queries > 0:
    #             n_results_per_data = n_results_per_data * 2
    #             for embedding in query_embedding:
    #                 if remaining_queries <= 0:
    #                     break
    #
    #                 result = self.collection.query(
    #                     query_embeddings=[embedding],
    #                     n_results=n_results_per_data
    #                 )
    #                 for res in result['metadatas']:
    #                     for metadata in res:
    #                         result_set.add(metadata['path'])
    #                 remaining_queries = query_num_per_cluster - len(result_set)
    #         result_list.append(list(result_set))
    #
    #
    #     return result_list, valid_image_paths

    def query_with_fixed_total_queries(self, query_embedding, valid_image_paths, query_num):

        total_images = len(query_embedding)
        queries_per_image = max(1, query_num // total_images + 1)
        result_paths_set = set()

        for embedding in query_embedding:
            result = self.collection.query(
                query_embeddings=[embedding],
                n_results=queries_per_image
            )
            for res in result['metadatas']:
                for metadata in res:
                    result_paths_set.add(metadata['path'])

        # result = self.collection.query(
        #     query_embeddings=query_embedding,
        #     n_results=queries_per_image,
        # )
        #
        # for res in result['metadatas']:
        #     for metadata in res:
        #         result_paths_set.add(metadata['path'])
        remaining_queries = query_num - len(result_paths_set)

        while remaining_queries > 0:
            queries_per_image = queries_per_image * 2
            for embedding in query_embedding:
                if remaining_queries <= 0:
                    break

                result = self.collection.query(
                    query_embeddings=[embedding],
                    n_results=queries_per_image
                )
                for res in result['metadatas']:
                    for metadata in res:
                        result_paths_set.add(metadata['path'])
                remaining_queries = query_num - len(result_paths_set)

        result_paths_list = list(result_paths_set)
        return result_paths_list, valid_image_paths

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

    def query_each_image_separately(self, query_embedding, valid_image_paths, k=1):
        result_list = []
        for i, img_embedding in enumerate(query_embedding):
            results = self.collection.query(
                query_embeddings=query_embedding[i],
                n_results=k
            )
            result_list.append(results)
        return result_list

    def _save_embeddings_to_db(self, vector_db_save_path, vector_db_name, image_embeddings, metadata, distance_func):
        client = PersistentClient(path=vector_db_save_path)
        collection = client.create_collection(name=vector_db_name, metadata={"hnsw:space": distance_func})
        max_batch_size = 41660

        for i in range(0, len(image_embeddings), max_batch_size):
            batch_embeddings = image_embeddings[i:i + max_batch_size]
            batch_metadata = metadata[i:i + max_batch_size]
            batch_ids = [str(j) for j in range(i, i + len(batch_embeddings))]

            collection.add(embeddings=batch_embeddings, metadatas=batch_metadata, ids=batch_ids)
