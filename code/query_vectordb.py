import argparse
import json
import sys
import torch
from VectorDB import VectorDB
from utils import load_clip_model_and_processor, query_by_sort, batch_image_to_embedding
from time_log import TimeLogger
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Query the vector database using the specified method.')
    parser.add_argument('--method', choices=[
        'query_vector_db',
        'cluster_and_query_vector_db',
        'query_with_fixed_total_queries',
        'query_vector_db_by_caption',
        'query_each_image_separately',
        'query_by_sort'
    ], required=True, help='Method to query the vector database')
    parser.add_argument('--vector_db_path', help='Path to load the vector database')
    parser.add_argument('--vector_db_name', help='Name of the vector database')
    parser.add_argument('--clip_path', help='Path to the clip model and processor')
    parser.add_argument('--dataset_path', help='Path to the image dataset')
    parser.add_argument('--load_embedding_from_file', action='store_true',
                        help='Whether to load embeddings from file or compute embeddings directly')
    parser.add_argument('--embedding_file_path', help='Path to the embedding file')
    parser.add_argument('--caption_path', help='Path to the caption JSON file (for caption-based queries)')
    parser.add_argument('--query_num', type=int,
                        help='Total number of images queried from the vector database (used in certain methods)')
    parser.add_argument('--cluster_num', type=int, default=3,
                        help='Number of clusters (used in clustering query method)')
    parser.add_argument('--random_state', type=int, default=93,
                        help='Random state for clustering (used in clustering query method)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing images')
    parser.add_argument('-k', type=int, default=1,
                        help='The number of images queried from the vector database for each image')
    parser.add_argument('-m', type=int, default=1, help='Parameter m for query_by_sort method')
    parser.add_argument('--device', default=None, help='Computation device (e.g., "cpu" or "cuda")')
    parser.add_argument('--result_file_name', help='json file name for saving results')
    parser.add_argument('--log_file_name', help='file name of time log')
    args = parser.parse_args()
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not args.log_file_name:
        args.log_file_name = "time_log" + args.method
    args.log_file_name = "logs/" + args.log_file_name
    time_logger = TimeLogger(log_filename=args.log_file_name, with_current_time=True)

    if not os.path.exists('results'):
        os.makedirs('results')
    if not args.result_file_name:
        args.result_file_name = "result_" + args.method + ".json"
    args.result_file_name = "results/" + args.result_file_name
    results_serializable = []

    if args.load_embedding_from_file:
        data = np.load(args.embedding_file_path)
        embeddings = data['embeddings'].astype(np.float32)
        query_embedding = [embedding.tolist() for embedding in embeddings]
        valid_image_paths = data['image_paths']
        time_logger.record_moment("Embedding file loaded")
    else:
        clip_model, clip_processor = load_clip_model_and_processor(
            args.device or torch.device("cuda" if torch.cuda.is_available() else "cpu", ), args.clip_path)
        time_logger.record_moment("Model loaded")
        image_paths = []
        for root, _, files in os.walk(args.dataset_path):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, filename))
        time_logger.start_period("Embedding")
        query_embedding, valid_image_paths = batch_image_to_embedding(image_paths, clip_model, clip_processor,
                                                                      args.device, args.batch_size)
        time_logger.end_period("Embedding")
    if args.method == 'query_by_sort':
        if not args.caption_path or not args.dataset_path:
            print('Error: --caption_path and --dataset_path are required for the query_by_sort method')
            sys.exit(1)
        time_logger.start_period("Query")
        results = query_by_sort(
            clip_model=clip_model,
            clip_processor=clip_processor,
            caption_path=args.caption_path,
            dataset_path=args.dataset_path,
            m=args.m,
            k=args.k,
            batch_size=args.batch_size,
            device=args.device
        )
        time_logger.end_period("Query")
        for caption, result in results:
            results_serializable.append({
                'query_caption': caption,
                'result': result
            })

    else:
        if not args.vector_db_path or not args.vector_db_name:
            print('Error: --vector_db_path and --vector_db_name are required for methods that use the vector database')
            sys.exit(1)
        vector_db = VectorDB(device=args.device)
        vector_db.load_vector_db(vector_db_load_path=args.vector_db_path, vector_db_name=args.vector_db_name)
        time_logger.record_moment("Vector database loaded")
        if args.method == 'query_vector_db':
            time_logger.start_period("Query")
            results, valid_image_paths = vector_db.query_vector_db(
                query_embedding=query_embedding, valid_image_paths=valid_image_paths, k=args.k
            )
            time_logger.end_period("Query")

            for i, image_path in enumerate(valid_image_paths):
                result_list = [result['path'] for result in results['metadatas'][i]]
                results_serializable.append({
                    'query_image': image_path,
                    'result': result_list
                })

        elif args.method == 'cluster_and_query_vector_db':
            time_logger.start_period("Query")
            results, valid_image_paths = vector_db.cluster_and_query_vector_db(
                query_embedding=query_embedding,
                valid_image_paths=valid_image_paths,
                query_num=args.query_num,
                cluster_num=args.cluster_num,
                random_state=args.random_state,
            )
            time_logger.end_period("Query")
            for cid, result in enumerate(results):
                results_serializable.append({
                    'cluster_id': cid,
                    'result': result})
        elif args.method == 'query_with_fixed_total_queries':
            time_logger.start_period("Query")
            results, valid_image_paths = vector_db.query_with_fixed_total_queries(
                query_embedding=query_embedding,
                valid_image_paths=valid_image_paths,
                query_num=args.query_num
            )
            time_logger.end_period("Query")
            for result in results:
                results_serializable.append({'result': result})

        elif args.method == 'query_vector_db_by_caption':
            if not args.caption_path:
                print('Error: --caption_path is required for the query_vector_db_by_caption method')
                sys.exit(1)
            time_logger.start_period("Query")
            results = vector_db.query_vector_db_by_caption(
                clip_model=clip_model,
                clip_processor=clip_processor,
                caption_path=args.caption_path,
                k=args.k
            )
            time_logger.end_period("Query")
            for result in results:
                for metadata in result.get('metadatas'):
                    for item in metadata:
                        results_serializable.append({'result': item.get('path')})
        elif args.method == 'query_each_image_separately':
            time_logger.start_period("Query")
            results = vector_db.query_each_image_separately(
                query_embedding=query_embedding,
                valid_image_paths=valid_image_paths,
                k=args.k
            )
            time_logger.end_period("Query")
            for result in results:
                for metadata in result.get('metadatas'):
                    for item in metadata:
                        results_serializable.append({'result': item.get('path')})
        else:
            print(f"Unknown method: {args.method}")
            sys.exit(1)

    with open(args.result_file_name, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=4)
    time_logger.record_moment("Results saved")


if __name__ == '__main__':
    main()
