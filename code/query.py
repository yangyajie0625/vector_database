import argparse
import json
import sys
import torch
from VectorDB import VectorDB
from utils import load_clip_model_and_processor, query_by_sort
from time_log import TimeLogger
import os

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
    parser.add_argument('--caption_path', help='Path to the caption JSON file (for caption-based queries)')
    parser.add_argument('--query_num', type=int, help='Total number of images queried from the vector database (used in certain methods)')
    parser.add_argument('--cluster_num', type=int, default=3, help='Number of clusters (used in clustering query method)')
    parser.add_argument('--random_state', type=int, default=93, help='Random state for clustering (used in clustering query method)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing images')
    parser.add_argument('-k', type=int, default=1, help='The number of images queried from the vector database for each image')
    parser.add_argument('-m', type=int, default=1, help='Parameter m for query_by_sort method')
    parser.add_argument('--device', default=None, help='Computation device (e.g., "cpu" or "cuda")')
    parser.add_argument('--result_file_name', help='json file name for saving results')
    args = parser.parse_args()
    if not os.path.exists('logs'):
        os.makedirs('logs')
    time_logger = TimeLogger(log_filename="logs/time_log"+args.method, with_current_time=True)
    clip_model, clip_processor = load_clip_model_and_processor(args.device or torch.device("cuda" if torch.cuda.is_available() else "cpu", ), args.clip_path)
    time_logger.record_moment("Model loaded")
    if not os.path.exists('results'):
        os.makedirs('results')
    if not args.result_file_name:
        args.result_file_name = "results/result_" + args.method + ".json"
    results_serializable = []
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
            if not args.dataset_path:
                print('Error: --dataset_path is required for the query_vector_db method')
                sys.exit(1)
            time_logger.start_period("Query")
            results, valid_image_paths = vector_db.query_vector_db(
                clip_model=clip_model,
                clip_processor=clip_processor,
                dataset_path=args.dataset_path,
                batch_size=args.batch_size,
                k=args.k
            )
            time_logger.end_period("Query")

            for i, image_path in enumerate(valid_image_paths):
                result_list = [result['path'] for result in results['metadatas'][i]]
                results_serializable.append({
                    'query_image':image_path,
                    'result':result_list
                })

        elif args.method == 'cluster_and_query_vector_db':
            if not args.dataset_path or not args.query_num:
                print('Error: --dataset_path and --query_num are required for the cluster_and_query_vector_db method')
                sys.exit(1)
            time_logger.start_period("Query")
            results, valid_image_paths = vector_db.cluster_and_query_vector_db(
                clip_model=clip_model,
                clip_processor=clip_processor,
                dataset_path=args.dataset_path,
                query_num=args.query_num,
                cluster_num=args.cluster_num,
                random_state=args.random_state,
                batch_size=args.batch_size
            )
            time_logger.end_period("Query")
            for result in results:
                for metadata in result.get('metadatas'):
                    for item in metadata:
                        results_serializable.append({'result':item.get('path')})
        elif args.method == 'query_with_fixed_total_queries':
            if not args.dataset_path or not args.query_num:
                print('Error: --dataset_path and --query_num are required for the query_vector_db_with_query_num method')
                sys.exit(1)
            time_logger.start_period("Query")
            results, valid_image_paths = vector_db.query_with_fixed_total_queries(
                clip_model=clip_model,
                clip_processor=clip_processor,
                dataset_path=args.dataset_path,
                query_num=args.query_num,
                batch_size=args.batch_size
            )
            time_logger.end_period("Query")
            for result in results:
                for metadata in result.get('metadatas'):
                    for item in metadata:
                        results_serializable.append({'result':item.get('path')})

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
                        results_serializable.append({'result':item.get('path')})
        elif args.method == 'query_each_image_separately':
            if not args.dataset_path:
                print('Error: --dataset_path is required for the query_vector_db_individual method')
                sys.exit(1)
            time_logger.start_period("Query")
            results = vector_db.query_each_image_separately(
                clip_model=clip_model,
                clip_processor=clip_processor,
                dataset_path=args.dataset_path,
                batch_size=args.batch_size,
                k=args.k
            )
            time_logger.end_period("Query")
            for result in results:
                for metadata in result.get('metadatas'):
                    for item in metadata:
                        results_serializable.append({'result':item.get('path')})
        else:
            print(f"Unknown method: {args.method}")
            sys.exit(1)

    with open(args.result_file_name, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=4)
    time_logger.record_moment("Results saved")


if __name__ == '__main__':
    main()
