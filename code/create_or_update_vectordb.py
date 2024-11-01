import argparse
import sys
import torch
from VectorDB import VectorDB
from utils import load_clip_model_and_processor
from time_log import TimeLogger
import os
def main():
    parser = argparse.ArgumentParser(description='Create or update a vector database.')
    parser.add_argument('action', choices=['create', 'update'], help='Operation type: create or update the vector database')
    parser.add_argument('--vector_db_path', required=True, help='Path to save or load the vector database')
    parser.add_argument('--vector_db_name', required=True, help='Name of the vector database')
    parser.add_argument('--clip_path', help='Path to the clip model and processor')
    parser.add_argument('--dataset_path', required=True, help='Path to the image dataset')
    parser.add_argument('--distance_func', choices=['l2', 'ip', 'cosine'], help='Distance function')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing images')
    parser.add_argument('--device', default=None, help='Computation device (e.g., "cpu" or "cuda")')
    args = parser.parse_args()
    if not os.path.exists('logs'):
        os.makedirs('logs')
    time_logger = TimeLogger(log_filename="logs/time_log_" + args.action, with_current_time=True)
    vector_db = VectorDB(device=args.device)
    clip_model, clip_processor = load_clip_model_and_processor(args.device or torch.device("cuda" if torch.cuda.is_available() else "cpu"), args.clip_path)
    time_logger.record_moment("Model loaded")
    image_paths = []
    for root, _, files in os.walk(args.dataset_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, filename))
    if args.action == 'create':
        vector_db.create_vector_db(
            clip_model=clip_model,
            clip_processor=clip_processor,
            image_paths=image_paths,
            vector_db_save_path=args.vector_db_path,
            vector_db_name=args.vector_db_name,
            distance_func=args.distance_func,
            batch_size=args.batch_size
        )
        time_logger.record_moment("Vector database created")

    elif args.action == 'update':
        vector_db.load_vector_db(vector_db_load_path=args.vector_db_path, vector_db_name=args.vector_db_name)
        vector_db.update_vector_db(
            clip_model=clip_model,
            clip_processor=clip_processor,
            image_paths=image_paths,
            batch_size=args.batch_size
        )
        time_logger.record_moment("Vector database updated")
    else:
        print(f"Unknown action type: {args.action}")
        sys.exit(1)

if __name__ == '__main__':
    main()
