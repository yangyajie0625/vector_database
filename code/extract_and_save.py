import argparse
import os
import numpy as np
from utils import batch_image_to_embedding, load_clip_model_and_processor


def extract_features(clip_path, dataset_path, output_file, device, batch_size):
    clip_model, clip_processor = load_clip_model_and_processor(device, clip_path)

    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, filename))

    embeddings, valid_paths = batch_image_to_embedding(image_paths=image_paths,
                                                       clip_model=clip_model,
                                                       clip_processor=clip_processor,
                                                       device=device,
                                                       batch_size=batch_size)

    np.savez(output_file, embeddings=np.array(embeddings, dtype=np.float32), image_paths=valid_paths)
    print(f"Embeddings is saved at {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features from images using a CLIP model and save them as embeddings.")

    parser.add_argument('--clip_path', required=True, help='Path to the CLIP model and processor')
    parser.add_argument('--dataset_path', required=True, help='Path to the image dataset')
    parser.add_argument('--output_file', required=True, help='File to save the output embeddings')
    parser.add_argument('--device', default=None, help='Computation device (e.g., "cpu" or "cuda")')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing images')

    args = parser.parse_args()
    extract_features(clip_path=args.clip_path,
                     dataset_path=args.dataset_path,
                     output_file=args.output_file,
                     device=args.device,
                     batch_size=args.batch_size)