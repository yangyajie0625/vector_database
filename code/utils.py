import torch
from PIL import Image
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from transformers import CLIPModel, CLIPProcessor
import os
import random
import json
from sklearn.metrics.pairwise import cosine_similarity
Image.MAX_IMAGE_PIXELS = None
def load_clip_model_and_processor(device, clip_path):
    model = CLIPModel.from_pretrained(clip_path).to(device)
    processor = CLIPProcessor.from_pretrained(clip_path)
    return model, processor
def batch_image_to_embedding(image_paths, clip_model, clip_processor, device, batch_size):
    """
        Processes a list of image file paths in batches, computes image embeddings using the CLIP model.
        Args:
            image_paths (list): A list of file paths to the images that need to be processed.
            clip_model (CLIPModel): A pre-trained CLIP model used to compute the image embeddings.
            clip_processor (CLIPProcessor): A processor to preprocess images before passing them to the CLIP model.
            device (torch.device): The device (CPU or GPU) on which the computations will be performed.
            batch_size (int): The number of images to process in each batch.

        Returns:
            tuple: A tuple containing:
                - embeddings (list of lists): A list of computed image embeddings, where each embedding
                  is a list of floats representing the image in vector space.
                - valid_paths (list of str): A list of file paths corresponding to the images
                  that were successfully processed and embedded.
        """

    embeddings = []  # List to store image embeddings
    valid_paths = []  # List to store valid image file paths, ensuring the lengths of embeddings and valid paths are consistent, since only successfully processed images are included.
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i:i + batch_size]
        images = []  # List to store loaded images
        current_valid_paths = []  # List to store valid paths in the current batch
        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                images.append(image)
                current_valid_paths.append(path)
            except Exception as e:
                print(f"Error processing image file: {path}, error: {e}")
                continue

        if not images:
            # Skip the batch if no valid images were found
            continue

        try:
            inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                batch_embeddings = clip_model.get_image_features(**inputs)
            embeddings.extend(batch_embeddings.cpu().numpy().tolist())
            valid_paths.extend(current_valid_paths)
        except Exception as e:
            print(f"Error processing images in batch {batch_paths}: {e}")
            continue

    return embeddings, valid_paths


def caption_to_embedding(captions, clip_model, clip_processor, device):
    caption_inputs = clip_processor(text=captions, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        caption_embeddings = clip_model.get_text_features(**caption_inputs).cpu().numpy()
    return caption_embeddings

def extract_frames_from_video(video_path, output_dir, interval=1, pbar=None):
    """
    Extracts each frame from a video file and saves it to a specified directory.

    Parameters:
    video_path (str): Path to the video file.
    output_dir (str): Directory to save the frames.
    interval (int): The interval of frames to save (default is to save every frame, interval of 1).
    pbar (tqdm): tqdm progress bar object to update the progress of the total frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video file: {video_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_count = 0
    saved_count = 0

    video_output_dir = os.path.join(output_dir, video_name)
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame_filename = os.path.join(video_output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1
        if pbar:
            pbar.update(1)

    cap.release()
    print(f"Video {video_name} extraction completed, {saved_count} frames saved.")

def extract_frames_from_directory(input_dir, output_dir, interval=1, max_workers=4):
    """
    Extracts frames from all video files in a directory and saves them to a specified output directory.

    Parameters:
    input_dir (str): Input directory containing the video files.
    output_dir (str): Output directory to save the extracted frames.
    interval (int): The interval of frames to save (default is to save every frame, interval of 1).
    max_workers (int): Maximum number of threads.
    """
    video_extensions = ['.mp4', '.mkv', '.webm']
    total_frames = 0
    video_files = []

    # Calculate the total number of frames for all video files
    for filename in os.listdir(input_dir):
        if any(filename.endswith(ext) for ext in video_extensions):
            video_path = os.path.join(input_dir, filename)
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                total_frames += frames
                video_files.append(video_path)
            cap.release()

    # Initialize total progress bar
    with tqdm(total=total_frames, desc="Extracting frames from all videos", miniters=100) as pbar:
        # Use ThreadPoolExecutor for multithreading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(extract_frames_from_video, video_path, output_dir, interval, pbar) for video_path in video_files]
            for future in futures:
                future.result()  # Wait for all threads to complete


def perform_kmeans_clustering(embeddings, cluster_num, random_state):
    kmeans = KMeans(n_clusters=cluster_num, random_state=random_state)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.labels_

    clustered_embeddings = {i: [] for i in range(cluster_num)}
    for label, embedding in zip(cluster_labels, embeddings):
        clustered_embeddings[label].append(embedding)
    return clustered_embeddings

def get_top_k_images_for_caption(image_embeddings, image_paths, caption_embedding, k):
    image_scores = []

    caption_embedding = caption_embedding.reshape(1, -1)
    scores = cosine_similarity(image_embeddings, caption_embedding).flatten()

    for image_path, score in zip(image_paths, scores):
        image_scores.append((image_path, score))

    image_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_images = [image_score[0] for image_score in image_scores[:k]]
    return top_k_images


def query_by_sort(clip_model, clip_processor, caption_path, dataset_path, device, m=1, k=1, batch_size=32):
    with open(caption_path, 'r', encoding='utf-8') as f:
        captions_data = json.load(f)

    captions = []
    for video_id, caption_list in captions_data.items():
        captions.extend(caption_list)
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))

    if m <= 1:
        image_paths = random.sample(image_paths, m * len(image_paths))
    else:
        image_paths = random.sample(image_paths, min(m, len(image_paths)))
    image_embeddings, _ = batch_image_to_embedding(image_paths, clip_model, clip_processor, device, batch_size)
    caption_embeddings = caption_to_embedding(captions, clip_model, clip_processor, device)

    results = []
    for caption, caption_embedding in zip(captions, caption_embeddings):
        top_k_images = get_top_k_images_for_caption(image_embeddings, image_paths, caption_embedding, k)
        results.append((caption, top_k_images))
    return results