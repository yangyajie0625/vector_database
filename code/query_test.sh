#!/bin/bash

CLIP_PATH='/clip-vit-large-patch14-336'
VECTOR_DB_PATH='/test_vector_db'
VECTOR_DB_NAME='test_vector_db'
DATASET_PATH='/test_target_dataset'
CAPTION_PATH='/caption.json'

QUERY_NUM=11
CLUSTER_NUM=3
RANDOM_STATE=93
BATCH_SIZE=32
K=1
M=1
DEVICE='cuda'

echo "Testing query_vector_db..."
python query.py \
    --method query_vector_db \
    --vector_db_path "$VECTOR_DB_PATH" \
    --vector_db_name "$VECTOR_DB_NAME" \
    --dataset_path "$DATASET_PATH" \
    --batch_size "$BATCH_SIZE" \
    -k "$K" \
    --clip_path "$CLIP_PATH" \
    --device "$DEVICE" \

echo "Testing cluster_and_query_vector_db..."
python query.py \
    --method cluster_and_query_vector_db \
    --vector_db_path "$VECTOR_DB_PATH" \
    --vector_db_name "$VECTOR_DB_NAME" \
    --dataset_path "$DATASET_PATH" \
    --query_num "$QUERY_NUM" \
    --cluster_num "$CLUSTER_NUM" \
    --random_state "$RANDOM_STATE" \
    --batch_size "$BATCH_SIZE" \
    --clip_path "$CLIP_PATH" \
    --device "$DEVICE" \

echo "Testing query_with_fixed_total_queries..."
python query.py \
    --method query_with_fixed_total_queries \
    --vector_db_path "$VECTOR_DB_PATH" \
    --vector_db_name "$VECTOR_DB_NAME" \
    --dataset_path "$DATASET_PATH" \
    --query_num "$QUERY_NUM" \
    --clip_path "$CLIP_PATH" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \

echo "Testing query_vector_db_by_caption..."
python query.py \
    --method query_vector_db_by_caption \
    --vector_db_path "$VECTOR_DB_PATH" \
    --vector_db_name "$VECTOR_DB_NAME" \
    --caption_path "$CAPTION_PATH" \
    -k "$K" \
    --clip_path "$CLIP_PATH" \
    --device "$DEVICE" \

echo "Testing query_each_image_separately..."
python query.py \
    --method query_each_image_separately \
    --vector_db_path "$VECTOR_DB_PATH" \
    --vector_db_name "$VECTOR_DB_NAME" \
    --dataset_path "$DATASET_PATH" \
    --batch_size "$BATCH_SIZE" \
    -k "$K" \
    --clip_path "$CLIP_PATH" \
    --device "$DEVICE" \

echo "Testing query_by_sort..."
python query.py \
    --method query_by_sort \
    --caption_path "$CAPTION_PATH" \
    --dataset_path "$DATASET_PATH" \
    -m "$M" \
    -k "$K" \
    --batch_size "$BATCH_SIZE" \
    --clip_path "$CLIP_PATH" \
    --device "$DEVICE" \
