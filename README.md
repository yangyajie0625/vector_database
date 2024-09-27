## vector database
These are the commands you can run to create, update, and query the vector database using the CLIP model. Each script is described with its function and usage example.
**create_vector_db.py**

Load the dataset from a specific path, use the CLIP model to extract features, and save them in the vector database.
```bash
python create_vector_db.py --model_path "/huggingface/clip-vit-large-patch14-336" \
--dataset_path "/huggingface/dataset" \
--vector_db_save_path "/huggingface/vector_db_cosine" \
--vector_db_name "vector_db" \
--distance_func cosine \
--batch_size 32
```
Valid options for `distance_func` are "l2", "ip", or "cosine", representing the three distance functions: Squared L2, Inner product, and Cosine similarity.


**update_vector_db.py**

Add a new dataset to the existing vector database.
```bash
python update_vector_db.py --model_path "/huggingface/clip-vit-large-patch14-336" \
--dataset_path "/huggingface/dataset" \
--vector_db_path "/huggingface/vector_db_cosine" \
--vector_db_name "vector_db" \
--batch_size 32
```

**query_vector_db.py**

Query the top-k similar images from a specific image in the vector database.
```bash
python query_vector_db.py --model_path "/huggingface/clip-vit-large-patch14-336" \

--query_image_path "/huggingface/dataset/image.jpg" \

--vector_db_load_path "/huggingface/vector_db" \

--vector_db_name "vector_db" \

--k 5
```

**individual_query_images.py**

Randomly sample `sample_num` images from `source_image_dir`, query the top-k similar images from the vector database for each image, and record the query time.
```bash
python individual_query_images.py --model_path "/huggingface/clip-vit-large-patch14-336" \

--vector_db_load_path "/huggingface/vector_db" \

--vector_db_name "vector_db" \

--source_image_dir "/huggingface/dataset" \

--sample_num 100 \

--k 5
```

**batch_query_images.py**

Randomly sample `sample_num` images from `source_image_dir`, query the top-k similar images from the vector database in batches, and record the query time.
```bash
python batch_query_images.py --model_path "/huggingface/clip-vit-large-patch14-336" \

--vector_db_load_path "/huggingface/vector_db" \

--vector_db_name "vector_db" \

--source_image_dir "/huggingface/dataset" \

--sample_num 100 \

--k 5
```

**extract_and_save.py**

Use the CLIP model to extract feature vectors from images in the specified directory, and save the embeddings and image paths as a `.npz` file.
```bash
python extract_and_save.py \

--model_path "/huggingface/clip-vit-large-patch14-336" \

--source_image_dir "/huggingface/dataset" \

--output_file embedding.npz
```

**query_from_embedding_file.py**

Query similar images from a stored embedding file and log the query time to a text file, outputting the query results.
```bash
python query_from_embedding_file.py \

--embeddings_file embedding.npz \

--vector_db_load_path "/huggingface/vector_db" \

--vector_db_name "vector_db" \

--k 10
```
