### Setup

First, install all the required libraries listed in requirements.txt:

```bash
git clone https://github.com/yangyajie0625/vector_database.git vector_db
cd vector_db
conda create -n vector_db python=3.10
conda activate vector_db
pip install -r requirements.txt
```



### Creating or updating a vector database

Load the source dataset, use the CLIP model to extract features, and save them in the vector database.

```bash
python create_or_update.py create 
--vector_db_path /path/to/save/vector_db \
--vector_db_name your_vector_db \
--clip_path /path/to/clip_model
--dataset_path /path/to/dataset \
--distance_func cosine \
--batch_size 32 \
--device cuda
```

Add a new source dataset to the existing vector database.
```bash
python create_or_update.py update 
--vector_db_path /path/to/load/vector_db \
--vector_db_name your_vector_db \
--clip_path /path/to/clip_model
--dataset_path /path/to/dataset \
--distance_func cosine \
--batch_size 32 \
--device cuda
```

Valid options for distance_func are "l2", "ip", or "cosine", representing the three distance functions: Squared L2, Inner product, and Cosine similarity.

### Querying a vector database

Run the script to query the vector database using target dataset, retrieves similar images.
```bash
python query_vector_db.py --method query_vector_db \
--vector_db_path /path/to/load/vector_db \
--vector_db_name your_vector_db \
--clip_path /path/to/clip_model \
--dataset_path /path/to/dataset \
--caption_path /path/to/caption/file \ 
--query_num 50 \ 
--cluster_num 3 \
--batch_size 32 \
--k 5 \ 
--m 1 \
--device cuda \
--result_file_name your_result_file_name
```

In this script, the **--method** parameter determines the specific way the vector database is queried. Below are the possible values for method and a description of each query type:

**query_vector_db**
This method queries the vector database using image embeddings. The script processes all images in the dataset, converts them into embeddings, and retrieves a set of similar images from the vector database for each image.

**cluster_and_query_vector_db**
This method clusters the image embeddings first, and then queries the vector database using the centroids of each cluster. It retrieves representative images from each cluster, ensuring diverse results.

**query_with_fixed_total_queries**
This method performs a fixed total number of queries across the entire dataset. The total number of queries is evenly distributed among the images in the dataset, with any remaining queries randomly assigned.

**query_vector_db_by_caption**
This method performs queries using captions instead of images. The script converts the captions into embeddings and retrieves the most relevant images from the vector database based on these caption embeddings.

**query_each_image_separately**
This method queries the vector database for each image in the dataset individually. Unlike batch processing, it handles each image one by one and retrieves the most similar images for each separately.

**query_by_sort**
This method sorts the captions based on cosine similarity and retrieves images accordingly. The captions are ranked by similarity, and images are retrieved based on this ranking.

For more details about the values and meaning of other parameters, refer to the Python script.