import csv
import numpy as np
import pandas as pd

import scann
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel

import json
import time
import os

def read_csv_column(file_path, column_name):
    try:
        # Open the CSV file for reading
        with open(file_path, 'r', newline='') as csvfile:
            # Create a CSV reader
            csv_reader = csv.DictReader(csvfile)

            # Check if the specified column exists
            if column_name not in csv_reader.fieldnames:
                raise ValueError(f"Column '{column_name}' not found in the CSV file.")

            # Extract the values from the specified column into a list of strings
            column_values = [row[column_name] for row in csv_reader]

        return column_values

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
file_path = 'dup_info_titles.csv'
column_name = 'Title'  # Replace with your column name

result = read_csv_column(file_path, column_name)

# if result is not None:
#     print(f"Values in '{column_name}' column: {result}")

df = pd.DataFrame(result, columns=['Title'])

PROJECT_ID = 'vertexaisemanticsearch'
REGION = 'us-central1'
vertexai.init(project=PROJECT_ID, location=REGION)
model = TextEmbeddingModel.from_pretrained("google/textembedding-gecko@latest")

def get_embedding(text: str) -> list:
    try:
        embeddings = model.get_embeddings([text])
        return embeddings[0].values
    except:
        return []
get_embedding.counter = 0

# Custom function to convert a string representation of a list to a list of floats
def convert_to_list_of_floats(string):
    try:
        # Split the string and convert each element to a float
        return [float(item) for item in string.strip('[]').split(',')]
    except (ValueError, TypeError):
        return []

# Specify the file path of your CSV file
csv_file_path = 'embedding.csv'


if os.path.isfile(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Convert the 'ListColumn' to a list of floats using the apply method
    df['embedding'] = df['embedding'].apply(convert_to_list_of_floats)
else:
    df['embedding'] = df['Title'].apply(lambda x: get_embedding(x))
    df = df[df['embedding'].apply(lambda x: len(x) > 0)]
    df = df.reset_index(drop=True)
    # Specify the file path where you want to save the CSV file
    csv_file_path = 'embedding.csv'
    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

record_count = len(df)
temp_arr = []
for i in range(record_count):
    if len(df.embedding[i]) > 0:
        temp_arr.append(df.embedding[i])
dataset = np.array(temp_arr)

normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
# configure ScaNN as a tree - asymmetric hash hybrid with reordering
# anisotropic quantization as described in the paper; see README

# use scann.scann_ops.build() to instead create a TensorFlow-compatible searcher
searcher = (
    scann.scann_ops_pybind.builder(normalized_dataset, 10, "dot_product")
    .tree(
        num_leaves=record_count,
        num_leaves_to_search=record_count,
        training_sample_size=record_count,
    )
    .score_ah(2, anisotropic_quantization_threshold=0.2)
    .reorder(100)
    .build()
)

def search(query: str) -> None:
    org_query = query
    results = {'dist': [], 'Title': []}

    start = time.time()
    query = model.get_embeddings([query])[0].values
    neighbors, distances = searcher.search(query, final_num_neighbors=10)
    end = time.time()
    print("Listing top 10 results similar to query = \"" + org_query + "\"")
    for id, dist in zip(neighbors, distances):
        print(f"[docid:{id}] [{dist}] -- {df.Title[int(id)]}...")
        results['dist'].append(dist)
        results['Title'].append(df.Title[int(id)])
    print("Latency (ms):", 1000 * (end - start))
    
    results_df = pd.DataFrame(results)
    # Specify the file path where you want to save the CSV file
    csv_file_path = 'search_result.csv'

    # Write the DataFrame to a CSV file
    results_df.to_csv(csv_file_path, index=False)
    

search("african diaspora")
