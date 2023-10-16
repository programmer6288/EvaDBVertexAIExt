import json
import time

import numpy as np
import pandas as pd
import scann
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel

PROJECT_ID = 'vertexaisemanticsearch'
REGION = 'us-central1'
DATASET_URI = "gs://cloud-samples-data/vertex-ai/dataset-management/datasets/bert_finetuning/wide_and_deep_trainer_container_tests_input.jsonl"
vertexai.init(project=PROJECT_ID, location=REGION)
model = TextEmbeddingModel.from_pretrained("google/textembedding-gecko@001")

records = []
with open("wide_and_deep_trainer_container_tests_input.jsonl") as f:
    for line in f:
        record = json.loads(line)
        records.append(record)

df = pd.DataFrame(records)
df.head(10)
        
def get_embedding(text: str) -> list:
    try:
        embeddings = model.get_embeddings([text])
        return embeddings[0].values
    except:
        return []


get_embedding.counter = 0

# This may take several minutes to complete.
df["embedding"] = df["textContent"].apply(lambda x: get_embedding(x))

record_count = len(records)
dataset = np.array([df.embedding[i] for i in range(record_count)])


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
    start = time.time()
    query = model.get_embeddings([query])[0].values
    neighbors, distances = searcher.search(query, final_num_neighbors=3)
    end = time.time()

    for id, dist in zip(neighbors, distances):
        print(f"[docid:{id}] [{dist}] -- {df.textContent[int(id)][:125]}...")
    print("Latency (ms):", 1000 * (end - start))
    

search("tell me about an animal")