import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

import time

PROJECT_ID = 'vertexaisemanticsearch'
REGION = 'us-central1'

vertexai.init(project=PROJECT_ID, location=REGION)

MODEL_NAME = "textembedding-gecko@latest"

TASK_TYPE = "RETRIEVAL_DOCUMENT"  # @param ["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING"]
TITLE = "Google"  # @param {type:"string"}
TEXT = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."  # @param {type:"string"}

# Verify the input is valid.
if not MODEL_NAME:
    raise ValueError("Please set MODEL_NAME.")
if not TASK_TYPE:
    raise ValueError("Please set TASK_TYPE.")
if not TEXT:
    raise ValueError("Please set TEXT.")
if TITLE and TASK_TYPE != "RETRIEVAL_DOCUMENT":
    raise ValueError("Title can only be provided if the task_type is RETRIEVAL_DOCUMENT")
    
# A function to generate text embedding.
def text_embedding(
  model_name: str, task_type: str, text: str, title: str = "") -> list:
    """Generate text embedding with a Large Language Model."""
    model = TextEmbeddingModel.from_pretrained(model_name)

    text_embedding_input = TextEmbeddingInput(
        task_type=task_type, title=title, text=text)
    embeddings = model.get_embeddings([text_embedding_input])
    return embeddings[0].values

# Get text embedding for the downstream task.
start = time.time()
embedding = text_embedding(
    model_name=MODEL_NAME, task_type=TASK_TYPE, text=TEXT, title=TITLE)
end = time.time()
print(len(embedding))
print("Latency (ms):", 1000 * (end - start))
print(embedding)
start = time.time()
embedding = text_embedding(MODEL_NAME, "SEMANTIC_SIMILARITY", TEXT)
end = time.time()
print(len(embedding))
print("Latency (ms):", 1000 * (end - start))
print(embedding)