import bigframes.pandas as bpd
from bigframes.ml.llm import PaLM2TextGenerator
from google.cloud import bigquery_connection_v1 as bq_connection
from IPython.display import Markdown

import pandas as pd

PROJECT_ID = "vertexaisemanticsearch"  # @param {type:"string"}

# Please fill in these values.
LOCATION = 'us-central1'
CONNECTION = "test-connection"  # @param {type:"string"}

connection_name = f"{PROJECT_ID}.{LOCATION}.{CONNECTION}"

# Initialize client and set request parameters
client = bq_connection.ConnectionServiceClient()
new_conn_parent = f"projects/{PROJECT_ID}/locations/{LOCATION}"
exists_conn_parent = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/connections/{CONNECTION}"
)
cloud_resource_properties = bq_connection.CloudResourceProperties({})

# Try to connect using provided connection
try:
    request = client.get_connection(
        request=bq_connection.GetConnectionRequest(name=exists_conn_parent)
    )
    CONN_SERVICE_ACCOUNT = f"serviceAccount:{request.cloud_resource.service_account_id}"
# Create a new connection on error
except Exception:
    connection = bq_connection.types.Connection(
        {"friendly_name": CONNECTION, "cloud_resource": cloud_resource_properties}
    )
    request = bq_connection.CreateConnectionRequest(
        {
            "parent": new_conn_parent,
            "connection_id": CONNECTION,
            "connection": connection,
        }
    )
    response = client.create_connection(request)
    CONN_SERVICE_ACCOUNT = (
        f"serviceAccount:{response.cloud_resource.service_account_id}"
    )
print(CONN_SERVICE_ACCOUNT)
bpd.options.bigquery.project = PROJECT_ID
bpd.options.bigquery.location = LOCATION

csv_file_path = 'search_result.csv'

df = pd.read_csv(csv_file_path)
NUM_NAMES = 5
NUM_EXAMPLES = len(df)
PHRASE = 'african diaspora'
few_shot_prompt = f"""Provide {NUM_NAMES} unique and creative book titles related to the phrase at the bottom of this prompt.
First, I will provide {NUM_EXAMPLES} examples to help with your thought process, in a list delimited by the ';' character.
Then, I will provide the phrase that I want you to generate book titles based on.
"""
few_shot_prompt += 'Examples: '
for title in df['Title']:
    few_shot_prompt += title + '; '
few_shot_prompt += '\n'
few_shot_prompt += f'Phrase: {PHRASE}'
print(few_shot_prompt)
TEMPERATURE = 0.5  # @param {type: "number"}

def predict(prompt: str, temperature: float = TEMPERATURE) -> str:
    # Create dataframe
    input = bpd.DataFrame(
        {
            "prompt": [prompt],
        }
    )

    # Return response
    return model.predict(input, temperature).ml_generate_text_llm_result.iloc[0]

# Get BigFrames session
session = bpd.get_global_session()

# Define the model
model = PaLM2TextGenerator(session=session, connection_name=connection_name)

# Invoke LLM with prompt
response = predict(few_shot_prompt)

print(f'Here are the {NUM_NAMES} book titles you requested')
print(response)