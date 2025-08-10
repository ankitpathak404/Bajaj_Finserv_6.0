import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HF_TOKEN"),
)

result = client.feature_extraction(
    "Today is a sunny day and I will get some ice cream.",
    model="BAAI/bge-small-en-v1.5",
)
print(len(result))