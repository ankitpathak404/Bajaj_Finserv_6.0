from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("chatgpt"))

vector_store_id = "vs_688e116ab5b08191a3c1561d9c3d78ec"  # Replace with your actual vector store ID

# List and print all files in the vector store
response = client.vector_stores.files.list(vector_store_id=vector_store_id)

print(f"📦 Files in vector store '{vector_store_id}':")
for file_obj in response.data:
    print(f"{file_obj.id}   and   {file_obj.attributes.get("file_name")}")
