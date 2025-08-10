from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("chatgpt"))

vector_store_id = "vs_688e116ab5b08191a3c1561d9c3d78ec"  # Replace with your vector store ID
UPLOAD_DIR = "."  # Local folder with PDFs

def upload_and_attach_pdfs(directory: str):
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            try:
                print(f"📄 Uploading: {filename}")
                with open(file_path, "rb") as f:
                    uploaded_file = client.files.create(
                        file=f,
                        purpose="assistants",
                    )

                print(f"🔗 Attaching: {filename}")
                # ✅ Now attach it to the vector store with metadata
                client.vector_stores.files.create(
                    vector_store_id=vector_store_id,
                    file_id=uploaded_file.id,
                    attributes={  # ✅ Optional filterable metadata
                        "file_name": filename
                    }
                )

                print(f"✅ Done: {filename}")

            except Exception as e:
                print(f"❌ Error for {filename}: {e}")

upload_and_attach_pdfs(UPLOAD_DIR)
