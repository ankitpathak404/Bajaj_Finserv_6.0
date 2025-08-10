from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()


# ✅ Initialize
client = OpenAI(api_key=os.getenv("chatgpt"))
vector_store_id = "vs_688e116ab5b08191a3c1561d9c3d78ec"  # replace with your own



uploaded_files = client.files.list().data

for f in uploaded_files:
    try:
        print(f"🗑️ Deleting uploaded file: {f.id} ({f.filename})")
        client.files.delete(f.id)
    except Exception as e:
        print(f"❌ Failed to delete uploaded file {f.id}: {e}")

### Step 2: List and delete files attached to the vector store

print("\n🔍 Checking vector store attachments...")
attached_files = client.vector_stores.files.list(vector_store_id=vector_store_id ).data

for f in attached_files:
    try:
        print(f"🗑️ Removing from vector store: {f.id} ({f.filename})")
        client.vector_stores.files.delete(vector_store_id=vector_store_id , file_id=f.id)
    except Exception as e:
        print(f"❌ Failed to remove file from vector store: {e}")

print("\n✅ Cleanup complete.")