from openai import OpenAI
import requests
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import asyncio
from groq import AsyncGroq

print("📁 Current Working Directory:", os.getcwd())

# Load .env file with your API key
load_dotenv()
app = FastAPI(root_path="/api/v1")
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
# List all files ending with .pdf in current directory
def get_current_pdfs():
    return [f for f in os.listdir(os.getcwd()) if f.lower().endswith(".pdf")]

print("📄 PDF files found:")
for file in get_current_pdfs():
    print("-", file)

def download_file_from_url(url, save_dir="."):
    filename = url.split("?")[0].split("/")[-1]
    if filename in get_current_pdfs():
        return "none"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download file")
    local_path = os.path.join(save_dir, filename)
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

client = OpenAI(api_key=os.getenv("chatgpt"))


# ✅ Set your vector store ID here
vector_store_id = "vs_688e116ab5b08191a3c1561d9c3d78ec"

# ✅ Async refine function for a single question
async def refine(question: str) -> str:
    results = client.vector_stores.search(
        vector_store_id=vector_store_id,
        query=question,
        max_num_results=4,
        rewrite_query=False,
        ranking_options={
            "ranker": "none"
        }
    )

    def format_results(results):
        formatted_results = ''
        for result in results:
            formatted_result = f"<result file_id='{result.file_id}'>"
            for part in result.content:
                formatted_result += f"<content>{part.text}</content>"
            formatted_results += formatted_result + "</result>"
        return f"<sources>{formatted_results}</sources>"

    context = format_results(results)

    try:
        
            prompt = f"""
You are a trusted assistant for a company, capable of answering questions, explaining content, generating code, and interpreting documents from any domain — legal, technical, policy-based, or scientific.

Instructions:
- Use the provided document context if it's relevant.
- If context is missing or unrelated, answer using your general knowledge while respecting the user's intent.
- Always keep answers short, clear, and in natural language — like a helpful human, not a robot.
- If the question asks for illegal, unethical, or private content (e.g., customer data, backend code, internal systems, employee info), respond with a clear refusal, without discussing the document at all.

Response rules:
- ❌ Do not say "the document doesn't mention..."
- ❌ Do not speculate on missing internal data
- ✅ Say “I can’t share that” or similar when privacy or security is at risk
- ✅ Avoid disclaimers
- ✅ Do not repeat the question
- ✅ Only give long answers when explicitly asked

Be ethical, private, and useful at all times.

Question:
"{question}"

Context:
\"\"\"{context}\"\"\"

Answer:
"""


            response = await groq_client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=[
                    {"role": "user", "content": prompt}
                ],
                   temperature=0.6,
                    max_completion_tokens=1024,
                    top_p=0.8,
                    
            )

            return response.choices[0].message.content.strip()

    except Exception as e:
            print(f"[Groq ERROR] {e}")
            return context.strip()

UPLOAD_DIR = "."

@app.post("/hackrx/run")
async def hackrx_run(
    payload: RunRequest,
    authorization: Optional[str] = Header(None)
):
    if not authorization or authorization != "Bearer 61b907454794627e4bc5798168e95ec604fd4af04338837ec0e6b3c81bdf06ce":
        raise HTTPException(status_code=401, detail="Unauthorized")

    doc_url = payload.documents
    questions = payload.questions

    if not doc_url or not questions:
        raise HTTPException(status_code=400, detail="Missing document URL or questions")

    try:

        # Refine answers for each question
        tasks = [refine(q) for q in questions]
        results = await asyncio.gather(*tasks)

        # Optional: Write results to file
        def write_results():
            with open("query_results.txt", "w", encoding="utf-8") as f:
                for i, item in enumerate(results):
                    f.write(f"Question: {questions[i]}\nAnswer: {item}\n\n")

            with open("chunked_contexts.txt", "w", encoding="utf-8") as f:
                for i in range(len(questions)):
                    f.write("🔹 Question:\n")
                    f.write(questions[i] + "\n\n")
                    f.write("🧠 Refined Answer:\n")
                    f.write(results[i] + "\n")
                    f.write("=" * 80 + "\n\n")
        write_results()

        return JSONResponse(content={"answers": results})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
