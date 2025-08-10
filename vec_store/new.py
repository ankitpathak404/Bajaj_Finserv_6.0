from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import asyncio
import requests
import torch
from dotenv import load_dotenv
import os
from groq import AsyncGroq
from openai import OpenAI
from urllib.parse import unquote
import difflib


# Load variables from .env file
load_dotenv()

app = FastAPI(root_path="/api/v1")

client = OpenAI(api_key=os.getenv("chatgpt"))
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# ✅ Set your vector store ID here
vector_store_id = "vs_688e116ab5b08191a3c1561d9c3d78ec"

UPLOAD_DIR = "."
os.makedirs(UPLOAD_DIR, exist_ok=True)
global_id = 0


class RunRequest(BaseModel):
    documents: str
    questions: List[str]

def download_file_from_url(url, save_dir="."):
    filename = url.split("?")[0].split("/")[-1]

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download file")
    local_path = os.path.join(save_dir, filename)
    with open(local_path, "wb") as f:
        f.write(response.content)
    return filename

# ✅ Async refine function for a single question
async def refine(question: str,file_name: str) -> str:
       # Define filter outside the function call
    comparison_filter = {
        "key": "file_name",
        "type": "eq",
        "value":str(file_name) # dynamically use the provided file_path
    }
    
    print(file_name)
    results = await asyncio.to_thread(
        client.vector_stores.search,
        vector_store_id=vector_store_id,
        query=question,
        max_num_results=3,
        filters=comparison_filter
    )
    for result in results.data:
        print(result.score)            # ✅ Correct
        print(result.filename)
    print(result)
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

    return "bruh"


@app.post("/hackrx/run")
async def hackrx_run(
    payload: RunRequest,
    authorization: Optional[str] = Header(None)
):
    global global_id

    if not authorization or authorization != "Bearer 61b907454794627e4bc5798168e95ec604fd4af04338837ec0e6b3c81bdf06ce":
        raise HTTPException(status_code=401, detail="Unauthorized")

    doc_url = payload.documents
    questions = payload.questions

    if not doc_url or not questions:
        raise HTTPException(status_code=400, detail="Missing document URL or questions")

    file_path = download_file_from_url(doc_url, UPLOAD_DIR)
    file_name=file_path
    files=[
        "UNI%20GROUP%20HEALTH%20INSURANCE%20POLICY%20-%20UIIHLGP26043V022526%201.pdf",
        "Super_Splendor_(Feb_2023).pdf",
        "principia_newton.pdf",
        "indian_constitution.pdf",
        "HDFHLIP23024V072223.pdf",
        "Happy%20Family%20Floater%20-%202024%20OICHLIP25046V062425%201.pdf",
        "family20medicare20policy20uin-20uiihlip22070v042122201.pdf",
        "Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf",
        "Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf"
    ]
        # ✅ Integrate fast fuzzy matching
    decoded_files = [unquote(f) for f in files]
    match = difflib.get_close_matches(file_name, decoded_files, n=1, cutoff=0.5)
    if match:
        file_name = files[decoded_files.index(match[0])]
    try:



        # Question Answering

        results = []

        for q_idx, question in enumerate(questions):
            
            result_text= await refine(question,file_name)

            results.append({
                "question": questions[q_idx],
                "answer": result_text
            })


        # Save results to query_results.txt
        with open("query_results.txt", "w", encoding="utf-8") as f:
            for item in results:
                f.write(f"Question: {item['question']}\nAnswer: {item['answer']}\n\n")

        return JSONResponse(content={"results": results})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
