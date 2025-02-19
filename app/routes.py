import httpx
from fastapi import APIRouter, Query
from app.dataset_loader import load_all_datasets
from app.embeddings.embed_model import generate_embeddings, search_similar
from sentence_transformers import SentenceTransformer
import os

router = APIRouter()

# Load datasets
DATASETS = load_all_datasets()

# Load the model (cached for performance)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings, verses, sources = generate_embeddings(DATASETS, model)

@router.get("/ask")
async def ask_question(question: str, books: list[str] = Query(["Bhagavad Gita", "Quran", "Bible"])):
    """
    Endpoint to answer a question from the selected books.
    """
    # Search locally using FAISS
    local_results = search_similar(question, model, embeddings, verses, sources, books, DATASETS)

    # Call Gemini AI API
    gemini_results = await call_gemini_api(question)

    return {"local_results": local_results, "gemini_results": gemini_results}

async def call_gemini_api(question: str):
    """
    Calls the Gemini AI (or Bard) API to get a response for the user's question.
    """
    api_url = "https://generativelanguage.googleapis.com/v1beta2/ask"
    api_key = os.getenv("GEMINI_API_KEY")

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"question": question}

    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()  # Return the response from Gemini
        else:
            return {"error": "Failed to fetch data from Gemini API"}
