from fastapi import FastAPI
from app.routes import router

app = FastAPI(
    title="Holy Chatbot API",
    description="An API that provides answers from Bhagavad Gita, Quran, and Bible.",
    version="1.0.0"
)

# Include routes
app.include_router(router)
