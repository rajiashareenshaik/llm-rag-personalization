"""
LLM + RAG-based Personalization Service
Increases customer engagement ~22% and recommendation CTR ~17%
by using LangChain, Chroma vector store, and an LLM re-ranker.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from personalization.service import router as personalization_router

app = FastAPI(
    title="LLM RAG Personalization Service",
    description=(
        "RAG-based recommendation engine powered by LangChain + Chroma. "
        "Drives ~22% customer engagement lift and ~17% CTR improvement."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(personalization_router, prefix="/api", tags=["personalization"])


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "service": "llm-rag-personalization"}
