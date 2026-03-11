"""
vectorstore.py
Builds and manages the Chroma vector store for item embeddings.
Used by the RAG retriever to find semantically similar items.
"""
import json
from pathlib import Path
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import VECTOR_DB_DIR

DATA_DIR = Path("data")
ITEMS_PATH = DATA_DIR / "items.jsonl"


def load_items() -> List[dict]:
    """Load items from JSONL file."""
    items = []
    with open(ITEMS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_documents(items: List[dict]) -> List[Document]:
    """Convert raw item dicts into LangChain Documents."""
    docs: List[Document] = []
    for item in items:
        text = (
            f"Item ID: {item['id']}\n"
            f"Title: {item.get('title', '')}\n"
            f"Description: {item.get('description', '')}\n"
            f"Category: {item.get('category', '')}\n"
            f"Tags: {', '.join(item.get('tags', []))}\n"
            f"Price: {item.get('price', 'N/A')}"
        )
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "item_id": item["id"],
                    "title": item.get("title", ""),
                    "category": item.get("category", ""),
                    "price": item.get("price"),
                    "tags": item.get("tags", []),
                },
            )
        )
    return docs


def get_vectorstore(rebuild: bool = False) -> Chroma:
    """Return (and optionally rebuild) the Chroma vector store."""
    embeddings = OpenAIEmbeddings()

    if rebuild:
        items = load_items()
        docs = build_documents(items)
        vs = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR,
        )
        vs.persist()
        print(f"Vector store (re)built with {vs._collection.count()} documents.")
        return vs

    return Chroma(
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR,
    )


if __name__ == "__main__":
    vs = get_vectorstore(rebuild=True)
    print(f"Done. Total docs: {vs._collection.count()}")
