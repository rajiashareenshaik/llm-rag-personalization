"""
retriever.py
Combines user event history with vector similarity search
to surface top candidate items for the LLM re-ranker.
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List

from config import TOP_K_CANDIDATES
from .vectorstore import get_vectorstore

DATA_DIR = Path("data")
EVENTS_PATH = DATA_DIR / "events.jsonl"


def _load_events() -> List[dict]:
    events = []
    if EVENTS_PATH.exists():
        with open(EVENTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    return events


# Cache events in memory at module load
_EVENTS: List[dict] = _load_events()
_VS = get_vectorstore(rebuild=False)


def get_user_profile(user_id: str) -> Dict[str, Any]:
    """Build a lightweight user profile from historical events."""
    user_events = [e for e in _EVENTS if e["user_id"] == user_id]

    # Weight-based preference aggregation
    item_scores: Dict[str, float] = {}
    for evt in user_events:
        iid = evt["item_id"]
        item_scores[iid] = item_scores.get(iid, 0.0) + evt.get("weight", 1.0)

    liked_items = sorted(item_scores, key=item_scores.get, reverse=True)[:10]

    return {
        "user_id": user_id,
        "event_count": len(user_events),
        "liked_items": liked_items,
        "item_scores": item_scores,
        "recent_events": user_events[-20:],
    }


def candidate_docs(
    user_id: str,
    context_query: str,
    k: int = TOP_K_CANDIDATES,
) -> List[Dict[str, Any]]:
    """Retrieve candidate items using RAG (user profile + vector search)."""
    profile = get_user_profile(user_id)

    # Build a rich semantic query
    query_text = (
        f"Context: {context_query}\n"
        f"User liked items: {profile['liked_items']}\n"
        f"Retrieve items most likely to drive engagement and click-through."
    )

    docs_with_scores = _VS.similarity_search_with_score(query_text, k=k)

    results = []
    for doc, score in docs_with_scores:
        results.append(
            {
                "item_id": doc.metadata.get("item_id"),
                "title": doc.metadata.get("title"),
                "category": doc.metadata.get("category"),
                "similarity_score": round(float(score), 4),
                "user_affinity": profile["item_scores"].get(
                    doc.metadata.get("item_id", ""), 0.0
                ),
                "snippet": doc.page_content[:500],
            }
        )

    # Boost items with known user affinity
    results.sort(
        key=lambda x: x["similarity_score"] + 0.5 * x["user_affinity"],
        reverse=True,
    )
    return results
