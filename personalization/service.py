"""
service.py
FastAPI router exposing the personalization endpoints.
"""
import json
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from .schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
)
from .recommender_chain import build_recommender_chain

router = APIRouter()
_chain = build_recommender_chain()


@router.post("/recommendations", response_model=RecommendationResponse)
async def recommend(req: RecommendationRequest):
    """
    Generate personalized recommendations for a user.

    - **user_id**: Unique identifier for the user
    - **context**: Placement (homepage, pdp, cart, email, etc.)
    - **k**: Number of items to return (default from env)
    - **filters**: Optional metadata filters (e.g. category)
    """
    start = time.perf_counter()
    try:
        raw: str = await _chain.ainvoke(req.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chain error: {exc}")

    # Strip markdown code fences if present
    raw = raw.strip().strip("```json").strip("```").strip()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="LLM returned invalid JSON. Check prompt or model output.",
        )

    items = [
        RecommendedItem(
            item_id=i["item_id"],
            score=min(max(float(i.get("score", 0.5)), 0.0), 1.0),
            reason=i.get("reason"),
        )
        for i in payload.get("items", [])
    ]

    latency_ms = round((time.perf_counter() - start) * 1000)

    return RecommendationResponse(
        user_id=req.user_id,
        context=req.context or "homepage",
        items=items,
        explanation=payload.get("explanation", ""),
        metadata={"latency_ms": latency_ms, "model": _chain.__class__.__name__},
    )


@router.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def recommend_get(
    user_id: str,
    context: Optional[str] = Query("homepage"),
    k: Optional[int] = Query(None),
):
    """GET convenience wrapper around the POST endpoint."""
    req = RecommendationRequest(user_id=user_id, context=context, k=k)
    return await recommend(req)
