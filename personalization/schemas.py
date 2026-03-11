from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    context: Optional[str] = Field("homepage", description="Placement context (homepage, pdp, cart, etc.)")
    k: Optional[int] = Field(None, description="Number of recommendations to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")


class RecommendedItem(BaseModel):
    item_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score between 0 and 1")
    reason: Optional[str] = Field(None, description="LLM-generated reason for recommendation")


class RecommendationResponse(BaseModel):
    user_id: str
    context: str
    items: List[RecommendedItem]
    explanation: str = Field(..., description="High-level LLM explanation for the ranking")
    metadata: Optional[Dict[str, Any]] = None
