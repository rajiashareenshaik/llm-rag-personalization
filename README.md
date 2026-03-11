# LLM + RAG Personalization Pipeline

> **Problem Statement:** Increase customer engagement by ~22% and improve recommendation click-through rates (CTR) by ~17% using an LLM + RAG-based personalization system.

## Architecture Overview

```
User Request
    |
    v
FastAPI Service  (main.py)
    |
    v
Personalization Router  (personalization/service.py)
    |
    v
LangChain RAG Chain  (personalization/recommender_chain.py)
    |           |
    |           v
    |     User Profile Builder  (personalization/retriever.py)
    |           |  events.jsonl (behavioral history)
    |           |
    v           v
LLM Re-Ranker <-- Chroma Vector Store  (personalization/vectorstore.py)
(GPT-4.1-mini)          |  items.jsonl (item catalog)
    |
    v
Ranked Recommendations  (JSON with scores + reasons)
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI GPT-4.1-mini (configurable) |
| Orchestration | LangChain 0.3+ |
| Vector Store | Chroma DB |
| Embeddings | OpenAI text-embedding-ada-002 |
| API Layer | FastAPI + Uvicorn |
| Data Schemas | Pydantic v2 |

## Pipeline Steps

1. **User Profile Building** - Aggregates behavioral events (views, clicks, purchases) with configurable weights into a preference profile
2. **RAG Retrieval** - Embeds a query combining user preferences + placement context, runs semantic similarity search over item embeddings in Chroma
3. **Affinity Boosting** - Re-scores retrieved candidates by combining vector similarity + user affinity score
4. **LLM Re-Ranking** - GPT-4.1-mini receives the user profile + candidates, selects top-K items, provides per-item reasons and an overall explanation
5. **Structured Response** - Returns typed Pydantic response with scores, reasons, latency metadata

## Project Structure

```
llm-rag-personalization/
  .env.example                    # Environment variables template
  config.py                       # App configuration
  main.py                         # FastAPI app entry point
  requirements.txt                # Python dependencies
  data/
    items.jsonl                   # Item catalog (add your real data)
    events.jsonl                  # User behavioral events
  personalization/
    __init__.py
    schemas.py                    # Pydantic request/response models
    vectorstore.py                # Chroma index builder + loader
    retriever.py                  # User profile + RAG candidate retrieval
    recommender_chain.py          # Core LangChain LLM re-ranking chain
    service.py                    # FastAPI router (POST + GET endpoints)
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/rajiashareenshaik/llm-rag-personalization.git
cd llm-rag-personalization
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Build the vector index

```bash
python -m personalization.vectorstore
```

### 4. Run the API

```bash
uvicorn main:app --reload --port 8000
```

### 5. Test the endpoint

```bash
curl -X POST http://localhost:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_1", "context": "homepage", "k": 5}'
```

## API Reference

### POST /api/recommendations

Request body:
```json
{
  "user_id": "user_1",
  "context": "homepage",
  "k": 5,
  "filters": {"category": "electronics"}
}
```

Response:
```json
{
  "user_id": "user_1",
  "context": "homepage",
  "items": [
    {
      "item_id": "item_7",
      "score": 0.94,
      "reason": "Matches user interest in electronics and fitness tracking"
    }
  ],
  "explanation": "Prioritized electronics with fitness crossover based on purchase history",
  "metadata": {"latency_ms": 820, "model": "RunnableSequence"}
}
```

### GET /api/recommendations/{user_id}?context=homepage&k=5

Convenience GET wrapper.

### GET /health

Health check.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| OPENAI_API_KEY | required | Your OpenAI API key |
| MODEL_NAME | gpt-4.1-mini | LLM model for re-ranking |
| VECTOR_DB_DIR | ./chroma_db | Chroma persistence directory |
| TOP_K_CANDIDATES | 20 | Candidates to retrieve from vector store |
| TOP_K_RETURN | 5 | Final items to return to caller |

## Adding Your Data

**Items** (`data/items.jsonl`) - one JSON object per line:
```json
{"id": "sku_123", "title": "Product Name", "description": "...", "category": "electronics", "price": 99.99, "tags": ["tag1"]}
```

**Events** (`data/events.jsonl`) - user behavioral events:
```json
{"user_id": "u1", "item_id": "sku_123", "event_type": "purchase", "weight": 3.0, "timestamp": "2026-01-01T00:00:00Z"}
```

Event weights: `purchase=3.0`, `add_to_cart=2.0`, `click=1.0`, `view=0.5`

After updating data, rebuild the index:
```bash
python -m personalization.vectorstore
```

## Extending the Pipeline

- **Swap vector DB**: Replace Chroma with Pinecone, Weaviate, or pgvector
- **Swap LLM**: Change `MODEL_NAME` to any LangChain-compatible model (Azure, Anthropic, Gemini)
- **Add filters**: Pass `filters` in the request to pre-filter by category, price range, etc.
- **Real-time events**: Replace `events.jsonl` with a Kafka consumer or Kinesis stream reader
- **A/B testing**: Wrap the chain in a feature flag to compare RAG vs baseline CTR

## Results

This pipeline architecture is designed to deliver:
- **~22% increase in customer engagement** via personalized, contextually-aware recommendations
- **~17% improvement in recommendation CTR** via LLM re-ranking with compelling per-item reasons

## License

MIT
