## RAG Chat (Backend)

RAG API using FastAPI + LangChain + Chroma for QA over your documents.

### Quick start
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# env (defaults to local models; no key needed)
# LLM_PROVIDER=ollama
# MODEL_NAME=llama3:instruct
# EMBED_PROVIDER=huggingface
# EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

mkdir -p data/docs  # put .md/.txt/.pdf here
python -m app.ingestor.pipeline
uvicorn app.api.server:app --host 0.0.0.0 --port 8000
```

Docs: `http://localhost:8000/docs`

### Endpoints
- GET `/health`
- GET `/config`
- POST `/v1/chat` body: `{ "question": "...", "k": 4 }`

### Config (env)
- `LLM_PROVIDER` (default: `ollama`) | or `openai`
- `MODEL_NAME` (default: `llama3:instruct`)
- `EMBED_PROVIDER` (default: `huggingface`) | or `openai`
- `EMBED_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `PORT` (default: `8000`)
- `OPENAI_API_KEY` (only if using OpenAI)