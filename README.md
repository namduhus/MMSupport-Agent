# MMSupport-Agent

RAG-based medical support assistant for foreign residents in Korea. The system uses Qdrant for vector search, Neo4j for cultural context tips, and FastAPI for the API layer. A Streamlit UI is provided for chat-based access.

## Stack
- Python 3.12
- FastAPI
- Qdrant
- Neo4j
- LangChain + Ollama (local)
- Streamlit

## Project Structure
- `apps/` API, agents, db clients, scripts
- `ui/` Streamlit app
- `data/` PDF sources (ingest target)

## Environment Variables
Create `.env` in the project root.

API and RAG:
- `NEO4J_URI=`
- `NEO4J_AUTH=`
- `QDRANT_URL=`
- `QDRANT_COLLECTION_NAME=`

LLM (Ollama on host):
- `LLM_HOST=host.docker.internal`
- `LLM_PORT=11434`
- `LLM_MODEL=`
- `LLM_TEMPERATURE=`

Ingestion (local script):
- `PDF_DIR=/absolute/path/to/data`

UI:
- `API_URL=http://localhost:8000`

## Run with Docker
Build and start services:
```
docker compose up --build
```

Services:
- API: `http://localhost:8000`
- UI: `http://localhost:8501`
- Neo4j Browser: `http://localhost:7474`
- Qdrant Dashboard: `http://localhost:6333/dashboard`

## Ingest PDFs to Qdrant
Place medical-related PDF documents in a folder and set `PDF_DIR` to that path.

```
python apps/scripts/pdf_to_qdrant.py
```

## API
POST `/rag`
Request JSON:
```
{
  "user_query": "감기 증상인데 어떻게 진료받아요?",
  "user_nationality": "KR",
  "user_age": 30,
  "preferred_language": "한국어"
}
```

Response JSON:
```
{
  "answer": "..."
}
```

## Notes
- Qdrant is accessed by the API container via `http://qdrant:6333`.
- Ollama runs on the host. Use `host.docker.internal` from inside containers.

## Contact
If you want to collaborate, request changes, or provide medical documents, please reach out by email.

Email: kndh2914@gmail.com

PRs are always welcome.🥳
