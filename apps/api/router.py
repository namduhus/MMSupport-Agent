from fastapi import APIRouter
from pydantic import BaseModel
from apps.agent.rag_agent.service import RAGAgentService, RAGQuery

router = APIRouter()
rag_service = RAGAgentService()


class RAGRequest(BaseModel):
    user_query: str
    user_nationality: str | None = None
    user_age: int | None = None
    preferred_language: str | None = None

class RAGResponse(BaseModel):
    answer: str

@router.post("/rag", response_model=RAGResponse, tags=["RAG Agent"])
def rag_endpoint(request: RAGRequest) -> RAGResponse:
    # RAG 질의 응답 엔드포인트
    query = RAGQuery(
        user_query=request.user_query,
        user_nationality=request.user_nationality,
        user_age=request.user_age,
        preferred_language=request.preferred_language,
    )
    result = rag_service.run(query)
    return RAGResponse(answer=result.answer)

@router.get("/health" ,tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}