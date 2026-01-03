from __future__ import annotations
import os

from dataclasses import dataclass

from apps.db.neo4j_client import Neo4jClient, Neo4jConfig
from apps.db.qdrant_client import QdrantVectorClient, QdrantConfig
from sentence_transformers import SentenceTransformer

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

def _parse_neo4j_config(auth_value: str) -> tuple[str, str]:
    """Neo4j auth 값 파싱"""
    # 값이 비어있거나, 형식이 다르면 기본값 사용
    if "/" not in auth_value:
        return ("neo4j", auth_value)
    username, password = auth_value.split("/", 1)
    return (username, password)

@dataclass(frozen=True)
class RAGQuery:
    """RAG 질의 입력 스키마."""

    user_query: str  # 사용자 질의
    user_nationality: str | None = None # 사용자 국적 Neo4j에서 문화적 맥락 반영
    user_age: int | None = None # 사용자 나이
    preferred_language: str | None = None # 선호 언어 번역 단계에서 출력 언어 선택

@dataclass(frozen=True)
class RAGResult:
    """RAG 응답 스키마."""

    answer: str # 진료 절차/비용 정보 대답


class RAGAgentService:
    """RAG 에이전트의 엔드투엔드 처리 서비스."""

    def __init__(self) -> None:
        # 환경변수 로드
        neo4j_uri: str = os.environ.get("NEO4J_URI","bolt://neo4j:7687")
        neo4j_auth: str = os.environ.get("NEO4J_AUTH", "neo4j/neo4j_pass")
        qdrant_url: str = os.environ.get("QDRANT_URL", "http://qdrant:6333")
        collection_name: str = os.environ.get("QDRANT_COLLECTION_NAME", "mm_support_collection")

        # Neo4j 클라이언트 초기화
        self._neo4j: Neo4jClient = Neo4jClient(
            Neo4jConfig(
                uri=neo4j_uri,
                auth=_parse_neo4j_config(neo4j_auth)
            )
        )

        # Qdrant 클라이언트 초기화
        self._qdrant: QdrantVectorClient = QdrantVectorClient(
            QdrantConfig(url=qdrant_url)
        )
        self._collection_name: str = collection_name

        # 임베딩 모델 초기화
        self._embedding_model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )

        # LLM 모델 초기화
        llm_host: str = os.environ.get("LLM_HOST", "host.docker.internal")
        llm_port: str = os.environ.get("LLM_PORT", "11434")
        llm_model: str = os.environ.get("LLM_MODEL", "ministral-3:8b")
        llm_temperature: float = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
        llm_base_url: str = f"http://{llm_host}:{llm_port}"

        self._llm = ChatOllama(
            model=llm_model,
            temperature = llm_temperature,
            base_url=llm_base_url,
        )

    def close(self) -> None:
        """리소스 정리"""
        self._neo4j.close()

    def run(self, query: RAGQuery) -> RAGResult:
        """
        RAG 처리 흐름을 단계별로 수행합니다.

        Args:
            query: 사용자 질의 및 메타 정보

        Returns:
            RAGResult: 최종 응답 텍스트
        """
        # 1) 다국어 입력을 표준 언어로 변환
        normalized_query: str = self._normalize_language(query)

        # 2) 입력을 벡터로 변환
        query_embedding: list[float] = self._embed_query(normalized_query)

        # 3) Qdrant에서 유사 문서 검색
        retrieved_docs: list[str] = self._search_qdrant(query_embedding)

        # 4) Neo4j에서 문화적 맥락 보정
        adjusted_context: str = self._adjust_with_graph(
            retrieved_docs, query.user_nationality
        )

        # 5) 최종 응답 구성
        result: RAGResult = self._build_response(adjusted_context,
                                                user_query= query.user_query,
                                                preferred_language=query.preferred_language,
                                                user_nationality=query.user_nationality)
        return result

    def _normalize_language(self, query: RAGQuery) -> str:
        """입력 언어를 내부 표준 언어로 변환합니다."""
        return query.user_query

    def _embed_query(self, normalized_query: str) -> list[float]:
        """질의 텍스트를 벡터로 변환합니다."""
        vector = self._embedding_model.encode(
            normalized_query, normalize_embeddings=True)
        return vector.tolist()

    def _search_qdrant(self, query_embedding: list[float]) -> list[str]:
        """Qdrant에서 유사 문서를 검색합니다."""
        results:  list[dict[str, object]] = self._qdrant.search(
            collection_name=self._collection_name,
            vector=query_embedding,
            limit=10,
        )
        texts = []
        for r in results:
            score = r.get("score", 0.0)
            text = r.get("payload", {}).get("text")
            if score >=0.75 and isinstance(text, str):
                texts.append(text)
        return texts
        
    def _adjust_with_graph(
        self, retrieved_docs: list[str], user_nationality: str | None
    ) -> str:
        """Neo4j로 문화적 맥락을 반영해 문서를 보정합니다."""
        # 국적이 없으면 문서만 반환
        if not user_nationality:
            return "\n".join(retrieved_docs)

        cypher = """
        MATCH (n:Nationality {code: $code})-[:HAS_TIP]->(t:CultureTip)
        RETURN t.text AS tip
        """
        rows = self._neo4j.run_query(cypher, {"code": user_nationality})
        tips = [row.get("tip") for row in rows if isinstance(row.get("tip"), str)]

        # Tip이 없으면 문서만 반환
        if not tips:
            return "\n".join(retrieved_docs)
        
        # 문서 + 팁 합치기
        tips_block = "국적별 안내사항: \n" + "\n".join(f"- {tip}" for tip in tips)
        return "\n".join(retrieved_docs) + "\n\n" + tips_block

    def _build_response(self, 
                        adjusted_context: str,
                        user_query: str,
                        preferred_language: str | None, 
                        user_nationality: str | None) -> RAGResult:
        """최종 응답을 구성합니다."""
        system_msg = SystemMessage(
            content="""
            너는 의료 지원 상담 전문가이며, 한국 거주 외국인을 위한 의료 상담 도우미다. 
            사용자는 한국어, 영어, 중국어, 베트남어, 태국어를 사용한다.
            너의 임무는 사용자의 질문에 대해 친절하고 이해하기 쉽게 대답한다.
            너는 제공된 문맥 정보를 바탕으로 정확하고 유용한 답변을 생성해야 한다.
            사용자 질문에 직접적으로 답변하고, 문맥에 없는 정보는 추측하지 마라.
            사용자가 선호 언어를 지정했다면 그 언어로 답변하라. 지정이 없으면 지정한 국가로 답변하라.
            국가 코드에 따른 기본 답변 언어는 다음과 같다: KR=한국어, CN=중국어, VN=베트남어, TH=태국어, US=영어.
            """
        )
        lang_hint = (f"선호 언어: {preferred_language}." if preferred_language else "선호언어: 미지정")
        country_hint = (f"국가 코드: {user_nationality}" if user_nationality else "국가 코드: 미지정")

        user_msg = HumanMessage(content=(
                                f"질문: {user_query}\n\n"
                                f"참고 문맥:{adjusted_context}\n\n"
                                f"{lang_hint}\n{country_hint}"
                            )
        )

        response = self._llm.invoke([system_msg, user_msg])
        return RAGResult(answer=response.content)
