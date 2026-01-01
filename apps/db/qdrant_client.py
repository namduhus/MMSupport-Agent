from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter


@dataclass(frozen=True)
class QdrantConfig:
    """Qdrant 접속 설정."""

    url: str
    api_key: str | None = None


class QdrantVectorClient:
    """Qdrant 벡터 검색 클라이언트 래퍼."""

    def __init__(self, config: QdrantConfig) -> None:
        # Qdrant 클라이언트 생성
        self._client: QdrantClient = QdrantClient(
            url=config.url, api_key=config.api_key
        )

    def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        query_filter: Filter | None = None,
    ) -> list[dict[str, object]]:
        """
        벡터 유사도 검색을 수행합니다.

        Args:
            collection_name: 컬렉션 이름
            vector: 검색 벡터
            limit: 반환 개수
            query_filter: 검색 필터

        Returns:
            검색 결과 리스트
        """
        # Qdrant 검색 실행
        results = self._client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            query_filter=query_filter,
        )
        # 결과를 dict 형태로 변환
        return [result.dict() for result in results]
