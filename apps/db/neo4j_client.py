from __future__ import annotations

from dataclasses import dataclass

from neo4j import GraphDatabase, Driver

@dataclass(frozen=True)
class Neo4jConfig:
    """
    Neo4j 접속 설정
    """
    uri: str
    auth: tuple[str, str]

class Neo4jClient:
    """
    Neo4j 드라이버 래퍼.
    """

    def __init__(self, config: Neo4jConfig) -> None:
        self._driver: Driver = GraphDatabase.driver(
            config.uri,
            auth=config.auth
        )

    def close(self) -> None:
        """드라이브 연결 종료"""
        self._driver.close()

    def run_query(self, cypher: str, parameters: dict[str, object] | None = None) -> list[dict[str, object]]:
        """
        Cypher 쿼리 실행 및 결과를 리스트로 반환.

        Args:
            cypher: 실행할 Cypher 쿼리
            parameters: 쿼리 파라미터

        Returns:
            리스트 형태의 결과
        """
        # 세션을 열고 쿼리를 실행
        with self._driver.session() as session:
            result = session.run(cypher, parameters or {})
            # 결과를 dict 형태로 반환
            return [record.data() for record in result]