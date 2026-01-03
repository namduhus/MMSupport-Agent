from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv


def _load_env() -> None:
    """환경 변수를 로드합니다."""
    # .env 로딩 (없으면 무시)
    load_dotenv()


def _get_api_url() -> str:
    """API 기본 URL을 반환합니다."""
    return os.environ.get("API_URL", "http://localhost:8000")


def _build_payload(
    user_query: str,
    user_nationality: str | None,
    user_age: int | None,
    preferred_language: str | None,
) -> dict[str, Any]:
    """RAG API 요청 페이로드를 구성합니다."""
    payload: dict[str, Any] = {"user_query": user_query}
    if user_nationality:
        payload["user_nationality"] = user_nationality
    if user_age is not None:
        payload["user_age"] = user_age
    if preferred_language:
        payload["preferred_language"] = preferred_language
    return payload


def _call_rag_api(api_url: str, payload: dict[str, Any]) -> str:
    """RAG API를 호출하고 응답 텍스트를 반환합니다."""
    response = requests.post(f"{api_url}/rag", json=payload)
    response.raise_for_status()
    data = response.json()
    return str(data.get("answer", ""))


def main() -> None:
    """Streamlit 앱 엔트리포인트."""
    _load_env()
    st.set_page_config(page_title="MMSupport RAG Chat", page_icon="💬")

    st.title("MMSupport RAG Chat")
    st.caption("RAG 기반 응급처치 데모")

    with st.sidebar:
        st.header("사용자 정보")
        user_age = st.number_input("나이", min_value=0, max_value=120, value=0)
        user_nationality = st.selectbox(
            "국가(코드)",
            options=["", "KR", "CN", "VN", "TH", "US"],
            help="국가 코드가 없으면 빈 값으로 둡니다.",
        )
        preferred_language = st.selectbox(
            "선호 언어",
            options=["", "한국어", "English", "中国话", "tiếng Việt", "ภาษาไทย"],
            help="선호 언어가 없으면 빈 값으로 둡니다.",
        )
        api_url = st.text_input("API URL", value=_get_api_url())

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("질문을 입력하세요.")
    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = _build_payload(
        user_query=prompt,
        user_nationality=user_nationality or None,
        user_age=int(user_age) if user_age > 0 else None,
        preferred_language=preferred_language or None,
    )

    with st.spinner("답변 생성 중입니다..."):
        try:
            answer = _call_rag_api(api_url, payload)
        except requests.RequestException as exc:
            answer = f"API 호출 실패: {exc}"

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)


if __name__ == "__main__":
    main()
