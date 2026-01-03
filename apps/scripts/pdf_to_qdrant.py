from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

from pathlib import Path


from pypdf import PdfReader
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client import QdrantClient

from sentence_transformers import SentenceTransformer

load_dotenv()  
@dataclass(frozen=True)
class QdrantSettings:
    url: str
    collection_name: str

def load_pdf_text(path: Path) -> str:
    """ PDF 파일의 텍스트 추출"""
    reader = PdfReader(str(path))
    texts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)

def chunk_text(text: str, chunk_size: int = 500, overlap: int= 50) -> list[str]:
    """긴 텍스트 청크로 분할"""
    chunks: list[str] = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


def main() -> None:
    qdrant_settings = QdrantSettings(
         url=os.environ.get("QDRANT_URL"),
         collection_name=os.environ.get("QDRANT_COLLECTION_NAME")
     )
    pdf_dir = Path(os.environ.get("PDF_DIR"))

    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    vector_size = model.get_sentence_embedding_dimension()

    client = QdrantClient(url=qdrant_settings.url)

    if not client.collection_exists(qdrant_settings.collection_name):
         client.create_collection(
            collection_name=qdrant_settings.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
         )
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"PDF 파일이 없습니다. {pdf_dir}")
        return
    
    points: list[PointStruct] = []
    point_id = 0

    total_pages = 0
    total_chunks = 0
    
    for pdf_path in pdf_files:
        print(f"처리중: {pdf_path.name}")
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        total_pages += page_count

        # 텍스트 추출
        texts: list[str] = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        full_text = "\n".join(texts)

        # 텍스트 청크로 분할
        chunks = chunk_text(full_text)
        total_chunks += len(chunks)

        print(f"[처리] {pdf_path.name}: {page_count} 페이지, {len(chunks)} 청크")

        # 임베딩 생성
        embeddings = model.encode(chunks, normalize_embeddings=True)

        for idx, chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embeddings[idx].tolist(),
                    payload={
                        "text": chunk,
                        "source": pdf_path.name,
                        "chunk_index": idx,
                    },
                )
            )
            point_id += 1
        
    # Qdrant 포인트 업서트
    client.upsert(
        collection_name=qdrant_settings.collection_name,
        points=points,
    )
    print(f"총 {len(pdf_files)}개 PDF 파일 처리 완료: {total_pages} 페이지, {total_chunks} 청크 업서트 완료.")

if __name__ == "__main__":
    main()