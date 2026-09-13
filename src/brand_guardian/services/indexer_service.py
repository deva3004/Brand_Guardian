"""Index compliance PDFs into the pgvector store on Supabase."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from brand_guardian.exception import BrandGuardianException
from brand_guardian.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "compliance_rules"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def index_documents() -> int:
    """Load, chunk, embed, and upsert all PDFs in data/ into pgvector. Returns chunk count."""
    try:
        pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
        if not pdf_paths:
            raise ValueError(f"No PDFs found in {DATA_DIR}")

        documents = []
        for pdf_path in pdf_paths:
            loaded = PyPDFLoader(str(pdf_path)).load()
            for doc in loaded:
                doc.page_content = doc.page_content.replace("\x00", "")
            documents.extend(loaded)
            logger.info(f"Loaded {len(loaded)} pages from {pdf_path.name}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} pages into {len(chunks)} chunks")

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection=os.environ["DATABASE_URL"],
            use_jsonb=True,
        )
        logger.info(f"Indexed {len(chunks)} chunks into pgvector collection '{COLLECTION_NAME}'")
        return len(chunks)
    except Exception as error:
        raise BrandGuardianException(error) from error


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    index_documents()
