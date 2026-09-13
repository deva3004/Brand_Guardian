"""Retrieve the compliance rule chunks most relevant to a piece of transcript text."""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

from brand_guardian.exception import BrandGuardianException
from brand_guardian.logger import get_logger
from brand_guardian.services.indexer_service import COLLECTION_NAME, EMBEDDING_MODEL

logger = get_logger(__name__)

DEFAULT_K = 5


@lru_cache(maxsize=1)
def _get_store() -> PGVector:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=os.environ["DATABASE_URL"],
        use_jsonb=True,
    )


def get_relevant_rules(query_text: str, k: int = DEFAULT_K) -> list[str]:
    """Return the k compliance rule chunks most similar to query_text."""
    try:
        store = _get_store()
        results = store.similarity_search(query_text, k=k)
        logger.info(f"Retrieved {len(results)} rule chunks for query ({len(query_text)} chars)")
        return [doc.page_content for doc in results]
    except Exception as error:
        raise BrandGuardianException(error) from error


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    for chunk in get_relevant_rules("You should always disclose paid partnerships in your video description."):
        print("---")
        print(chunk)
