"""
Retrieval helpers shared by every agent.

Keeps the Qdrant / embedding plumbing in one place so the agent nodes
in agents.py stay readable and focused on prompting + reasoning.
"""

from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models

import config


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    """Connect once to the existing Qdrant collection (cached)."""
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=config.QDRANT_URL,
        collection_name=config.COLLECTION_NAME,
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """Single shared Gemini client (cached)."""
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
    )


def _chapter_filter(chapter: int) -> models.Filter:
    """Qdrant filter for an exact chapter. langchain_qdrant nests metadata
    under the 'metadata' payload key, hence 'metadata.chapter'."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.chapter",
                match=models.MatchValue(value=chapter),
            )
        ]
    )


def _chapter_range_filter(start: int, end: int) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.chapter",
                range=models.Range(gte=start, lte=end),
            )
        ]
    )


def retrieve(query: str, k: int, chapter: int | None = None):
    """Similarity search, optionally pinned to one chapter."""
    store = get_vector_store()
    flt = _chapter_filter(chapter) if chapter is not None else None
    return store.similarity_search(query=query, k=k, filter=flt)


def retrieve_range(query: str, k: int, start: int, end: int):
    """Similarity search across a chapter range (for arc summaries)."""
    store = get_vector_store()
    return store.similarity_search(
        query=query, k=k, filter=_chapter_range_filter(start, end)
    )


def multi_query_retrieve(queries: list[str], k_each: int):
    """Run several sub-queries and de-duplicate the union of results.

    Used by the Character and Analyser agents so a relationship question like
    'how are A and B connected' pulls chunks about A, about B, AND about both.
    """
    store = get_vector_store()
    seen: set[str] = set()
    merged = []
    for q in queries:
        for doc in store.similarity_search(query=q, k=k_each):
            key = doc.page_content[:120]  # cheap dedupe on chunk prefix
            if key not in seen:
                seen.add(key)
                merged.append(doc)
    return merged


def format_context(docs) -> str:
    """Render retrieved docs into a context block that exposes chapter numbers
    to the LLM, so answers can cite 'Chapter N'."""
    parts = []
    for i, d in enumerate(docs, 1):
        chapter = d.metadata.get("chapter", "?")
        parts.append(f"[Chunk {i} — Chapter {chapter}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)
