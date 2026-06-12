import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import config

CHAPTER_RE = re.compile(r"\nChapter (\d+)\n={40}\n")


def load_chapter_documents(path: Path) -> list[Document]:
    """Split the novel into one Document per chapter, tagged with its number."""
    text = path.read_text(encoding="utf-8")
    parts = CHAPTER_RE.split(text)

    docs: list[Document] = []
    i = 1
    while i < len(parts) - 1:
        chapter_num = int(parts[i])
        body = parts[i + 1].strip()
        if body:
            docs.append(
                Document(
                    page_content=body,
                    metadata={"chapter": chapter_num, "source": str(path)},
                )
            )
        i += 2
    return docs


def main() -> None:
    txt_path = Path(__file__).parent / config.NOVEL_FILE
    if not txt_path.exists():
        print(f"Error: Could not find {txt_path}.")
        print("Run test.py first to scrape the novel, or place the .txt here.")
        return

    chapter_docs = load_chapter_documents(txt_path)
    if not chapter_docs:
        print("No chapters found. Is the file using the 'Chapter N\\n====' format?")
        return
    print(f"Loaded {len(chapter_docs)} chapters.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(chapter_docs)
    print(f"Total chunks: {len(chunks)} (every chunk tagged with its chapter)")

    embedding_model = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)

    try:
        client = QdrantClient(url=config.QDRANT_URL)

        if client.collection_exists(config.COLLECTION_NAME):
            client.delete_collection(config.COLLECTION_NAME)

        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIM, distance=Distance.COSINE
            ),
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=config.COLLECTION_NAME,
            embedding=embedding_model,
        )

        for i in range(0, len(chunks), config.BATCH_SIZE):
            batch = chunks[i : i + config.BATCH_SIZE]
            vector_store.add_documents(batch)
            done = min(i + config.BATCH_SIZE, len(chunks))
            print(f"Indexed {done}/{len(chunks)} chunks...")

        print(f"Done! Indexed '{config.NOVEL_TITLE}' to Qdrant with chapter metadata.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()