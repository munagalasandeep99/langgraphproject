"""
Central configuration for the multi-agent novel Q&A system.

Everything that used to be hardcoded across index.py / chat.py / novel.py
lives here now, so the agents stay in sync and you change settings in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ancient_godly_monarch")

NOVEL_FILE = os.getenv("NOVEL_FILE", "ancient_godly_monarch.txt")
NOVEL_TITLE = "Ancient Godly Monarch"

EMBEDDING_MODEL = "models/gemini-embedding-2-preview"
EMBEDDING_DIM = 3072  # must match the embedding model's output dimension
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.3  # a little creativity for synthesis, still grounded

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
BATCH_SIZE = 100

# Set NEO4J_URI in your .env to enable the graph.
# If it's blank or unreachable, the agents fall back to vector-only retrieval.
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
USE_NEO4J = bool(NEO4J_URI)

GRAPH_EXTRACT_CHARS = 6000

# Higher k = more context = richer/more "spoiler-complete" answers.
K_CHARACTER = 10
K_PLOT = 8
K_ANALYSER = 12
K_SUMMARY = 14
