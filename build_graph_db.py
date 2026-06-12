"""
Build the Neo4j knowledge graph from the novel.

For each chapter, the LLM extracts characters and the relationships between them,
which are written into Neo4j. The Character and Analyser agents then query this
graph so relationship questions are answered from structured facts, not just a
lucky similarity search.

Usage:
    python build_graph_db.py              # process ALL chapters (slow + many LLM calls)
    python build_graph_db.py --limit 50   # process the first 50 chapters (testing)

Prereqs: NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD set in .env, and the novel
text present (run scrape_novel.py first).
"""

import argparse
import json
import re
from pathlib import Path

import config
import neo4j_store
from index import load_chapter_documents
from retrieval import get_llm

EXTRACT_PROMPT = """From this excerpt of the novel "{title}" (Chapter {chapter}),
extract the characters and the relationships between them.

Return ONLY compact JSON, no markdown, in this exact shape:
{{
  "characters": [{{"name": "...", "description": "one short phrase"}}],
  "relationships": [{{"a": "...", "b": "...", "type": "ally|rival|family|romance|master-disciple|enemy|other", "description": "one short phrase"}}]
}}

Rules:
- Use the most complete name for each character (no pronouns).
- Only include relationships actually shown in THIS excerpt.
- If none, return empty arrays.

Excerpt:
{text}"""


def extract_chapter(llm, chapter: int, text: str) -> dict:
    prompt = EXTRACT_PROMPT.format(
        title=config.NOVEL_TITLE,
        chapter=chapter,
        text=text[: config.GRAPH_EXTRACT_CHARS],
    )
    try:
        raw = llm.invoke(prompt).content
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {"characters": [], "relationships": []}
        data.setdefault("characters", [])
        data.setdefault("relationships", [])
        return data
    except Exception:
        return {"characters": [], "relationships": []}


def main(limit: int | None) -> None:
    if not neo4j_store.is_available():
        print("Neo4j is not reachable. Check NEO4J_URI / credentials in .env, then retry.")
        return

    txt_path = Path(__file__).parent / config.NOVEL_FILE
    if not txt_path.exists():
        print(f"Error: {txt_path} not found. Run scrape_novel.py first.")
        return

    chapter_docs = load_chapter_documents(txt_path)
    chapter_docs.sort(key=lambda d: d.metadata["chapter"])
    if limit:
        chapter_docs = chapter_docs[:limit]
    print(f"Building graph from {len(chapter_docs)} chapters.\n")

    neo4j_store.setup_schema()
    llm = get_llm()

    total_chars, total_rels = 0, 0
    for doc in chapter_docs:
        chapter = doc.metadata["chapter"]
        data = extract_chapter(llm, chapter, doc.page_content)

        for c in data["characters"]:
            name = (c.get("name") or "").strip()
            if name:
                neo4j_store.upsert_character(name, c.get("description"), chapter)
                total_chars += 1

        for r in data["relationships"]:
            a, b = (r.get("a") or "").strip(), (r.get("b") or "").strip()
            if a and b and a != b:
                neo4j_store.upsert_relationship(
                    a, b, (r.get("type") or "other").strip(),
                    r.get("description"), chapter,
                )
                total_rels += 1

        print(f"Chapter {chapter}: +{len(data['characters'])} characters, "
              f"+{len(data['relationships'])} relationships")

    print(f"\nDone. Upserted {total_chars} character mentions and "
          f"{total_rels} relationships into Neo4j.")
    print("Next: python graph.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N chapters (for testing).")
    args = parser.parse_args()
    main(args.limit)
