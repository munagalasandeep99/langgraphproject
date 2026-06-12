"""
Neo4j knowledge-graph layer.

Holds the connection + all Cypher in one place. Everything degrades gracefully:
if NEO4J_URI is unset or the server is unreachable, is_available() returns False
and the agents simply skip graph enrichment (vector-only behaviour).

Graph schema
------------
(:Character {name, description})
(:Chapter {number})
(c:Character)-[:APPEARS_IN]->(ch:Chapter)
(a:Character)-[:RELATES_TO {type, description, chapter}]->(b:Character)
"""

from functools import lru_cache

import config

try:
    from neo4j import GraphDatabase
except ImportError:  # neo4j package not installed
    GraphDatabase = None


@lru_cache(maxsize=1)
def get_driver():
    """Return a verified driver, or None if Neo4j isn't configured/reachable."""
    if not config.NEO4J_URI or GraphDatabase is None:
        return None
    try:
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
        )
        driver.verify_connectivity()
        return driver
    except Exception:
        return None


def is_available() -> bool:
    return get_driver() is not None


def run_cypher(query: str, params: dict | None = None) -> list[dict]:
    driver = get_driver()
    if driver is None:
        return []
    with driver.session() as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]


# --------------------------------------------------------------------------
# Schema + writes (used by build_graph_db.py)
# --------------------------------------------------------------------------
def setup_schema() -> None:
    run_cypher(
        "CREATE CONSTRAINT character_name IF NOT EXISTS "
        "FOR (c:Character) REQUIRE c.name IS UNIQUE"
    )
    run_cypher(
        "CREATE CONSTRAINT chapter_number IF NOT EXISTS "
        "FOR (ch:Chapter) REQUIRE ch.number IS UNIQUE"
    )


def upsert_character(name: str, description: str | None, chapter: int | None) -> None:
    run_cypher(
        """
        MERGE (c:Character {name: $name})
        SET c.description = coalesce($description, c.description)
        WITH c
        FOREACH (_ IN CASE WHEN $chapter IS NULL THEN [] ELSE [1] END |
            MERGE (ch:Chapter {number: $chapter})
            MERGE (c)-[:APPEARS_IN]->(ch))
        """,
        {"name": name, "description": description, "chapter": chapter},
    )


def upsert_relationship(
    a: str, b: str, rel_type: str, description: str | None, chapter: int | None
) -> None:
    run_cypher(
        """
        MERGE (x:Character {name: $a})
        MERGE (y:Character {name: $b})
        MERGE (x)-[r:RELATES_TO {type: $rel_type}]->(y)
        SET r.description = $description, r.chapter = $chapter
        """,
        {"a": a, "b": b, "rel_type": rel_type, "description": description, "chapter": chapter},
    )


# --------------------------------------------------------------------------
# Reads (used by the agents)
# --------------------------------------------------------------------------
def get_character_facts(name: str) -> dict | None:
    rows = run_cypher(
        """
        MATCH (c:Character) WHERE toLower(c.name) = toLower($name)
        OPTIONAL MATCH (c)-[:APPEARS_IN]->(ch:Chapter)
        OPTIONAL MATCH (c)-[r:RELATES_TO]-(o:Character)
        RETURN c.name AS name,
               c.description AS description,
               [x IN collect(DISTINCT ch.number) WHERE x IS NOT NULL] AS chapters,
               [rel IN collect(DISTINCT {type: r.type, other: o.name, detail: r.description})
                  WHERE rel.other IS NOT NULL] AS relationships
        """,
        {"name": name},
    )
    return rows[0] if rows else None


def get_relationship_between(a: str, b: str) -> list[dict]:
    return run_cypher(
        """
        MATCH (x:Character)-[r:RELATES_TO]-(y:Character)
        WHERE toLower(x.name) = toLower($a) AND toLower(y.name) = toLower($b)
        RETURN x.name AS from_name, y.name AS to_name,
               r.type AS type, r.description AS detail, r.chapter AS chapter
        """,
        {"a": a, "b": b},
    )


def format_graph_facts(names: list[str]) -> str:
    """Build a readable 'Knowledge graph facts' block for the given characters.

    Returns '' if Neo4j is unavailable or nothing is found, so callers can just
    concatenate it onto their vector context.
    """
    if not is_available() or not names:
        return ""

    lines: list[str] = []
    for name in names:
        facts = get_character_facts(name)
        if not facts:
            continue
        chapters = sorted(facts.get("chapters") or [])
        ch_str = f" (appears in chapters {chapters[0]}–{chapters[-1]})" if chapters else ""
        lines.append(f"- {facts['name']}{ch_str}: {facts.get('description') or 'n/a'}")
        for rel in facts.get("relationships") or []:
            lines.append(
                f"    • {rel['type']} with {rel['other']}: {rel.get('detail') or ''}".rstrip()
            )

    # If exactly two characters were asked about, surface the direct edge too.
    if len(names) == 2:
        for edge in get_relationship_between(names[0], names[1]):
            ch = f" [ch. {edge['chapter']}]" if edge.get("chapter") else ""
            lines.append(
                f"- Direct link: {edge['from_name']} —{edge['type']}→ "
                f"{edge['to_name']}{ch}: {edge.get('detail') or ''}".rstrip()
            )

    if not lines:
        return ""
    return "Knowledge graph facts:\n" + "\n".join(lines)
