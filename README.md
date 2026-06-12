# Multi-Agent Novel Q&A — LangGraph + Qdrant + Neo4j + Gemini

A router-and-specialist **multi-agent graph** (LangGraph) for asking questions
about the novel *Ancient Godly Monarch*. It combines two stores:

- **Qdrant** — semantic retrieval over chapter-tagged text chunks.
- **Neo4j** — a knowledge graph of characters and their relationships.

This hybrid is aimed at the three weak spots of the original single-pass RAG:
flat plot answers, poor character-relationship answers, and spoiler-shy replies.

## The agents

```
START -> router -> (character | plot | summary | analyser) -> END
```

- **Router** — classifies the question and extracts a chapter / range, then dispatches.
- **Character** — pulls text from several angles AND queries the Neo4j graph for
  who-relates-to-whom; relationship questions get both characters plus the direct edge.
- **Plot** — events of a chapter, with retrieval pinned to that chapter's metadata.
- **Summary** — arcs / chapter ranges via range-filtered retrieval.
- **Analyser** — decomposes the question into sub-queries, synthesises across chapters,
  and also folds in graph facts.

The Neo4j layer is OPTIONAL and degrades gracefully: if NEO4J_URI is unset or
the server is unreachable, the agents run vector-only and graph.py prints
"Knowledge graph (Neo4j): off".

## Knowledge-graph schema

```
(:Character {name, description})
(:Chapter {number})
(c:Character)-[:APPEARS_IN]->(ch:Chapter)
(a:Character)-[:RELATES_TO {type, description, chapter}]->(b:Character)
```

## File layout

```
config.py           # all settings (Gemini, Qdrant, Neo4j) in one place
scrape_novel.py     # build ancient_godly_monarch.txt from all_chapters.txt (resumable)
index.py            # chapter-aware vector indexing -> Qdrant
build_graph_db.py   # LLM entity/relationship extraction -> Neo4j
retrieval.py        # Qdrant helpers (chapter / range / multi-query)
neo4j_store.py      # Neo4j connection + Cypher + graceful fallback
agents.py           # router + 4 specialists, graph-enriched
graph.py            # LangGraph wiring + chat loop  (run this)
docker-compose.yml  # Qdrant + optional local Neo4j
requirements.txt
.env.example
test.py             # original chapter "patcher" (fills gaps in an existing file)
all_chapters.txt    # 2052 chapter URLs
```

## Setup & run

**1. Environment**
```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install playwright              # scraper only
playwright install chromium
```

**2. Configure .env**
```bash
cp .env.example .env                # Windows: copy .env.example .env
```
Fill in GOOGLE_API_KEY, and the NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD
you were given (Aura looks like neo4j+s://xxxx.databases.neo4j.io).

**3. Start the stores**
```bash
docker-compose up -d                # Qdrant (+ local Neo4j if not using a hosted URL)
```
Qdrant dashboard: http://localhost:6333/dashboard
Neo4j browser (if local): http://localhost:7474

**4. Get the novel text** (start small to confirm the scraper works)
```bash
python scrape_novel.py --limit 50
```

**5. Build both indexes**
```bash
python index.py                     # -> Qdrant (vector)
python build_graph_db.py --limit 50 # -> Neo4j (knowledge graph)
```

**6. Ask questions**
```bash
python graph.py
```

Once the 50-chapter test looks good, scrape and index the rest:
```bash
python scrape_novel.py              # full run (resumable, checkpoints every 30)
python index.py
python build_graph_db.py
```

## Try these

| Question | Agent | Uses |
|---|---|---|
| What happened in chapter 47? | Plot | Qdrant (chapter-filtered) |
| What's the relationship between Qin Wentian and Mo Qingcheng? | Character | Qdrant + Neo4j |
| Summarise the Emperor Star Academy arc | Summary | Qdrant (range) |
| How does the cultivation power system work across the story? | Analyser | Qdrant + Neo4j |

## Notes

- index.py and build_graph_db.py are independent — you can run vector-only first
  and add the graph later.
- build_graph_db.py makes one LLM call per chapter, so use --limit while testing.
- Re-run index.py after re-scraping; the chapter metadata is what the Plot/Summary
  agents depend on.
- Your .env is gitignored; never commit keys or the Neo4j password.
