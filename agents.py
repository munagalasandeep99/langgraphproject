"""
Agent nodes for the LangGraph.

Each node takes the shared AgentState, does its own retrieval + prompting,
and writes `answer` (and the route/debug info) back into the state.

The router classifies the query; the four specialists answer it. Every
specialist prompt is deliberately spoiler-rich, because "it's not giving
spoilers" was one of the main complaints about the old single-pass setup.
"""

import json
import re
from typing import Optional, TypedDict

import config
import neo4j_store
from retrieval import (
    get_llm,
    retrieve,
    retrieve_range,
    multi_query_retrieve,
    format_context,
)


class AgentState(TypedDict, total=False):
    query: str
    route: str                 # "character" | "plot" | "summary" | "analyser"
    chapter: Optional[int]     # extracted target chapter, if any
    chapter_range: Optional[tuple[int, int]]
    answer: str
    history: list[tuple[str, str]]


SPOILER_RULE = (
    "Give full, spoiler-rich detail. Do NOT hold back plot points, character "
    "fates, power-level reveals, or twists. The reader wants spoilers. "
    "Use ONLY the provided context — never invent events. If something isn't "
    "in the context, say so plainly, then answer what you can."
)


def _history_block(state: AgentState) -> str:
    history = state.get("history") or []
    if not history:
        return ""
    recent = history[-3:]
    lines = "\n".join(f"Q: {q}\nA: {a}" for q, a in recent)
    return f"Previous conversation:\n{lines}\n"


ROUTER_PROMPT = """You are the router for a Q&A system about the novel "{title}".
Classify the user's question into exactly one specialist and extract a chapter
number or range if the question names one.

Specialists:
- "character": about a specific person, their personality, fate, powers, or the
  RELATIONSHIP between characters.
- "plot": what HAPPENED — events of a specific chapter or a small set of chapters.
- "summary": summarise an ARC, a section, or a range of chapters at a high level.
- "analyser": cross-cutting questions needing synthesis across many chapters
  (themes, power-system logic, "why" / "how does X work over the whole story").

Return ONLY compact JSON, no markdown, in this shape:
{{"route": "...", "chapter": <int or null>, "chapter_range": [<int>, <int>] or null}}

Question: {query}"""


def _heuristic_route(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ("summar", "arc", "recap", "overview", "tl;dr")):
        return "summary"
    if any(w in q for w in ("relationship", "between", "theme", "why ", "how does")):
        return "analyser"
    if "chapter" in q or re.search(r"\bch\.?\s*\d+", q):
        return "plot"
    return "character"


def router_node(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = ROUTER_PROMPT.format(title=config.NOVEL_TITLE, query=state["query"])
    try:
        raw = llm.invoke(prompt).content
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        data = json.loads(raw)
        route = data.get("route", "").strip().lower()
        if route not in {"character", "plot", "summary", "analyser"}:
            route = _heuristic_route(state["query"])
        chapter = data.get("chapter")
        rng = data.get("chapter_range")
        chapter_range = tuple(rng) if isinstance(rng, list) and len(rng) == 2 else None
    except Exception:
        route = _heuristic_route(state["query"])
        chapter, chapter_range = None, None

    return {"route": route, "chapter": chapter, "chapter_range": chapter_range}


def _extract_names(query: str) -> list[str]:
    """Ask the LLM for the character names in the query (for the graph lookup)."""
    if not neo4j_store.is_available():
        return []
    prompt = (
        f'List the character names mentioned in this question about '
        f'"{config.NOVEL_TITLE}". Return ONLY a JSON array of strings, no markdown.\n\n'
        f"Question: {query}"
    )
    try:
        raw = get_llm().invoke(prompt).content
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        names = json.loads(raw)
        return [n.strip() for n in names if isinstance(n, str) and n.strip()][:3]
    except Exception:
        return []


def character_node(state: AgentState) -> AgentState:
    query = state["query"]
    sub_queries = [
        query,
        f"{query} relationship",
        f"{query} fate outcome death",
        f"{query} power cultivation level abilities",
    ]
    docs = multi_query_retrieve(sub_queries, k_each=max(3, config.K_CHARACTER // 3))
    context = format_context(docs)

    graph_facts = neo4j_store.format_graph_facts(_extract_names(query))
    if graph_facts:
        context = f"{graph_facts}\n\n---\n\n{context}"

    prompt = f"""You are a character expert on "{config.NOVEL_TITLE}".
{SPOILER_RULE}

When the question is about a relationship between two characters, cover BOTH
characters and how they connect (allies, rivals, family, romance, betrayal).
Mention which part of the story (early / mid / late chapters) events occur in
when the chapter tags make it inferable.

{_history_block(state)}
--- Context ---
{context}
--- End context ---

Question: {query}

Answer in detail:"""
    answer = get_llm().invoke(prompt).content
    return {"answer": answer}


def plot_node(state: AgentState) -> AgentState:
    query = state["query"]
    chapter = state.get("chapter")
    docs = retrieve(query, k=config.K_PLOT, chapter=chapter)
    if not docs and chapter is not None:
        docs = retrieve(query, k=config.K_PLOT)
    context = format_context(docs)

    scope = f"Chapter {chapter}" if chapter is not None else "the relevant chapters"
    prompt = f"""You are a plot expert on "{config.NOVEL_TITLE}".
{SPOILER_RULE}

Summarise what happens in {scope} as a clear, ordered sequence of events.
Name the characters involved and the consequences. Reference chapter numbers.

{_history_block(state)}
--- Context ---
{context}
--- End context ---

Question: {query}

Ordered account of events:"""
    answer = get_llm().invoke(prompt).content
    return {"answer": answer}


def summary_node(state: AgentState) -> AgentState:
    query = state["query"]
    rng = state.get("chapter_range")
    if rng:
        start, end = rng
        docs = retrieve_range(query, k=config.K_SUMMARY, start=start, end=end)
        scope = f"chapters {start}–{end}"
    else:
        docs = retrieve(query, k=config.K_SUMMARY)
        scope = "this arc/section"
    context = format_context(docs)

    prompt = f"""You are summarising an arc of "{config.NOVEL_TITLE}".
{SPOILER_RULE}

Give a structured summary of {scope}: the main thread, key turning points,
character developments, and how it sets up what comes next. Keep it readable —
a few short paragraphs, not a chunk dump.

{_history_block(state)}
--- Context ---
{context}
--- End context ---

Question: {query}

Arc summary:"""
    answer = get_llm().invoke(prompt).content
    return {"answer": answer}


def analyser_node(state: AgentState) -> AgentState:
    query = state["query"]
    llm = get_llm()
    decompose = f"""Break this question about "{config.NOVEL_TITLE}" into 3-4 short
search queries that together cover everything needed to answer it well.
Return ONLY a JSON array of strings, no markdown.

Question: {query}"""
    try:
        raw = llm.invoke(decompose).content
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        sub_queries = json.loads(raw)
        if not isinstance(sub_queries, list) or not sub_queries:
            sub_queries = [query]
    except Exception:
        sub_queries = [query]

    docs = multi_query_retrieve(sub_queries, k_each=max(3, config.K_ANALYSER // 4))
    context = format_context(docs)

    graph_facts = neo4j_store.format_graph_facts(_extract_names(query))
    if graph_facts:
        context = f"{graph_facts}\n\n---\n\n{context}"

    prompt = f"""You are an analytical expert on "{config.NOVEL_TITLE}".
{SPOILER_RULE}

This question needs synthesis across multiple chapters. Connect the pieces into
ONE coherent answer rather than listing chunks. Explain causes, mechanics, and
significance. Cite chapter numbers where the context provides them.

{_history_block(state)}
--- Context (gathered from several sub-queries) ---
{context}
--- End context ---

Question: {query}

Synthesised answer:"""
    answer = llm.invoke(prompt).content
    return {"answer": answer}


SPECIALISTS = {
    "character": character_node,
    "plot": plot_node,
    "summary": summary_node,
    "analyser": analyser_node,
}
