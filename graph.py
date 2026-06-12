"""
Builds the LangGraph and runs the conversational loop.

Graph shape:
    START -> router -> (character | plot | summary | analyser) -> END

The router writes `route` into the state; a conditional edge sends the state
to the matching specialist; each specialist writes `answer` and ends.

Run:  python graph.py
"""

from langgraph.graph import StateGraph, START, END

import config
import neo4j_store
from agents import AgentState, router_node, SPECIALISTS


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router", router_node)
    for name, node in SPECIALISTS.items():
        g.add_node(name, node)

    g.add_edge(START, "router")

    g.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {name: name for name in SPECIALISTS},
    )

    for name in SPECIALISTS:
        g.add_edge(name, END)

    return g.compile()


def main() -> None:
    app = build_graph()

    print("=" * 56)
    print(f"  {config.NOVEL_TITLE} — Multi-Agent Spoiler Assistant")
    print("  Router → Character / Plot / Summary / Analyser")
    graph_on = neo4j_store.is_available()
    print(f"  Knowledge graph (Neo4j): {'ON' if graph_on else 'off (vector-only)'}")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 56)

    history: list[tuple[str, str]] = []

    while True:
        print()
        query = input("Your question: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            result = app.invoke({"query": query, "history": history})
            route = result.get("route", "?")
            answer = result.get("answer", "(no answer)")

            print(f"\n[routed to: {route} agent]")
            print("=" * 56)
            print(answer)
            print("=" * 56)

            history.append((query, answer))
        except Exception as e:
            print(f"\nError: {e}")
            print("Is Qdrant running (docker-compose up -d) and is the "
                  "collection indexed (python index.py)?")


if __name__ == "__main__":
    main()
