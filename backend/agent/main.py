"""LangGraph orchestrator for the AI News Monitoring Agent."""
import logging
import uuid

from langgraph.graph import END, START, StateGraph

from .database import deduplicate_articles, save_results
from .nodes.cross_ref import cross_reference_check
from .nodes.digest import filter_and_build_digest, send_approval_request
from .nodes.fetchers import fetch_all_sources
from .nodes.validators import validate_articles
from .state import AgentState

logger = logging.getLogger(__name__)


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("fetch_sources", fetch_all_sources)
    builder.add_node("deduplicate", deduplicate_articles)
    builder.add_node("validate_articles", validate_articles)
    builder.add_node("cross_reference", cross_reference_check)
    builder.add_node("build_digest", filter_and_build_digest)
    builder.add_node("send_approval", send_approval_request)
    builder.add_node("save_results", save_results)

    builder.add_edge(START, "fetch_sources")
    builder.add_edge("fetch_sources", "deduplicate")
    builder.add_edge("deduplicate", "validate_articles")
    builder.add_edge("validate_articles", "cross_reference")
    builder.add_edge("cross_reference", "build_digest")
    builder.add_edge("build_digest", "send_approval")
    builder.add_edge("send_approval", "save_results")
    builder.add_edge("save_results", END)

    return builder.compile()


async def run_agent(run_id: str = None) -> AgentState:
    """Run the full agent pipeline and return final state."""
    if run_id is None:
        run_id = str(uuid.uuid4())

    logger.info(f"Starting agent run: {run_id}")

    graph = build_graph()
    initial_state: AgentState = {
        "raw_articles": [],
        "deduplicated": [],
        "validated": [],
        "actionable": [],
        "digest": "",
        "approval_status": "pending",
        "run_id": run_id,
        "stats": {},
        "errors": [],
    }

    result = await graph.ainvoke(initial_state)
    logger.info(
        f"Run {run_id} complete. Stats: {result.get('stats', {})} | "
        f"Status: {result.get('approval_status')}"
    )
    return result
