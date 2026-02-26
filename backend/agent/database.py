"""MongoDB operations and deduplication node for the agent."""
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient

from .state import AgentState

logger = logging.getLogger(__name__)

_client = None
_db = None


def get_db():
    global _client, _db
    if _db is None:
        _client = AsyncIOMotorClient(os.environ["MONGO_URL"])
        _db = _client[os.environ["DB_NAME"]]
    return _db


async def deduplicate_articles(state: AgentState) -> Dict:
    """Remove articles already seen in the last 7 days."""
    articles = state.get("raw_articles", [])
    if not articles:
        return {"deduplicated": [], "stats": {**state.get("stats", {}), "after_dedup": 0}}

    try:
        db = get_db()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        urls = [a["url"] for a in articles if a.get("url")]
        seen_docs = await db.articles.find(
            {"url": {"$in": urls}, "fetched_at": {"$gte": cutoff}},
            {"url": 1}
        ).to_list(None)
        seen_urls = {doc["url"] for doc in seen_docs}
        new_articles = [a for a in articles if a.get("url") and a["url"] not in seen_urls]
    except Exception as e:
        logger.warning(f"Dedup DB check failed (proceeding without): {e}")
        new_articles = articles

    # Cap at 50 for LLM validation cost control
    new_articles = new_articles[:50]
    return {
        "deduplicated": new_articles,
        "stats": {**state.get("stats", {}), "after_dedup": len(new_articles)},
    }


async def save_results(state: AgentState) -> Dict:
    """Persist articles and digest to MongoDB."""
    try:
        db = get_db()
        now = datetime.now(timezone.utc).isoformat()
        run_id = state.get("run_id", "unknown")

        # Upsert validated articles
        for article in state.get("validated", []):
            doc = {**article, "fetched_at": now, "sent_to_user": False}
            await db.articles.update_one(
                {"url": article.get("url", "")},
                {"$set": doc},
                upsert=True
            )

        # Save pending digest
        if state.get("digest") and state.get("actionable"):
            await db.digests.insert_one({
                "run_id": run_id,
                "digest_text": state["digest"],
                "articles": state["actionable"],
                "stats": state.get("stats", {}),
                "status": "pending",
                "created_at": now,
            })

        # Save run record
        await db.agent_runs.update_one(
            {"run_id": run_id},
            {"$set": {
                "run_id": run_id,
                "stats": state.get("stats", {}),
                "status": "completed",
                "errors": state.get("errors", []),
                "updated_at": now,
            }},
            upsert=True
        )
        logger.info(f"Run {run_id} saved to DB")
    except Exception as e:
        logger.error(f"save_results DB error: {e}")

    return {}


async def get_pending_digest(db) -> Optional[Dict]:
    return await db.digests.find_one({"status": "pending"}, sort=[("created_at", -1)])
