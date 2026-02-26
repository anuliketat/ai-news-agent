"""Standalone script to run the agent directly (for local testing)."""
import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure backend root is on path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    from agent.main import run_agent
    logger.info("Starting standalone agent run...")
    result = await run_agent()
    stats = result.get("stats", {})
    logger.info(
        f"Done. "
        f"Fetched={stats.get('total_fetched', 0)}, "
        f"NewArticles={stats.get('after_dedup', 0)}, "
        f"Actionable={stats.get('actionable', 0)}, "
        f"Status={result.get('approval_status')}"
    )


if __name__ == "__main__":
    asyncio.run(main())
