"""Cross-reference checker using Google News RSS (free, no API key)."""
import asyncio
import logging
import urllib.parse
from typing import Dict, List

import aiohttp
import feedparser

from ..state import AgentState

logger = logging.getLogger(__name__)

GNEWS_BASE = "https://news.google.com/rss/search"
MIN_SOURCES_FOR_VERIFIED = 2


async def _count_confirming_sources(
    session: aiohttp.ClientSession, query: str, original_domain: str
) -> int:
    """Return how many distinct news sources report the same story."""
    try:
        url = (
            f"{GNEWS_BASE}?q={urllib.parse.quote(query[:100])}"
            f"&hl=en-IN&gl=IN&ceid=IN:en"
        )
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            text = await resp.text(errors="replace")
        feed = feedparser.parse(text)

        domains = set()
        for entry in feed.entries[:15]:
            link = entry.get("link", "")
            if link and original_domain not in link:
                # Extract domain from link
                try:
                    parsed = urllib.parse.urlparse(link)
                    domains.add(parsed.netloc.replace("www.", ""))
                except Exception:
                    pass

        return len(domains)
    except Exception as e:
        logger.debug(f"Google News RSS error for '{query[:60]}': {e}")
        return 0


async def cross_reference_check(state: AgentState) -> Dict:
    """Upgrade unverified articles to verified if 2+ other sources confirm."""
    articles = list(state.get("validated", []))
    needs_check = [
        a for a in articles if a.get("needs_cross_reference") and
        a.get("validation_status") == "unverified"
    ]

    if not needs_check:
        return {"validated": articles}

    connector = aiohttp.TCPConnector(limit=10, ssl=False)
    headers = {"User-Agent": "Mozilla/5.0 NewsAgent/1.0"}

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        semaphore = asyncio.Semaphore(5)

        async def check_one(article: Dict) -> None:
            async with semaphore:
                count = await _count_confirming_sources(
                    session,
                    article.get("title", ""),
                    article.get("source_domain", ""),
                )
                if count >= MIN_SOURCES_FOR_VERIFIED:
                    article["validation_status"] = "verified"
                    article["cross_reference_count"] = count
                    article["credibility_score"] = min(
                        100, article.get("credibility_score", 60) + 15
                    )
                    article["reasoning"] = (
                        f"{article.get('reasoning', '')} "
                        f"[{count} other sources confirm]"
                    ).strip()
                else:
                    article["cross_reference_count"] = count

        await asyncio.gather(*[check_one(a) for a in needs_check])

    # Re-count after cross-ref
    verified = sum(1 for a in articles if a.get("validation_status") == "verified")
    logger.info(
        f"Cross-ref complete: {len(needs_check)} checked, {verified} total verified now"
    )
    return {
        "validated": articles,
        "stats": {**state.get("stats", {}), "verified_after_xref": verified},
    }
