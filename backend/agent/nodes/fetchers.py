"""Async source fetchers — RSS, HTTP APIs, with language auto-translation."""
import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import aiohttp
import feedparser
from bs4 import BeautifulSoup

from ..state import AgentState

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0 Safari/537.36 NewsAgent/1.0"
)

# Finance keywords — strict UPI/CC/banking focus to filter out stock/corporate news
_FINANCE_RELEVANT_KW = [
    # Payments & UPI
    "upi", "credit card", "debit card", "cashback", "reward points", "reward program",
    "contactless payment", "tap to pay", "nfc payment", "digital payment",
    # Banks & issuers
    "hdfc", "icici bank", "axis bank", "amex", "american express", "sbi card", "rupay",
    "kotak mahindra", "indusind bank", "yes bank", "idfc first",
    # Payment apps
    "paytm", "amazon pay", "gpay", "phonepe", "bhim", "google pay",
    # Banking products
    "rbi", "repo rate", "neft", "imps", "rtgs", "bank offer", "banking scheme",
    "emi offer", "loan rate", "fd rate", "fixed deposit", "savings account",
    "fd interest", "savings rate", "rd rate", "recurring deposit",
    # Card-specific
    "card launch", "card benefit", "lounge access", "milestone benefit",
    "annual fee", "credit score", "credit limit", "card reward", "welcome bonus",
    "spend offer", "fuel surcharge", "forex markup",
    # Tax & govt finance
    "tax slab", "income tax", "itr", "gst rate", "subsidy", "govt scheme",
    "insurance premium", "lic", "national pension", "epf", "ppf",
]


def _clean_html(raw: str) -> str:
    try:
        return BeautifulSoup(raw, "html.parser").get_text(separator=" ").strip()[:600]
    except Exception:
        return raw[:600]


def _is_finance_relevant(title: str, content: str) -> bool:
    """Return True if the article has at least some finance signal."""
    text = (title + " " + content).lower()
    return any(kw in text for kw in _FINANCE_RELEVANT_KW)


# ---------------------------------------------------------------------------
# Translation helpers (sync, called via run_in_executor)
# ---------------------------------------------------------------------------

def _detect_lang(text: str) -> str:
    """Return ISO language code or 'en' on failure."""
    try:
        from langdetect import detect, LangDetectException
        clean = re.sub(r"http\S+|[^a-zA-Z\u0900-\u097F\u0C00-\u0C7F\s]", " ", text)
        clean = clean.strip()
        if len(clean) < 20:
            return "en"
        return detect(clean)
    except Exception:
        return "en"


def _translate_sync(text: str) -> str:
    """Translate text to English using Google Translate."""
    if not text:
        return text
    try:
        from deep_translator import GoogleTranslator
        # deep_translator max per call ~ 5000 chars
        return GoogleTranslator(source="auto", target="en").translate(text[:4000]) or text
    except Exception as e:
        logger.debug(f"Translation error: {e}")
        return text


async def _maybe_translate(article: Dict) -> Dict:
    """Detect language; if non-English, translate title + content in thread pool.
    Skips domains that are known to publish in English only."""
    _ENGLISH_DOMAINS = {
        "arxiv.org", "huggingface.co", "techcrunch.com", "venturebeat.com",
        "news.ycombinator.com", "github.com", "thehindu.com", "indianexpress.com",
        "economictimes.com", "moneycontrol.com", "livemint.com", "bankbazaar.com",
        "ndtv.com", "cardinsider.com",
    }
    domain = article.get("source_domain", "").lower()
    if any(d in domain for d in _ENGLISH_DOMAINS):
        return article  # Skip — known English source

    title = article.get("title", "")
    content = article.get("content", "")
    detection_text = f"{title} {content[:150]}"

    loop = asyncio.get_running_loop()
    lang = await loop.run_in_executor(None, _detect_lang, detection_text)

    if lang in ("en", "und", ""):
        return article

    logger.info(f"Translating article from '{lang}': {title[:60]}")
    translated_title = await loop.run_in_executor(None, _translate_sync, title)
    translated_content = await loop.run_in_executor(None, _translate_sync, content)

    return {
        **article,
        "title": translated_title or title,
        "content": translated_content or content,
        "original_language": lang,
        "translated": True,
    }


async def fetch_rss(
    session: aiohttp.ClientSession,
    url: str,
    category: str,
    source_domain: str,
    source_type: str,
    limit: int = 10,
) -> List[Dict]:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            text = await resp.text(errors="replace")
        feed = feedparser.parse(text)
        articles = []
        for entry in feed.entries[:limit]:
            content = entry.get("summary", entry.get("description", ""))
            articles.append(
                {
                    "url": entry.get("link", "").strip(),
                    "title": entry.get("title", "").strip()[:250],
                    "content": _clean_html(content),
                    "source_domain": source_domain,
                    "category": category,
                    "source_type": source_type,
                    "published_at": entry.get("published", ""),
                }
            )
        return [a for a in articles if a["url"] and a["title"]]
    except Exception as e:
        logger.warning(f"RSS fetch failed for {url}: {e}")
        return []


async def fetch_hackernews(session: aiohttp.ClientSession) -> List[Dict]:
    """Fetch HN top stories filtered for AI/DS/finance keywords."""
    KEYWORDS = [
        "ai", "llm", "gpt", "claude", "machine learning", "neural", "langchain",
        "huggingface", "openai", "anthropic", "python", "data science", "inference",
        "model", "agent", "fine-tun", "rag", "retrieval", "embedding",
    ]
    try:
        async with session.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            ids = await resp.json()

        async def fetch_item(sid):
            try:
                async with session.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as r:
                    return await r.json()
            except Exception:
                return None

        stories = await asyncio.gather(*[fetch_item(sid) for sid in ids[:40]])
        articles = []
        for s in stories:
            if not s or s.get("type") != "story":
                continue
            title_lower = s.get("title", "").lower()
            if any(kw in title_lower for kw in KEYWORDS):
                articles.append(
                    {
                        "url": s.get("url") or f"https://news.ycombinator.com/item?id={s['id']}",
                        "title": s.get("title", ""),
                        "content": (
                            f"HackerNews: {s.get('score', 0)} points, "
                            f"{s.get('descendants', 0)} comments"
                        ),
                        "source_domain": "news.ycombinator.com",
                        "category": "tech",
                        "source_type": "community",
                        "published_at": "",
                    }
                )
        return articles[:10]
    except Exception as e:
        logger.warning(f"HackerNews fetch failed: {e}")
        return []


async def fetch_all_sources(state: AgentState) -> Dict:
    """Fetch all configured sources in parallel, then auto-translate non-English."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml, text/xml, */*"}
    connector = aiohttp.TCPConnector(limit=20, ssl=False)

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks = [
            # Finance — general + credit-card/UPI focused
            fetch_rss(session, "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
                      "finance", "economictimes.com", "news"),
            fetch_rss(session, "https://economictimes.indiatimes.com/wealth/rss",
                      "finance", "economictimes.com", "news"),
            fetch_rss(session, "https://www.moneycontrol.com/rss/business.xml",
                      "finance", "moneycontrol.com", "news"),
            fetch_rss(session, "https://www.livemint.com/rss/money",
                      "finance", "livemint.com", "news"),
            fetch_rss(session, "https://feeds.feedburner.com/ndtvprofit-latest",
                      "finance", "ndtv.com", "news"),
            fetch_rss(session, "https://www.bankbazaar.com/rss.xml",
                      "finance", "bankbazaar.com", "news", 8),
            # Tech
            fetch_rss(session, "https://arxiv.org/rss/cs.AI",
                      "tech", "arxiv.org", "research"),
            fetch_rss(session, "https://huggingface.co/blog/feed.xml",
                      "tech", "huggingface.co", "official"),
            fetch_rss(session, "https://techcrunch.com/feed/",
                      "tech", "techcrunch.com", "news", 8),
            fetch_rss(session, "https://venturebeat.com/feed/",
                      "tech", "venturebeat.com", "news", 8),
            fetch_hackernews(session),
            # Government
            fetch_rss(session, "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",
                      "govt", "pib.gov.in", "official"),
            fetch_rss(session, "https://www.thehindu.com/news/national/feeder/default.rss",
                      "govt", "thehindu.com", "news"),
            fetch_rss(session, "https://indianexpress.com/section/india/feed/",
                      "govt", "indianexpress.com", "news"),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_articles = []
    errors = list(state.get("errors", []))
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append(f"Source {i} error: {result}")
        elif isinstance(result, list):
            all_articles.extend(result)

    # Filter finance articles: keep only those with at least one finance keyword
    filtered = []
    for a in all_articles:
        if a.get("category") == "finance" and not _is_finance_relevant(
            a.get("title", ""), a.get("content", "")
        ):
            continue  # skip irrelevant finance article (e.g. stock tips, corporate news)
        filtered.append(a)

    # Auto-translate non-English articles (parallel, thread pool)
    translate_semaphore = asyncio.Semaphore(5)

    async def translate_limited(art):
        async with translate_semaphore:
            return await _maybe_translate(art)

    translated_articles = await asyncio.gather(
        *[translate_limited(a) for a in filtered], return_exceptions=True
    )
    final_articles = [
        a for a in translated_articles if isinstance(a, dict)
    ]

    translated_count = sum(1 for a in final_articles if a.get("translated"))
    if translated_count:
        logger.info(f"Auto-translated {translated_count} non-English articles")

    logger.info(f"Fetched {len(final_articles)} articles ({len(all_articles)} raw, {len(all_articles) - len(filtered)} finance-filtered)")
    return {
        "raw_articles": final_articles,
        "stats": {"total_fetched": len(final_articles), "translated": translated_count},
        "errors": errors,
    }
