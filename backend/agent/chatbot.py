"""Conversational chatbot with multi-query web search and strong result grounding."""
import asyncio
import logging
import os
import re
import time
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

# ---------------------------------------------------------------------------
# System prompt — forces grounding, bans generic responses
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a personal assistant for a specific person. Know them well:
- 30-year-old Data Scientist, Hyderabad (Madhapur), ₹2.3L/month
- Regular spends: Swiggy, Zomato, Amazon, flights, hotels, dining, entertainment
- Cards likely owned: HDFC, ICICI, Axis, possibly Amex or SBI
- Goals: squeeze every rupee of cashback/rewards, save taxes, grow in AI/ML

━━ MANDATORY RULES — NEVER BREAK THESE ━━

1. EXTRACT FROM SEARCH RESULTS FIRST
   The search results below contain real, current information.
   Read each result carefully. Pull out: card names, exact percentages, 
   platform names, deadlines, caps, eligibility. Use these as your answer.

2. BE BRUTALLY SPECIFIC
   ❌ Bad: "Some credit cards offer cashback on food delivery."
   ✅ Good: "HDFC Millennia gives 5% cashback on Swiggy/Zomato (capped ₹1000/month).
             Axis Flipkart card gives 1.5% unlimited. SBI Cashback gives 5% online."

3. PERSONALIZE TO THIS USER
   Relate every answer to their actual spending. If they ask about food cashback,
   mention Swiggy/Zomato specifically. If they ask about travel, mention 
   flight/hotel platforms they'd use.

4. CITE INLINE
   When stating a specific fact from search results, add the source name inline:
   "According to CardInsider, the HDFC Millennia gives..."
   Don't just dump links — weave them into the answer.

5. ADMIT GAPS
   If search results don't cover something, say:
   "The search results don't mention this specifically. Based on general knowledge: ..."
   Never silently fill gaps with hallucinated numbers.

6. TELEGRAM FORMAT
   Use <b>bold</b> for card names and key numbers.
   Use bullet points (•) for comparisons.
   Max 350 words unless user asks for deep dive.
   No filler phrases: "It's worth noting", "In conclusion", "As we can see".
"""

# ---------------------------------------------------------------------------
# Intent detection → targeted multi-query search
# ---------------------------------------------------------------------------

_FINANCE_KW = ["credit card", "cashback", "reward", "upi", "bank", "card", "emi",
               "hdfc", "icici", "axis", "amex", "sbi", "rupay", "paytm", "gpay",
               "interest rate", "fd", "loan", "insurance", "invest"]
_TECH_KW = ["ai", "llm", "ml", "model", "gpt", "claude", "python", "data science",
            "machine learning", "tool", "framework", "langchain", "agent", "openai",
            "huggingface", "job", "salary", "interview", "resume"]
_GOVT_KW = ["tax", "itr", "gst", "income tax", "scheme", "subsidy", "telangana",
            "hyderabad", "govt", "government", "pib", "rbi", "budget", "rebate"]


def _detect_intent(text: str) -> str:
    t = text.lower()
    if any(k in t for k in _FINANCE_KW):
        return "finance"
    if any(k in t for k in _TECH_KW):
        return "tech"
    if any(k in t for k in _GOVT_KW):
        return "govt"
    return "general"


def _build_queries(user_message: str, intent: str) -> List[str]:
    """Build 2–3 targeted, specific search queries."""
    msg = user_message.strip()

    if intent == "finance":
        return [
            f"{msg} India 2026",
            f"best {msg} HDFC ICICI Axis SBI cashback India 2026",
            f"{msg} site:cardinsider.com OR site:bankbazaar.com OR site:cardexpert.in",
        ]
    elif intent == "tech":
        return [
            f"{msg} 2026",
            f"{msg} latest announcement release",
        ]
    elif intent == "govt":
        return [
            f"{msg} India official 2026",
            f"{msg} Hyderabad Telangana",
        ]
    else:
        return [
            f"{msg} India 2026",
            msg,
        ]


# ---------------------------------------------------------------------------
# Web search — parallel multi-query with deduplication
# ---------------------------------------------------------------------------

def _ddg_search_sync(query: str, max_results: int = 5) -> List[Dict]:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results, region="in-en"))
    except Exception as e:
        logger.warning(f"DDG search error for '{query[:60]}': {e}")
        return []


async def _search_one(query: str, max_results: int = 5) -> List[Dict]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _ddg_search_sync, query, max_results)


async def search_multi(queries: List[str], results_per_query: int = 4) -> List[Dict]:
    """Run all queries in parallel, deduplicate by URL, return top results."""
    all_results = await asyncio.gather(
        *[_search_one(q, results_per_query) for q in queries],
        return_exceptions=True,
    )
    seen_urls = set()
    merged = []
    for batch in all_results:
        if isinstance(batch, list):
            for r in batch:
                url = r.get("href", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    merged.append(r)
    return merged[:10]  # cap at 10 unique results


def _format_results(results: List[Dict]) -> str:
    if not results:
        return "No search results found."
    lines = [
        "## SEARCH RESULTS (read carefully — extract specific facts from these)\n"
    ]
    for i, r in enumerate(results, 1):
        title = r.get("title", "").strip()
        url = r.get("href", "").strip()
        body = r.get("body", "").strip()[:500]
        lines.append(
            f"[{i}] {title}\n"
            f"    Source: {url}\n"
            f"    {body}\n"
        )
    lines.append("## END OF SEARCH RESULTS\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conversation history (MongoDB)
# ---------------------------------------------------------------------------

async def get_history(db, chat_id: str, limit: int = 8) -> List[Dict]:
    docs = await db.conversations.find(
        {"chat_id": chat_id}, {"_id": 0, "role": 1, "content": 1}
    ).sort("ts", -1).to_list(limit)
    # Return chronological (oldest first), strip search context from stored history
    history = []
    for d in reversed(docs):
        content = d["content"]
        # Remove search results block that was stored (keep only the user question part)
        content = re.sub(r"\n\n## SEARCH RESULTS.*?## END OF SEARCH RESULTS\n", "", content, flags=re.DOTALL)
        content = re.sub(r"\n\[Digest Article.*?\n\n", "", content, flags=re.DOTALL)
        history.append({"role": d["role"], "content": content.strip()})
    return history


async def save_message(db, chat_id: str, role: str, content: str):
    from datetime import datetime, timezone
    await db.conversations.insert_one({
        "chat_id": chat_id,
        "role": role,
        "content": content[:5000],
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    # Prune — keep last 60 messages per chat
    all_docs = await db.conversations.find(
        {"chat_id": chat_id}, {"_id": 1}
    ).sort("ts", -1).to_list(200)
    if len(all_docs) > 60:
        old_ids = [d["_id"] for d in all_docs[60:]]
        await db.conversations.delete_many({"_id": {"$in": old_ids}})


async def clear_history(db, chat_id: str):
    await db.conversations.delete_many({"chat_id": chat_id})


# ---------------------------------------------------------------------------
# Markdown → Telegram HTML
# ---------------------------------------------------------------------------

def _md_to_html(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Convert markdown links [text](url) → Telegram HTML
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r'<a href="\2">\1</a>', text)
    # Unescape HTML entities that might conflict
    text = text.replace("&", "&amp;").replace("&amp;amp;", "&amp;")
    # Fix double-encoded entities from the above
    text = re.sub(r"&amp;(lt|gt|amp|quot);", r"&\1;", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Detect digest item reference ("more about item 2", "explain 3")
# ---------------------------------------------------------------------------

def _extract_item_ref(text: str) -> Optional[int]:
    for pattern in [
        r"\b(?:item|number|#|article|point)\s*(\d+)\b",
        r"\b(?:more about|explain|tell me about|details? (?:on|of|about))\s+(?:item\s*)?(\d+)\b",
        r"\b(\d+)(?:st|nd|rd|th)?\s+(?:item|article|point|one)\b",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

_last_req: Dict[str, float] = {}
_RATE_LIMIT_SEC = 3


# ---------------------------------------------------------------------------
# Main chat function
# ---------------------------------------------------------------------------

async def get_chat_response(db, chat_id: str, user_message: str) -> str:
    """Pipeline: detect intent → multi-query search → LLM with grounding → save."""

    now = time.time()
    if now - _last_req.get(chat_id, 0) < _RATE_LIMIT_SEC:
        return "⏳ Still processing... try again in a moment."
    _last_req[chat_id] = now

    intent = _detect_intent(user_message)

    # ── Digest item reference ──────────────────────────────────────────────
    digest_context = ""
    item_ref = _extract_item_ref(user_message)
    if item_ref:
        latest = await db.digests.find_one(
            {"status": {"$in": ["sent", "pending"]}}, sort=[("created_at", -1)]
        )
        if latest:
            arts = latest.get("articles", [])
            if 1 <= item_ref <= len(arts):
                art = arts[item_ref - 1]
                # Use article title as the primary search seed
                user_message_for_search = (
                    f"{art.get('title', user_message)} India 2026 details"
                )
                digest_context = (
                    f"\n## Digest Article #{item_ref} (user is asking about this)\n"
                    f"Title: {art.get('title','')}\n"
                    f"Summary: {art.get('summary','No summary available.')}\n"
                    f"Source URL: {art.get('url','')}\n"
                    f"Category: {art.get('category','')} | "
                    f"Credibility: {art.get('credibility_score',0)}/100\n"
                    f"Why it matters: {art.get('why_it_matters','')}\n"
                )
            else:
                user_message_for_search = user_message
        else:
            user_message_for_search = user_message
    else:
        user_message_for_search = user_message

    # ── Multi-query search ─────────────────────────────────────────────────
    queries = _build_queries(user_message_for_search, intent)
    logger.info(f"Chat search queries: {queries}")
    results = await search_multi(queries, results_per_query=4)
    search_block = _format_results(results)
    logger.info(f"Chat search: {len(results)} unique results from {len(queries)} queries")

    # ── Conversation history ───────────────────────────────────────────────
    history = await get_history(db, chat_id, limit=8)

    # ── Build LLM messages ─────────────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)

    # User turn: question + search context + digest context
    user_turn = (
        f"Question: {user_message}\n\n"
        f"{search_block}"
        + (f"\n{digest_context}" if digest_context else "")
        + "\n\nAnswer based on the search results above. Be specific."
    )
    messages.append({"role": "user", "content": user_turn})

    # ── LLM call ──────────────────────────────────────────────────────────
    if not HF_TOKEN:
        return "⚠️ HF_TOKEN not set. Cannot generate response."

    answer = ""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                HF_API_URL,
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": HF_MODEL,
                    "messages": messages,
                    "max_tokens": 900,
                    "temperature": 0.5,   # lower temp = more faithful to search results
                },
            )
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"LLM {resp.status_code}: {resp.text[:200]}")
                answer = "I couldn't reach the LLM right now. Please try again."
    except Exception as e:
        logger.error(f"Chat LLM error: {e}")
        answer = "Something went wrong. Please try again."

    # ── Post-process ──────────────────────────────────────────────────────
    answer = _md_to_html(answer)

    # Append source links (top 3 unique, non-redundant with inline citations)
    if results:
        top_sources = results[:4]
        sources_block = "\n\n🔗 <b>Sources:</b>\n" + "\n".join(
            f'• <a href="{r["href"]}">{r["title"][:65]}</a>'
            for r in top_sources if r.get("href")
        )
        answer += sources_block

    # ── Save to history ───────────────────────────────────────────────────
    await save_message(db, chat_id, "user", user_turn)     # store with search context
    await save_message(db, chat_id, "assistant", answer)

    return answer
