"""Conversational chatbot with web search for Telegram."""
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

SYSTEM_PROMPT = """\
You are a smart personal finance and tech assistant for a specific user. Here is their profile:
- 30-year-old Data Scientist in Hyderabad (Madhapur), ₹2.3L/month salary
- Spends on food delivery, travel, dining out, entertainment
- Primary interest: UPI cashback, credit card rewards & offers (HDFC, ICICI, Axis, Amex, SBI, RuPay)
- Also interested in: AI/ML/LLM career news, income tax saving, Telangana govt schemes

When answering:
- Be concise and conversational — this is Telegram, not a blog
- Use <b>bold</b> for key terms, use bullet points for lists
- Always cite sources using <a href="URL">Title</a> links when search results are provided
- Highlight what is directly actionable for THIS user (card name, cashback %, deadline, how to activate)
- If answer is in the search results, use that. Otherwise use your training knowledge and mention it.
- Keep responses under 600 words unless user asks for a deep dive
- Never hallucinate specific cashback numbers — if unsure, say "check the official website"
"""

# Simple in-memory rate limiter (per chat_id)
_last_req: Dict[str, float] = {}
_RATE_LIMIT_SEC = 3


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

def _ddg_search_sync(query: str, max_results: int = 5) -> List[Dict]:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(
                query,
                max_results=max_results,
                region="in-en",
            ))
    except Exception as e:
        logger.warning(f"DDG search error: {e}")
        return []


async def search_web(query: str, max_results: int = 5) -> List[Dict]:
    """Async DuckDuckGo search — returns list of {title, href, body}."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _ddg_search_sync, query, max_results)


def _format_results(results: List[Dict]) -> str:
    if not results:
        return ""
    lines = ["<search_results>"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r.get('title', '')}\n"
            f"   URL: {r.get('href', '')}\n"
            f"   {r.get('body', '')[:350]}"
        )
    lines.append("</search_results>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conversation history (MongoDB)
# ---------------------------------------------------------------------------

async def get_history(db, chat_id: str, limit: int = 10) -> List[Dict]:
    docs = await db.conversations.find(
        {"chat_id": chat_id}, {"_id": 0, "role": 1, "content": 1}
    ).sort("ts", -1).to_list(limit)
    return [{"role": d["role"], "content": d["content"]} for d in reversed(docs)]


async def save_message(db, chat_id: str, role: str, content: str):
    from datetime import datetime, timezone
    await db.conversations.insert_one({
        "chat_id": chat_id,
        "role": role,
        "content": content[:4000],
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
# Markdown → Telegram HTML (Llama sometimes returns markdown)
# ---------------------------------------------------------------------------

def _md_to_html(text: str) -> str:
    # Bold: **text** → <b>text</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    # Italic: *text* → <i>text</i>  (avoid matching **)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    # Code: `text` → <code>text</code>
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    # Strip code fences (```...```)
    text = re.sub(r"```[\s\S]*?```", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Detect if user references a digest item ("more about item 2", "details 3")
# ---------------------------------------------------------------------------

def _extract_item_ref(text: str) -> Optional[int]:
    m = re.search(r"\b(?:item|number|#|article|point)\s*(\d+)\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Plain number at end: "more about 3"
    m2 = re.search(r"\bmore\s+(?:about|on|details?)\s+(\d+)\b", text, re.IGNORECASE)
    if m2:
        return int(m2.group(1))
    return None


# ---------------------------------------------------------------------------
# Main chat function
# ---------------------------------------------------------------------------

async def get_chat_response(db, chat_id: str, user_message: str) -> str:
    """Full pipeline: web search → history → LLM → save → return HTML response."""

    # Rate limit
    now = time.time()
    if now - _last_req.get(chat_id, 0) < _RATE_LIMIT_SEC:
        return "⏳ One moment — processing your previous message..."
    _last_req[chat_id] = now

    # 1. Build search query
    search_query = user_message

    # 2. Check for "more about item N" — enrich query with digest article title
    digest_article_context = ""
    item_ref = _extract_item_ref(user_message)
    if item_ref:
        latest = await db.digests.find_one(
            {"status": {"$in": ["sent", "pending"]}}, sort=[("created_at", -1)]
        )
        if latest:
            articles = latest.get("articles", [])
            if 1 <= item_ref <= len(articles):
                art = articles[item_ref - 1]
                title = art.get("title", "")
                search_query = f"{title} details {art.get('category','')} India 2026"
                digest_article_context = (
                    f"\n[Digest Article #{item_ref}]\n"
                    f"Title: {art.get('title','')}\n"
                    f"Summary: {art.get('summary','')}\n"
                    f"Source: {art.get('url','')}\n"
                    f"Category: {art.get('category','')} | "
                    f"Credibility: {art.get('credibility_score',0)}/100\n"
                )

    # 3. Search web
    results = await search_web(search_query)
    search_context = _format_results(results)

    # 4. Load conversation history
    history = await get_history(db, chat_id)

    # 5. Build messages for LLM
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)

    user_content = user_message
    if digest_article_context:
        user_content += f"\n{digest_article_context}"
    if search_context:
        user_content += f"\n\n{search_context}"

    messages.append({"role": "user", "content": user_content})

    # 6. Call LLM
    if not HF_TOKEN:
        return "⚠️ LLM not configured. Set HF_TOKEN in .env."

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
                    "temperature": 0.65,
                },
            )
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"LLM error {resp.status_code}: {resp.text[:200]}")
                answer = "Sorry, I couldn't generate a response right now. Try again in a moment."
    except Exception as e:
        logger.error(f"Chat LLM error: {e}")
        answer = "Something went wrong on my end. Please try again."

    # 7. Convert any markdown to Telegram HTML
    answer = _md_to_html(answer)

    # 8. Append source links
    if results:
        sources = "\n\n🔗 <b>Sources:</b>\n" + "\n".join(
            f'• <a href="{r["href"]}">{r["title"][:70]}</a>'
            for r in results[:3] if r.get("href")
        )
        answer += sources

    # 9. Save to conversation history
    await save_message(db, chat_id, "user", user_message)
    await save_message(db, chat_id, "assistant", answer)

    return answer
