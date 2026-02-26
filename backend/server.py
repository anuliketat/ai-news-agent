"""FastAPI server for AI News Monitoring Agent."""
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, FastAPI, Header, HTTPException, Request
from motor.motor_asyncio import AsyncIOMotorClient
from starlette.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
mongo_url = os.environ["MONGO_URL"]
_mongo_client = AsyncIOMotorClient(mongo_url)
db = _mongo_client[os.environ["DB_NAME"]]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="AI News Agent API")
api_router = APIRouter(prefix="/api")

TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
BACKEND_URL = os.environ.get("BACKEND_URL", "")
AGENT_SECRET_KEY = os.environ.get("AGENT_SECRET_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

_MAX_TG_CHARS = 3800


def _split_telegram_message(text: str) -> list:
    """Split a long message into ≤3800-char chunks at paragraph boundaries."""
    if len(text) <= _MAX_TG_CHARS:
        return [text]
    chunks = []
    current = ""
    for para in text.split("\n\n"):
        block = (current + "\n\n" + para).strip() if current else para
        if len(block) <= _MAX_TG_CHARS:
            current = block
        else:
            if current:
                chunks.append(current)
            # If single para still too long, hard-split
            if len(para) > _MAX_TG_CHARS:
                for i in range(0, len(para), _MAX_TG_CHARS):
                    chunks.append(para[i : i + _MAX_TG_CHARS])
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    if TELEGRAM_BOT_TOKEN and BACKEND_URL:
        from agent.telegram_handler import setup_webhook
        webhook_url = f"{BACKEND_URL}/api/telegram/webhook"
        ok = await setup_webhook(webhook_url)
        if ok:
            logger.info(f"Telegram webhook registered: {webhook_url}")

        # Register bot commands so they appear in Telegram's menu
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setMyCommands",
                json={"commands": [
                    {"command": "refresh", "description": "Check for new updates right now"},
                    {"command": "top",     "description": "Today's top 5 most credible articles"},
                    {"command": "history", "description": "Browse last 7 digest runs"},
                    {"command": "status",  "description": "Show last run stats"},
                    {"command": "help",    "description": "Show all commands"},
                ]}
            )
        logger.info("Telegram bot commands registered")


@app.on_event("shutdown")
async def shutdown():
    _mongo_client.close()


# ---------------------------------------------------------------------------
# Agent trigger
# ---------------------------------------------------------------------------
async def _run_agent_task(run_id: str):
    try:
        from agent.main import run_agent
        await run_agent(run_id)
    except Exception as e:
        logger.error(f"Agent run {run_id} failed: {e}", exc_info=True)
        await db.agent_runs.update_one(
            {"run_id": run_id},
            {"$set": {"status": "failed", "error": str(e),
                      "updated_at": datetime.now(timezone.utc).isoformat()}},
            upsert=True,
        )
        # Notify via Telegram if this was a manual refresh
        run_doc = await db.agent_runs.find_one({"run_id": run_id})
        if run_doc and run_doc.get("triggered_by") == "telegram_refresh" and TELEGRAM_CHAT_ID:
            from agent.telegram_handler import send_message
            await send_message(
                TELEGRAM_CHAT_ID,
                "❌ <b>Refresh failed.</b> Please try again or wait for the next scheduled run."
            )


@api_router.post("/agent/trigger")
async def trigger_agent(
    background_tasks: BackgroundTasks,
    authorization: str = Header(default=None),
):
    """Trigger a news agent run (called by GitHub Actions or manually)."""
    if AGENT_SECRET_KEY and authorization != f"Bearer {AGENT_SECRET_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    run_id = str(uuid.uuid4())
    # Record run start
    await db.agent_runs.insert_one({
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    background_tasks.add_task(_run_agent_task, run_id)
    return {"status": "triggered", "run_id": run_id}


# ---------------------------------------------------------------------------
# Telegram webhook
# ---------------------------------------------------------------------------
@api_router.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """Handle incoming Telegram messages (YES approval, details N, feedback)."""
    try:
        data = await request.json()
    except Exception:
        return {"ok": True}

    message = data.get("message", {})
    chat_id = str(message.get("chat", {}).get("id", ""))
    text = message.get("text", "").strip()

    if chat_id != str(TELEGRAM_CHAT_ID):
        return {"ok": True}

    text_upper = text.upper()
    text_lower = text.lower()

    if text_upper == "YES":
        await _handle_approval(chat_id)
    elif text_upper in ("NO", "SKIP"):
        await _handle_rejection(chat_id)
    elif text_lower in ("/refresh", "refresh"):
        await _handle_refresh(chat_id)
    elif text_lower in ("/status", "status"):
        await _handle_status(chat_id)
    elif text_lower in ("/history", "history"):
        await _handle_history(chat_id)
    elif text_lower.startswith("/top") or text_lower == "top":
        await _handle_top(chat_id)
    elif text_lower.startswith("details "):
        await _handle_details(chat_id, text)
    elif text_lower.startswith("feedback "):
        await _handle_feedback(chat_id, text)
    elif text_lower in ("/start", "/help", "help"):
        await _send_help(chat_id)

    return {"ok": True}


async def _handle_approval(chat_id: str):
    from agent.telegram_handler import send_message

    pending = await db.digests.find_one(
        {"status": "pending"}, sort=[("created_at", -1)]
    )
    if not pending:
        await send_message(chat_id, "No pending digest found. The next run is scheduled.")
        return

    await db.digests.update_one(
        {"_id": pending["_id"]},
        {"$set": {"status": "sent", "sent_at": datetime.now(timezone.utc).isoformat()}},
    )

    digest_text = pending.get("digest_text", "")
    chunks = _split_telegram_message(digest_text)
    for chunk in chunks:
        if chunk.strip():
            await send_message(chat_id, chunk)


async def _handle_rejection(chat_id: str):
    from agent.telegram_handler import send_message
    await db.digests.update_many(
        {"status": "pending"},
        {"$set": {"status": "rejected", "updated_at": datetime.now(timezone.utc).isoformat()}},
    )
    await send_message(chat_id, "Digest skipped. See you at the next scheduled run! ✅")


async def _handle_details(chat_id: str, text: str):
    from agent.telegram_handler import send_message
    try:
        num = int(text.split(" ")[1])
    except (ValueError, IndexError):
        await send_message(chat_id, "Usage: <b>details 1</b> (replace 1 with item number)")
        return

    latest = await db.digests.find_one(
        {"status": {"$in": ["sent", "pending"]}}, sort=[("created_at", -1)]
    )
    if not latest:
        await send_message(chat_id, "No recent digest found.")
        return

    articles = latest.get("articles", [])
    if not (1 <= num <= len(articles)):
        await send_message(chat_id, f"Item {num} not found. Digest has {len(articles)} items.")
        return

    art = articles[num - 1]
    msg = (
        f"<b>{art.get('title', 'Untitled')}</b>\n\n"
        f"{art.get('content', 'No content available.')}\n\n"
        f"🔗 <a href=\"{art.get('url', '')}\">Read full article</a>\n"
        f"Source: {art.get('source_domain', '')} | {art.get('source_type', '').capitalize()}\n"
        f"Credibility: {art.get('credibility_score', 'N/A')}/100"
    )
    await send_message(chat_id, msg)


async def _handle_feedback(chat_id: str, text: str):
    from agent.telegram_handler import send_message
    parts = text.split(" ", 2)
    if len(parts) < 3:
        await send_message(chat_id, "Usage: <b>feedback 1 too generic</b>")
        return
    try:
        num = int(parts[1])
        feedback_text = parts[2]
    except ValueError:
        await send_message(chat_id, "Usage: <b>feedback 1 too generic</b>")
        return

    latest = await db.digests.find_one(
        {"status": {"$in": ["sent", "pending"]}}, sort=[("created_at", -1)]
    )
    if latest:
        articles = latest.get("articles", [])
        if 1 <= num <= len(articles):
            await db.articles.update_one(
                {"url": articles[num - 1].get("url", "")},
                {"$set": {"user_feedback": feedback_text}},
            )
    await send_message(chat_id, f"Thanks! Feedback noted for item {num} 👍")


async def _handle_refresh(chat_id: str):
    """Trigger an on-demand agent run and notify user."""
    from agent.telegram_handler import send_message

    # Prevent duplicate concurrent runs
    running = await db.agent_runs.find_one({"status": "running"})
    if running:
        elapsed_sec = 0
        try:
            started = datetime.fromisoformat(running.get("started_at", ""))
            elapsed_sec = int((datetime.now(timezone.utc) - started).total_seconds())
        except Exception:
            pass
        await send_message(
            chat_id,
            f"⏳ A run is already in progress (started {elapsed_sec}s ago).\n"
            f"You'll get a digest preview shortly — no need to refresh again."
        )
        return

    # Acknowledge immediately
    await send_message(
        chat_id,
        "🔄 <b>Refreshing now...</b>\n"
        "<i>Fetching latest news from all sources. "
        "You'll get a preview in ~15–30 seconds.</i>"
    )

    # Kick off agent run in background
    run_id = str(uuid.uuid4())
    await db.agent_runs.insert_one({
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "triggered_by": "telegram_refresh",
    })

    import asyncio
    asyncio.create_task(_run_agent_task(run_id))


async def _handle_status(chat_id: str):
    """Show last run stats and pending digest info."""
    from agent.telegram_handler import send_message

    last_run = await db.agent_runs.find_one({}, sort=[("started_at", -1)])
    pending = await db.digests.find_one({"status": "pending"}, sort=[("created_at", -1)])
    total_articles = await db.articles.count_documents({})
    total_runs = await db.agent_runs.count_documents({})

    stats = last_run.get("stats", {}) if last_run else {}
    status_emoji = {"completed": "✅", "running": "⏳", "failed": "❌"}.get(
        last_run.get("status", ""), "❓"
    ) if last_run else "❓"

    msg = (
        "<b>📊 Agent Status</b>\n\n"
        f"{status_emoji} <b>Last run:</b> {last_run.get('status', 'N/A') if last_run else 'Never'}\n"
        f"   Fetched: {stats.get('total_fetched', 0)} articles\n"
        f"   New (after dedup): {stats.get('after_dedup', 0)}\n"
        f"   Verified: {stats.get('verified_after_xref', stats.get('verified', 0))}\n"
        f"   Actionable sent: {stats.get('actionable', 0)}\n"
        f"   Translated: {stats.get('translated', 0)}\n\n"
        f"📬 <b>Pending digest:</b> {'Yes — reply YES to receive' if pending else 'None'}\n"
        f"🗄 <b>DB:</b> {total_articles} articles stored | {total_runs} total runs\n\n"
        f"<i>Send /refresh to check for new updates now</i>"
    )
    await send_message(chat_id, msg)


async def _handle_history(chat_id: str):
    """Show last 7 digest runs with date, count, and status."""
    from agent.telegram_handler import send_message

    digests = await db.digests.find(
        {}, {"digest_text": 0}
    ).sort("created_at", -1).to_list(7)

    if not digests:
        await send_message(chat_id, "No digest history yet. Send /refresh to run now!")
        return

    STATUS_ICON = {"sent": "✅", "pending": "📬", "rejected": "🚫", "skipped": "⏭"}
    lines = ["<b>📜 Digest History (last 7 runs)</b>\n"]

    for i, d in enumerate(digests, 1):
        created = d.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created)
            # Convert to IST (+5:30)
            from datetime import timedelta
            ist = dt + timedelta(hours=5, minutes=30)
            date_str = ist.strftime("%b %d, %I:%M %p IST")
        except Exception:
            date_str = created[:16]

        stats = d.get("stats", {})
        articles = d.get("articles", [])
        verified = stats.get("verified_after_xref", stats.get("verified", 0))
        total = len(articles)
        status = d.get("status", "unknown")
        icon = STATUS_ICON.get(status, "❓")

        lines.append(
            f"{i}. <b>{date_str}</b> — {total} items ({verified} verified) {icon}"
        )

    lines.append("\n<i>Send /top to get today's best articles</i>")
    await send_message(chat_id, "\n".join(lines))


async def _handle_top(chat_id: str):
    """Re-send top 5 most credible articles from the latest digest."""
    from agent.telegram_handler import send_message
    from agent.nodes.digest import STATUS_EMOJI, STATUS_LABEL

    # Get the most recent sent or pending digest
    latest = await db.digests.find_one(
        {"status": {"$in": ["sent", "pending"]}},
        sort=[("created_at", -1)]
    )

    if not latest:
        await send_message(
            chat_id,
            "No digest available yet. Send /refresh to fetch the latest news!"
        )
        return

    articles = latest.get("articles", [])
    if not articles:
        await send_message(chat_id, "No articles in the latest digest.")
        return

    # Sort by credibility and take top 5
    top5 = sorted(articles, key=lambda x: x.get("credibility_score", 0), reverse=True)[:5]

    created = latest.get("created_at", "")
    try:
        from datetime import timedelta
        dt = datetime.fromisoformat(created)
        ist = dt + timedelta(hours=5, minutes=30)
        date_str = ist.strftime("%b %d, %I:%M %p IST")
    except Exception:
        date_str = "Latest"

    CATEGORY_EMOJI = {"finance": "💳", "tech": "🤖", "govt": "🏛"}
    lines = [f"<b>⭐ Top 5 Articles — {date_str}</b>\n"]

    for i, art in enumerate(top5, 1):
        status = art.get("validation_status", "unverified")
        score = art.get("credibility_score", 0)
        cat = art.get("category", "tech")
        title = art.get("title", "Untitled")
        summary = art.get("summary", "").strip()
        why = art.get("why_it_matters", "")
        url = art.get("url", "")
        translated = art.get("translated", False)
        lang_tag = f" <i>[{art.get('original_language','?')}→en]</i>" if translated else ""

        link = f'<a href="{url}">Read</a>' if url else ""
        summary_line = f"\n   📝 <i>{summary[:180]}</i>" if summary else ""
        why_line = f"\n   📌 {why}" if why else ""

        lines.append(
            f"{i}. {CATEGORY_EMOJI.get(cat,'📰')} <b>{title}</b>{lang_tag}\n"
            f"   {STATUS_EMOJI.get(status,'⚠️')} {STATUS_LABEL.get(status,'Unverified')} · {score}/100"
            f"{summary_line}"
            f"{why_line}\n"
            f"   🔗 {link}"
        )

    lines.append("\n<i>Send /history to browse all past digests</i>")
    await send_message(chat_id, "\n\n".join(lines))


async def _send_help(chat_id: str):
    from agent.telegram_handler import send_message
    help_text = (
        "<b>AI News Agent — Commands</b>\n\n"
        "🔄 <b>/refresh</b> — Check for new updates right now\n"
        "⭐ <b>/top</b> — Re-send today's top 5 most credible articles\n"
        "📜 <b>/history</b> — Browse last 7 digest runs\n"
        "📊 <b>/status</b> — Show last run stats &amp; pending digest\n\n"
        "📬 <b>When you get a digest preview:</b>\n"
        "  • Reply <b>YES</b> — Receive the full digest\n"
        "  • Reply <b>NO</b> — Skip this digest\n\n"
        "📖 <b>After receiving digest:</b>\n"
        "  • <b>details 1</b> — Full content of item 1\n"
        "  • <b>feedback 2 too generic</b> — Submit feedback\n\n"
        "⏰ <i>Runs automatically at 9 AM and 6 PM IST</i>"
    )
    await send_message(chat_id, help_text)


# ---------------------------------------------------------------------------
# Status & history endpoints
# ---------------------------------------------------------------------------
@api_router.get("/agent/status")
async def get_agent_status():
    last_run = await db.agent_runs.find_one({}, {"_id": 0}, sort=[("started_at", -1)])
    pending_digest = await db.digests.find_one(
        {"status": "pending"}, {"_id": 0, "digest_text": 0, "articles": 0},
        sort=[("created_at", -1)]
    )
    return {
        "last_run": last_run,
        "pending_digest": pending_digest,
    }


@api_router.get("/agent/history")
async def get_digest_history():
    digests = await db.digests.find(
        {}, {"_id": 0, "digest_text": 0}
    ).sort("created_at", -1).to_list(20)
    return {"digests": digests, "count": len(digests)}


@api_router.get("/agent/runs")
async def get_agent_runs():
    runs = await db.agent_runs.find(
        {}, {"_id": 0}
    ).sort("started_at", -1).to_list(20)
    return {"runs": runs, "count": len(runs)}


@api_router.get("/agent/articles")
async def get_recent_articles(limit: int = 50, category: str = None):
    query = {}
    if category:
        query["category"] = category
    articles = await db.articles.find(
        query, {"_id": 0}
    ).sort("fetched_at", -1).to_list(limit)
    return {"articles": articles, "count": len(articles)}


@api_router.get("/")
async def root():
    return {
        "service": "AI News Monitoring Agent",
        "status": "running",
        "endpoints": [
            "POST /api/agent/trigger",
            "GET /api/agent/status",
            "GET /api/agent/history",
            "GET /api/agent/runs",
            "GET /api/agent/articles",
            "POST /api/telegram/webhook",
        ],
    }


app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
