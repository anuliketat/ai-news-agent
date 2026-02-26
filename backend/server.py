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
    # Register Telegram webhook if configured
    if TELEGRAM_BOT_TOKEN and BACKEND_URL:
        from agent.telegram_handler import setup_webhook
        webhook_url = f"{BACKEND_URL}/api/telegram/webhook"
        ok = await setup_webhook(webhook_url)
        if ok:
            logger.info(f"Telegram webhook registered: {webhook_url}")


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

    if text_upper == "YES":
        await _handle_approval(chat_id)
    elif text_upper in ("NO", "SKIP"):
        await _handle_rejection(chat_id)
    elif text_lower := text.lower():
        if text_lower.startswith("details "):
            await _handle_details(chat_id, text)
        elif text_lower.startswith("feedback "):
            await _handle_feedback(chat_id, text)
        elif text_lower in ("/start", "/help"):
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


async def _send_help(chat_id: str):
    from agent.telegram_handler import send_message
    help_text = (
        "<b>AI News Agent — Commands</b>\n\n"
        "📬 When you get a digest preview:\n"
        "  • Reply <b>YES</b> — Receive full digest\n"
        "  • Reply <b>NO</b> — Skip this digest\n\n"
        "📖 After receiving digest:\n"
        "  • <b>details 1</b> — Full content of item 1\n"
        "  • <b>feedback 2 too generic</b> — Submit feedback\n\n"
        "ℹ️ Runs automatically at 9 AM and 6 PM IST"
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
