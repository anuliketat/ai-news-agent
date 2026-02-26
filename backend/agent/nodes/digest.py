"""Build the Telegram digest and send the approval request."""
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List

from ..state import AgentState
from ..telegram_handler import send_message

logger = logging.getLogger(__name__)

TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

CATEGORY_EMOJI = {"finance": "💳", "tech": "🤖", "govt": "🏛"}
STATUS_EMOJI = {"verified": "✅", "unverified": "⚠️", "conflicting": "❌"}
STATUS_LABEL = {"verified": "Verified", "unverified": "Unverified", "conflicting": "Conflicting"}
MAX_DIGEST_ITEMS = 15

# UPI/credit-card keywords for priority boosting finance articles
_CC_UPI_BOOST_KW = [
    "upi", "credit card", "cashback", "reward", "offer", "hdfc card",
    "icici card", "axis card", "amex", "sbi card", "rupay", "paytm",
    "amazon pay", "gpay", "phonepe", "lounge", "milestone", "emi offer",
    "zero fee", "annual fee", "welcome bonus", "card launch", "card benefit",
    "spend offer", "surcharge", "bonus points",
]


def _cc_upi_boost(article: Dict) -> int:
    """Return 0-30 priority boost for UPI/credit-card finance articles."""
    if article.get("category") != "finance":
        return 0
    text = (article.get("title", "") + " " + article.get("content", "")).lower()
    hits = sum(1 for kw in _CC_UPI_BOOST_KW if kw in text)
    return min(hits * 10, 30)


def _format_article(idx: int, article: Dict) -> str:
    status = article.get("validation_status", "unverified")
    credibility = article.get("credibility_score", 50)
    source_type = article.get("source_type", "news").capitalize()
    url = article.get("url", "")
    title = article.get("title", "Untitled")
    why = article.get("why_it_matters", "")
    reasoning = article.get("reasoning", "")
    summary = article.get("summary", "").strip()
    translated = article.get("translated", False)

    link_text = f'<a href="{url}">Source</a>' if url else "No link"
    lang_tag = f" <i>[Auto-translated from {article.get('original_language','?')}]</i>" if translated else ""
    matters_line = f"\n   📌 <i>Why it matters</i>: {why}" if why else ""
    summary_line = f"\n   📝 {summary}" if summary else ""
    validation_line = f"\n   <i>{reasoning}. Credibility: {credibility}/100</i>" if reasoning else ""

    return (
        f"{idx}️⃣ <b>{title}</b>{lang_tag}\n"
        f"   {STATUS_EMOJI.get(status, '⚠️')} <i>{STATUS_LABEL.get(status, 'Unverified')}</i> — {source_type}"
        f"{summary_line}"
        f"{matters_line}\n"
        f"   🔗 {link_text} | {source_type}"
        f"{validation_line}"
    )


def _build_digest_text(articles: List[Dict], stats: Dict) -> str:
    now = datetime.now(timezone.utc)
    hour = (now.hour + 5) % 24  # IST offset approx
    time_label = "9 AM" if hour < 12 else "6 PM"
    date_str = now.strftime("%b %d")

    verified_count = stats.get("verified_after_xref", stats.get("verified", 0))
    unverified_count = sum(1 for a in articles if a.get("validation_status") == "unverified")

    header = (
        f"<b>📊 Daily Digest — {date_str}, {time_label} IST</b>\n"
        f"<i>Found {len(articles)} updates ({verified_count} verified, {unverified_count} unverified)</i>\n"
    )

    # Group by category
    categories = {}
    for art in articles:
        cat = art.get("category", "tech")
        categories.setdefault(cat, []).append(art)

    sections = []
    idx = 1
    for cat, items in [("finance", categories.get("finance", [])),
                       ("tech", categories.get("tech", [])),
                       ("govt", categories.get("govt", []))]:
        if not items:
            continue
        emoji = CATEGORY_EMOJI.get(cat, "📰")
        section_header = f"\n━━━━━━━━━━━━━━━━━━━━\n{emoji} <b>{cat.upper()}</b> ({len(items)} updates)\n"
        section_items = "\n\n".join(_format_article(idx + i, items[i]) for i in range(len(items)))
        idx += len(items)
        sections.append(section_header + section_items)

    footer = (
        "\n\n━━━━━━━━━━━━━━━━━━━━\n"
        "<i>Reply <b>details &lt;number&gt;</b> for full article (e.g., details 1)</i>\n"
        "<i>Reply <b>feedback &lt;number&gt; &lt;text&gt;</b> to improve filtering</i>"
    )

    return header + "".join(sections) + footer


async def filter_and_build_digest(state: AgentState) -> Dict:
    """Filter validated articles to actionable ones and build digest."""
    validated = state.get("validated", [])

    # Filter: remove conflicting and non-actionable
    actionable = [
        a for a in validated
        if a.get("validation_status") != "conflicting" and a.get("is_actionable", False)
    ]

    # Sort by combined score: credibility + UPI/CC keyword boost, cap at MAX_DIGEST_ITEMS
    actionable.sort(
        key=lambda x: x.get("credibility_score", 0) + _cc_upi_boost(x),
        reverse=True,
    )
    actionable = actionable[:MAX_DIGEST_ITEMS]

    stats = {**state.get("stats", {}), "actionable": len(actionable)}

    if not actionable:
        digest = ""
        logger.info("No actionable items found for this run")
    else:
        digest = _build_digest_text(actionable, stats)
        logger.info(f"Built digest with {len(actionable)} actionable items")

    return {"actionable": actionable, "digest": digest, "stats": stats}


async def send_approval_request(state: AgentState) -> Dict:
    """Send preview message to Telegram asking user to approve."""
    if not TELEGRAM_CHAT_ID:
        logger.warning("TELEGRAM_CHAT_ID not set, skipping Telegram notification")
        return {"approval_status": "skipped"}

    actionable = state.get("actionable", [])
    stats = state.get("stats", {})

    if not actionable:
        await send_message(TELEGRAM_CHAT_ID, "✅ <b>No new actionable updates today.</b>")
        return {"approval_status": "skipped"}

    verified_count = stats.get("verified_after_xref", stats.get("verified", 0))
    unverified_count = sum(1 for a in actionable if a.get("validation_status") == "unverified")

    preview = (
        f"🔔 <b>News Digest Ready</b>\n\n"
        f"Found <b>{len(actionable)} actionable updates</b>:\n"
        f"  ✅ Verified: {verified_count}\n"
        f"  ⚠️ Unverified: {unverified_count}\n\n"
        f"Categories: "
        + ", ".join(
            f"{CATEGORY_EMOJI.get(c, '📰')} {c.upper()} ({sum(1 for a in actionable if a.get('category') == c)})"
            for c in ["finance", "tech", "govt"]
            if any(a.get("category") == c for a in actionable)
        )
        + f"\n\n<b>Reply YES to receive the full digest.</b>"
    )

    sent = await send_message(TELEGRAM_CHAT_ID, preview)
    if sent:
        logger.info(f"Approval request sent to chat {TELEGRAM_CHAT_ID}")
    return {"approval_status": "pending"}
