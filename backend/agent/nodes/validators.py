"""LLM-based article validation using HuggingFace Llama 3.3."""
import asyncio
import json
import logging
import os
import re
from typing import Dict, List

import httpx

from ..state import AgentState

logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

USER_PROFILE = (
    "30-year-old Data Scientist in Hyderabad (Madhapur), ₹2.3L/month salary. "
    "PRIMARY FOCUS: UPI cashback offers, credit card rewards/cashback (HDFC, ICICI, Axis, Amex, SBI, RuPay), "
    "new card offers, banking schemes, EMI deals for lifestyle spends (food delivery, travel, dining, entertainment). "
    "Also interested in: Data Science/MLE/Agentic AI/LLM career news, "
    "income tax saving, Telangana/Hyderabad govt schemes, travel policies. "
    "Goal: maximise cashback/rewards, find actionable card/UPI offers, grow AI/DS career."
)

VALIDATION_PROMPT = """\
Analyze this article for credibility, relevance, and generate a summary.

Title: {title}
Source domain: {source_domain}
Source type hint: {source_type}
Content snippet: {content}

User profile: {user_profile}

Return ONLY valid JSON (no prose, no markdown, no code fences):
{{
  "source_type": "official|news|community|research",
  "validation_status": "verified|unverified|conflicting",
  "credibility_score": <integer 0-100>,
  "reasoning": "<one concise sentence explaining credibility>",
  "is_actionable": <true|false>,
  "why_it_matters": "<one sentence tailored to this user's UPI/credit card/career interests, or null>",
  "needs_cross_reference": <true|false>,
  "summary": "<2-3 sentence plain-English summary of the key facts in this article>"
}}

Scoring guide:
- Official govt/bank/research (RBI, PIB, arxiv, huggingface, bank websites) → 90-100, verified
- Established news (ET, Mint, TechCrunch, The Hindu) → 70-85, unverified
- Community/aggregators → 40-65, unverified
- Unverified rumour/tweet → 10-35, unverified
Flag conflicting only if content directly contradicts a known fact.
Boost is_actionable=true for UPI offers, credit card cashback, card launches, reward program changes."""


# ---------------------------------------------------------------------------
# Rule-based fallback (used when LLM not configured or call fails)
# ---------------------------------------------------------------------------

_OFFICIAL = {"rbi.org.in", "pib.gov.in", "arxiv.org", "huggingface.co",
             "incometax.gov.in", "hdfcbank.com", "sbi.co.in", "icicibank.com",
             "axisbank.com", "indiabudget.gov.in", "telangana.gov.in",
             "github.com"}
_NEWS = {"economictimes.com", "moneycontrol.com", "livemint.com", "thehindu.com",
         "ndtv.com", "techcrunch.com", "venturebeat.com", "indianexpress.com",
         "bankbazaar.com", "cardinsider.com", "paisabazaar.com"}
_COMMUNITY = {"news.ycombinator.com", "reddit.com", "twitter.com", "x.com"}

# UPI / credit-card keywords for detecting high-relevance finance articles
_CC_UPI_KW = [
    "upi", "credit card", "cashback", "reward", "offer", "discount", "hdfc card",
    "icici card", "axis card", "amex", "sbi card", "rupay", "paytm", "amazon pay",
    "card launch", "card benefit", "milestone benefit", "lounge access", "emi offer",
    "zero fee", "surcharge waiver", "bonus points", "spend-based", "welcome bonus",
    "annual fee waiver", "gpay", "phonepe", "bhim", "credit limit",
]


def _extract_summary(content: str, max_sentences: int = 3) -> str:
    """Extract first N meaningful sentences as a plain-text summary."""
    if not content or len(content.strip()) < 50:
        return ""
    content = re.sub(r"\s+", " ", content).strip()
    sentences = re.split(r"(?<=[.!?])\s+", content)
    good = [s.strip() for s in sentences if len(s.strip()) > 30]
    return " ".join(good[:max_sentences])[:400]


def _is_cc_upi_relevant(article: Dict) -> bool:
    text = (article.get("title", "") + " " + article.get("content", "")).lower()
    return any(kw in text for kw in _CC_UPI_KW)


def _rule_based(article: Dict) -> Dict:
    domain = article.get("source_domain", "").lower()
    summary = _extract_summary(article.get("content", ""))
    # Boost why_it_matters for credit card / UPI articles
    if article.get("category") == "finance" and _is_cc_upi_relevant(article):
        why = "Relevant to your UPI/credit card rewards — check if actionable."
    else:
        why = None

    if any(d in domain for d in _OFFICIAL):
        return {**article,
                "validation_status": "verified", "credibility_score": 92,
                "reasoning": f"Official source: {domain}",
                "is_actionable": True,
                "why_it_matters": why or "Actionable update from official body.",
                "needs_cross_reference": False,
                "summary": summary}
    elif any(d in domain for d in _NEWS):
        return {**article,
                "validation_status": "unverified", "credibility_score": 72,
                "reasoning": f"Established news outlet: {domain}",
                "is_actionable": True,
                "why_it_matters": why or "Stay informed; verify before acting.",
                "needs_cross_reference": True,
                "summary": summary}
    elif any(d in domain for d in _COMMUNITY):
        return {**article,
                "validation_status": "unverified", "credibility_score": 45,
                "reasoning": f"Community source: {domain}",
                "is_actionable": False,
                "why_it_matters": None,
                "needs_cross_reference": True,
                "summary": summary}
    else:
        return {**article,
                "validation_status": "unverified", "credibility_score": 55,
                "reasoning": f"Unknown/aggregator source: {domain}",
                "is_actionable": True,
                "why_it_matters": why or "Verify from primary sources.",
                "needs_cross_reference": True,
                "summary": summary}


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Single-article async validator
# ---------------------------------------------------------------------------

async def _validate_one(
    article: Dict, client: httpx.AsyncClient, semaphore: asyncio.Semaphore
) -> Dict:
    if not HF_TOKEN:
        return _rule_based(article)

    async with semaphore:
        try:
            prompt = VALIDATION_PROMPT.format(
                title=article.get("title", ""),
                source_domain=article.get("source_domain", ""),
                source_type=article.get("source_type", "news"),
                content=article.get("content", "")[:400],
                user_profile=USER_PROFILE,
            )
            resp = await client.post(
                HF_API_URL,
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": HF_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a credibility analyser. "
                                "Always respond with valid JSON only, no other text."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 350,
                    "temperature": 0.1,
                },
                timeout=40.0,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = _extract_json(content)
                if parsed:
                    return {**article, **parsed}
                logger.warning(f"Could not parse LLM JSON for: {article.get('title', '')[:60]}")
        except Exception as e:
            logger.warning(f"LLM validation error for '{article.get('title', '')[:60]}': {e}")

    return _rule_based(article)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

async def validate_articles(state: AgentState) -> Dict:
    articles = state.get("deduplicated", [])
    if not articles:
        return {"validated": []}

    semaphore = asyncio.Semaphore(5)
    async with httpx.AsyncClient() as client:
        tasks = [_validate_one(a, client, semaphore) for a in articles]
        validated = await asyncio.gather(*tasks)

    verified = sum(1 for v in validated if v.get("validation_status") == "verified")
    unverified = sum(1 for v in validated if v.get("validation_status") == "unverified")
    conflicting = sum(1 for v in validated if v.get("validation_status") == "conflicting")

    logger.info(
        f"Validation complete: {verified} verified, {unverified} unverified, {conflicting} conflicting"
    )
    return {
        "validated": list(validated),
        "stats": {
            **state.get("stats", {}),
            "verified": verified,
            "unverified": unverified,
            "conflicting": conflicting,
        },
    }
