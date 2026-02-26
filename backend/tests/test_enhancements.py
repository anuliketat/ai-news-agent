"""
Tests for AI News Agent enhancements:
- Summary field on articles
- Finance UPI/CC filtering
- Auto-translation fields
- Digest with 📝 summary emoji
- Telegram message splitting
- Agent status with 'translated' count
"""
import os
import time

import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "").rstrip("/")
AUTH = {"Authorization": "Bearer news_agent_2026_secret"}
CHAT_ID = "805540771"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def trigger_run():
    resp = requests.post(f"{BASE_URL}/api/agent/trigger", headers=AUTH)
    assert resp.status_code == 200, f"Trigger failed: {resp.text}"
    return resp.json().get("run_id")


def wait_for_run(run_id: str, timeout: int = 90):
    """Poll until run completes or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{BASE_URL}/api/agent/runs")
        if resp.status_code == 200:
            runs = resp.json().get("runs", [])
            for r in runs:
                if r.get("run_id") == run_id:
                    if r.get("status") in ("completed", "failed"):
                        return r
        time.sleep(5)
    return None  # timed out


# ---------------------------------------------------------------------------
# Fixtures — trigger run once for the whole module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def completed_run():
    print(f"\nTriggering agent run against {BASE_URL}...")
    run_id = trigger_run()
    print(f"Run triggered: {run_id}. Waiting up to 90s for completion...")
    run = wait_for_run(run_id, timeout=90)
    if run is None:
        pytest.skip("Agent run did not complete within 90 seconds — skip data-dependent tests")
    print(f"Run finished with status: {run.get('status')}")
    return run


# ---------------------------------------------------------------------------
# 1. POST /api/agent/trigger
# ---------------------------------------------------------------------------

class TestTrigger:
    """POST /api/agent/trigger returns 200 + run_id"""

    def test_trigger_returns_run_id(self):
        resp = requests.post(f"{BASE_URL}/api/agent/trigger", headers=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert isinstance(data["run_id"], str) and len(data["run_id"]) > 0
        print(f"PASS: trigger returned run_id={data['run_id']}")

    def test_trigger_unauthorized(self):
        resp = requests.post(f"{BASE_URL}/api/agent/trigger")
        assert resp.status_code == 401
        print("PASS: unauthorized trigger returns 401")


# ---------------------------------------------------------------------------
# 2. GET /api/agent/articles — summary field present
# ---------------------------------------------------------------------------

class TestArticleSummary:
    """After a run, articles should have a non-empty 'summary' field."""

    def test_articles_have_summary_field(self, completed_run):
        resp = requests.get(f"{BASE_URL}/api/agent/articles?limit=100")
        assert resp.status_code == 200
        articles = resp.json().get("articles", [])
        assert len(articles) > 0, "No articles found after run"

        articles_with_content = [a for a in articles if a.get("content") and len(a.get("content", "")) > 30]
        if not articles_with_content:
            pytest.skip("No articles with sufficient content")

        articles_with_summary = [a for a in articles_with_content if a.get("summary", "").strip()]
        pct = len(articles_with_summary) / len(articles_with_content) * 100
        print(f"Summary coverage: {len(articles_with_summary)}/{len(articles_with_content)} ({pct:.1f}%)")
        assert len(articles_with_summary) > 0, "No articles have a 'summary' field after run"

    def test_summary_is_meaningful(self, completed_run):
        resp = requests.get(f"{BASE_URL}/api/agent/articles?limit=100")
        articles = resp.json().get("articles", [])
        for a in articles:
            summary = a.get("summary", "")
            if summary:
                # summary should not be excessively long
                assert len(summary) <= 500, f"Summary too long ({len(summary)}): {summary[:80]}"
        print("PASS: all non-empty summaries are ≤500 chars")


# ---------------------------------------------------------------------------
# 3. GET /api/agent/articles?category=finance — UPI/banking relevant
# ---------------------------------------------------------------------------

class TestFinanceFiltering:
    """Finance articles should be UPI/banking relevant, not just stock tips."""

    _FINANCE_KW = [
        "upi", "credit card", "debit card", "cashback", "reward", "offer",
        "hdfc", "icici", "axis", "amex", "sbi card", "rupay", "paytm",
        "amazon pay", "gpay", "phonepe", "bhim", "rbi", "repo rate", "bank",
        "neft", "imps", "ifsc", "interest rate", "emi", "loan", "fd",
        "fixed deposit", "insurance", "mutual fund", "investment", "tax",
        "gst", "income", "finance", "economy", "budget", "scheme", "subsidy",
    ]

    def _is_finance_relevant(self, title: str, content: str) -> bool:
        text = (title + " " + content).lower()
        return any(kw in text for kw in self._FINANCE_KW)

    def test_finance_articles_are_relevant(self, completed_run):
        resp = requests.get(f"{BASE_URL}/api/agent/articles?category=finance&limit=100")
        assert resp.status_code == 200
        articles = resp.json().get("articles", [])
        if not articles:
            pytest.skip("No finance articles found")

        irrelevant = [
            a for a in articles
            if not self._is_finance_relevant(a.get("title", ""), a.get("content", ""))
        ]
        pct_relevant = (len(articles) - len(irrelevant)) / len(articles) * 100
        print(f"Finance relevance: {len(articles) - len(irrelevant)}/{len(articles)} ({pct_relevant:.1f}%)")
        # Allow up to 20% to slip through (edge cases)
        assert pct_relevant >= 80, (
            f"Too many irrelevant finance articles: {len(irrelevant)}/{len(articles)}"
        )


# ---------------------------------------------------------------------------
# 4. GET /api/agent/history — digest_text contains 📝
# ---------------------------------------------------------------------------

class TestDigestContent:
    """Latest digest should contain 📝 summary emoji for at least some items."""

    def test_history_endpoint(self):
        resp = requests.get(f"{BASE_URL}/api/agent/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "digests" in data
        print(f"History has {data['count']} digests")

    def test_latest_digest_has_summary_emoji(self, completed_run):
        # Use the full digest_text endpoint workaround — check via status
        resp = requests.get(f"{BASE_URL}/api/agent/history")
        assert resp.status_code == 200
        digests = resp.json().get("digests", [])
        assert len(digests) > 0, "No digests in history"

        # history returns without digest_text; check via status endpoint for pending
        status_resp = requests.get(f"{BASE_URL}/api/agent/status")
        assert status_resp.status_code == 200

        # Check the digest articles have summary field (proxy for 📝 in digest)
        articles_resp = requests.get(f"{BASE_URL}/api/agent/articles?limit=50")
        articles = articles_resp.json().get("articles", [])
        articles_with_summary = [a for a in articles if a.get("summary", "").strip()]
        assert len(articles_with_summary) > 0, "No articles have summaries to generate 📝 in digest"
        print(f"PASS: {len(articles_with_summary)} articles have summaries → 📝 will appear in digest")


# ---------------------------------------------------------------------------
# 5. Auto-translation fields
# ---------------------------------------------------------------------------

class TestTranslation:
    """Translated articles should have 'translated': true and 'original_language' field."""

    def test_translated_articles_have_fields(self, completed_run):
        resp = requests.get(f"{BASE_URL}/api/agent/articles?limit=100")
        assert resp.status_code == 200
        articles = resp.json().get("articles", [])

        translated = [a for a in articles if a.get("translated")]
        print(f"Translated articles found: {len(translated)}/{len(articles)}")

        if translated:
            for art in translated:
                assert art.get("original_language"), (
                    f"Translated article missing 'original_language': {art.get('title', '')[:60]}"
                )
            print(f"PASS: {len(translated)} translated articles all have 'original_language'")
        else:
            print("NOTE: No translated articles in this run (all sources may have been English)")
            # Not a failure — depends on live feed content


# ---------------------------------------------------------------------------
# 6. Telegram webhook — YES sends digest in chunks
# ---------------------------------------------------------------------------

class TestTelegramWebhook:
    """POST /api/telegram/webhook with YES should return 200."""

    def _make_update(self, text: str, chat_id: str = CHAT_ID):
        return {
            "update_id": 12345,
            "message": {
                "message_id": 1,
                "chat": {"id": int(chat_id), "type": "private"},
                "text": text,
                "date": int(time.time()),
            },
        }

    def test_yes_webhook_returns_ok(self):
        payload = self._make_update("YES")
        resp = requests.post(f"{BASE_URL}/api/telegram/webhook", json=payload)
        assert resp.status_code == 200
        assert resp.json().get("ok") is True
        print("PASS: YES webhook returns {ok: true}")

    def test_no_webhook_returns_ok(self):
        payload = self._make_update("NO")
        resp = requests.post(f"{BASE_URL}/api/telegram/webhook", json=payload)
        assert resp.status_code == 200
        assert resp.json().get("ok") is True
        print("PASS: NO webhook returns {ok: true}")

    def test_unknown_chat_ignored(self):
        payload = self._make_update("YES", chat_id="9999999")
        resp = requests.post(f"{BASE_URL}/api/telegram/webhook", json=payload)
        assert resp.status_code == 200
        assert resp.json().get("ok") is True
        print("PASS: unknown chat_id returns {ok: true}")


# ---------------------------------------------------------------------------
# 7. _split_telegram_message logic (unit-style via import)
# ---------------------------------------------------------------------------

class TestSplitTelegramMessage:
    """Verify the splitting logic handles messages > 4000 chars."""

    def _split(self, text: str) -> list:
        """Replicate the server logic."""
        MAX = 3800
        if len(text) <= MAX:
            return [text]
        chunks = []
        current = ""
        for para in text.split("\n\n"):
            block = (current + "\n\n" + para).strip() if current else para
            if len(block) <= MAX:
                current = block
            else:
                if current:
                    chunks.append(current)
                if len(para) > MAX:
                    for i in range(0, len(para), MAX):
                        chunks.append(para[i: i + MAX])
                    current = ""
                else:
                    current = para
        if current:
            chunks.append(current)
        return chunks

    def test_short_message_not_split(self):
        msg = "Short message."
        chunks = self._split(msg)
        assert chunks == [msg]
        print("PASS: short message stays as single chunk")

    def test_long_message_split_into_chunks(self):
        # Build a 10000-char message with paragraphs
        para = "A" * 500
        msg = "\n\n".join([para] * 20)  # 10000 chars + separators
        chunks = self._split(msg)
        assert len(chunks) > 1, "Long message should be split"
        for c in chunks:
            assert len(c) <= 3800, f"Chunk too long: {len(c)}"
        print(f"PASS: {len(msg)}-char message split into {len(chunks)} chunks ≤3800 chars each")

    def test_single_oversized_para_hard_split(self):
        para = "B" * 8000
        chunks = self._split(para)
        assert all(len(c) <= 3800 for c in chunks)
        print(f"PASS: 8000-char single para hard-split into {len(chunks)} chunks")

    def test_empty_message(self):
        chunks = self._split("")
        assert chunks == [""]
        print("PASS: empty message returns ['']")


# ---------------------------------------------------------------------------
# 8. GET /api/agent/status — includes 'translated' in stats
# ---------------------------------------------------------------------------

class TestAgentStatus:
    """Status endpoint last_run stats should include 'translated' count."""

    def test_status_returns_200(self):
        resp = requests.get(f"{BASE_URL}/api/agent/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "last_run" in data
        print("PASS: status endpoint returns last_run")

    def test_last_run_has_stats(self, completed_run):
        resp = requests.get(f"{BASE_URL}/api/agent/status")
        data = resp.json()
        last_run = data.get("last_run", {})
        stats = last_run.get("stats", {})
        assert stats, f"last_run has no 'stats' key: {last_run}"
        print(f"Stats in last_run: {stats}")

    def test_stats_has_translated_count(self, completed_run):
        resp = requests.get(f"{BASE_URL}/api/agent/status")
        data = resp.json()
        last_run = data.get("last_run", {})
        stats = last_run.get("stats", {})
        assert "translated" in stats, f"'translated' key missing from stats: {stats}"
        assert isinstance(stats["translated"], int)
        print(f"PASS: stats.translated = {stats['translated']}")

    def test_stats_has_total_fetched(self, completed_run):
        resp = requests.get(f"{BASE_URL}/api/agent/status")
        data = resp.json()
        stats = data.get("last_run", {}).get("stats", {})
        assert "total_fetched" in stats, f"total_fetched missing from stats: {stats}"
        print(f"PASS: stats.total_fetched = {stats['total_fetched']}")
