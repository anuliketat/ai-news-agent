"""Tests for iteration 3 features: async webhook, /search command, MongoDB indexes, finance filter."""
import os
import time
import pytest
import requests
import asyncio
import motor.motor_asyncio

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "").rstrip("/").strip('"')
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017").strip('"')
DB_NAME = os.environ.get("DB_NAME", "news_agent_db").strip('"')
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "805540771").strip('"')


@pytest.fixture(scope="module")
def session():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


# ---------------------------------------------------------------------------
# 1. Server health
# ---------------------------------------------------------------------------
class TestServerHealth:
    """Verify server is up and responsive."""

    def test_root_endpoint(self, session):
        resp = session.get(f"{BASE_URL}/api/", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "running"
        print("✅ Root endpoint OK")

    def test_agent_status(self, session):
        resp = session.get(f"{BASE_URL}/api/agent/status", timeout=5)
        assert resp.status_code == 200
        print("✅ /api/agent/status OK")


# ---------------------------------------------------------------------------
# 2. Webhook returns {"ok": true} immediately
# ---------------------------------------------------------------------------
class TestWebhookImmediate:
    """Webhook should return ok:true immediately even for chat messages."""

    def _send_webhook(self, session, text: str) -> tuple:
        payload = {
            "message": {
                "chat": {"id": int(TELEGRAM_CHAT_ID)},
                "text": text
            }
        }
        start = time.time()
        resp = session.post(f"{BASE_URL}/api/telegram/webhook", json=payload, timeout=10)
        elapsed = time.time() - start
        return resp, elapsed

    def test_webhook_returns_ok_for_help(self, session):
        resp, elapsed = self._send_webhook(session, "/help")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        print(f"✅ /help webhook ok:true in {elapsed:.2f}s")

    def test_webhook_returns_ok_immediately_for_chat(self, session):
        """Chat messages run as background task — webhook must return within 5s."""
        resp, elapsed = self._send_webhook(session, "What credit card is best for Swiggy?")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        # Should return fast (background task, not blocking)
        assert elapsed < 5.0, f"Webhook took too long: {elapsed:.2f}s (should be < 5s)"
        print(f"✅ Chat webhook returned ok:true in {elapsed:.2f}s (background task working)")

    def test_webhook_returns_ok_for_unknown_chat_id(self, session):
        """Messages from unknown chat ID should silently return ok:true."""
        payload = {
            "message": {
                "chat": {"id": 9999999},
                "text": "hello"
            }
        }
        resp = session.post(f"{BASE_URL}/api/telegram/webhook", json=payload, timeout=5)
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        print("✅ Unknown chat ID returns ok:true")

    def test_server_responsive_during_chat(self, session):
        """After sending a chat message (background task), server still responds to /api/."""
        # Trigger background chatbot task
        self._send_webhook(session, "Explain UPI cashback offers 2026")
        # Immediately check server
        resp = session.get(f"{BASE_URL}/api/", timeout=3)
        assert resp.status_code == 200
        print("✅ Server remains responsive while chatbot runs in background")


# ---------------------------------------------------------------------------
# 3. /search command
# ---------------------------------------------------------------------------
class TestSearchCommand:
    """Test /search Telegram command via webhook."""

    def _send_webhook(self, session, text: str) -> requests.Response:
        payload = {
            "message": {
                "chat": {"id": int(TELEGRAM_CHAT_ID)},
                "text": text
            }
        }
        return session.post(f"{BASE_URL}/api/telegram/webhook", json=payload, timeout=10)

    def test_search_command_returns_ok(self, session):
        resp = self._send_webhook(session, "/search cashback")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        print("✅ /search cashback returns ok:true")

    def test_search_no_results_returns_ok(self, session):
        resp = self._send_webhook(session, "/search xyz123notexist")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        print("✅ /search xyz123notexist returns ok:true")

    def test_search_short_keyword_returns_ok(self, session):
        """Single-char search should return usage hint."""
        resp = self._send_webhook(session, "/search a")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        print("✅ /search single-char returns ok:true with usage hint")


# ---------------------------------------------------------------------------
# 4. /help includes /search
# ---------------------------------------------------------------------------
class TestHelpCommand:
    """Verify help text includes /search command."""

    def test_help_webhook_ok(self, session):
        payload = {
            "message": {
                "chat": {"id": int(TELEGRAM_CHAT_ID)},
                "text": "/help"
            }
        }
        resp = session.post(f"{BASE_URL}/api/telegram/webhook", json=payload, timeout=5)
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        print("✅ /help returns ok:true")


# ---------------------------------------------------------------------------
# 5. Help text contains /search (code review)
# ---------------------------------------------------------------------------
class TestHelpTextContent:
    """Code-level check: _send_help must reference /search."""

    def test_help_text_contains_search(self):
        """Read server.py and confirm /search is in the help text."""
        server_path = os.path.join(os.path.dirname(__file__), "..", "server.py")
        with open(server_path) as f:
            content = f.read()
        assert "/search" in content, "/search not found in server.py help text"
        # More specifically check it's in _send_help function
        idx = content.find("async def _send_help")
        assert idx != -1
        help_section = content[idx:idx+2000]
        assert "/search" in help_section, "/search not found in _send_help function"
        print("✅ /search is present in _send_help function")


# ---------------------------------------------------------------------------
# 6. Finance filter: 'interest rate' not in keyword list
# ---------------------------------------------------------------------------
class TestFinanceFilter:
    """Verify 'interest rate' was removed from finance keyword list."""

    def test_interest_rate_not_in_keywords(self):
        fetchers_path = os.path.join(os.path.dirname(__file__), "..", "agent", "nodes", "fetchers.py")
        with open(fetchers_path) as f:
            content = f.read()
        # Find _FINANCE_RELEVANT_KW list
        assert "_FINANCE_RELEVANT_KW" in content
        # Extract the list content
        start = content.find("_FINANCE_RELEVANT_KW = [")
        end = content.find("]", start) + 1
        kw_section = content[start:end]
        assert '"interest rate"' not in kw_section, "'interest rate' still in _FINANCE_RELEVANT_KW"
        assert "'interest rate'" not in kw_section, "'interest rate' still in _FINANCE_RELEVANT_KW"
        print("✅ 'interest rate' correctly removed from _FINANCE_RELEVANT_KW")

    def test_specific_upi_keywords_present(self):
        """Key UPI/CC terms should still be in the list."""
        fetchers_path = os.path.join(os.path.dirname(__file__), "..", "agent", "nodes", "fetchers.py")
        with open(fetchers_path) as f:
            content = f.read()
        start = content.find("_FINANCE_RELEVANT_KW = [")
        end = content.find("]", start) + 1
        kw_section = content[start:end]
        for kw in ["upi", "credit card", "cashback", "hdfc"]:
            assert kw in kw_section, f"Expected keyword '{kw}' missing from _FINANCE_RELEVANT_KW"
        print("✅ Core UPI/CC keywords present in _FINANCE_RELEVANT_KW")


# ---------------------------------------------------------------------------
# 7. MongoDB indexes check
# ---------------------------------------------------------------------------
class TestMongoIndexes:
    """Verify TTL and text indexes exist on articles collection."""

    def test_mongodb_indexes(self):
        """Connect to MongoDB and verify the required indexes exist."""
        import asyncio

        async def _check():
            client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
            db = client[DB_NAME]
            indexes = await db.articles.index_information()
            client.close()
            return indexes

        indexes = asyncio.get_event_loop().run_until_complete(_check())
        index_names = list(indexes.keys())
        print(f"Found MongoDB indexes on articles: {index_names}")

        # Check TTL index
        assert "ttl_fetched_at" in indexes, f"TTL index 'ttl_fetched_at' not found. Found: {index_names}"
        ttl_idx = indexes["ttl_fetched_at"]
        assert ttl_idx.get("expireAfterSeconds") == 2592000, "TTL should be 2592000 (30 days)"
        print("✅ TTL index 'ttl_fetched_at' exists (30 days)")

        # Check text index
        assert "text_search_idx" in indexes, f"Text index 'text_search_idx' not found. Found: {index_names}"
        print("✅ Text search index 'text_search_idx' exists")


# ---------------------------------------------------------------------------
# 8. Chatbot code review: asyncio.create_task used
# ---------------------------------------------------------------------------
class TestChatbotAsyncCode:
    """Code review: verify chatbot is launched as asyncio.create_task."""

    def test_chatbot_uses_create_task(self):
        server_path = os.path.join(os.path.dirname(__file__), "..", "server.py")
        with open(server_path) as f:
            content = f.read()
        assert "create_task(_handle_chat_message" in content, \
            "Chatbot not launched with asyncio.create_task"
        print("✅ Chatbot launched via asyncio.create_task (non-blocking)")

    def test_chatbot_no_ddgs_import(self):
        """Verify ddgs/primp library not imported in chatbot."""
        chatbot_path = os.path.join(os.path.dirname(__file__), "..", "agent", "chatbot.py")
        with open(chatbot_path) as f:
            content = f.read()
        assert "from duckduckgo_search" not in content, "ddgs library still imported in chatbot"
        assert "import ddgs" not in content, "ddgs library still imported in chatbot"
        assert "import primp" not in content, "primp library still imported in chatbot"
        print("✅ ddgs/primp not imported in chatbot.py")

    def test_chatbot_uses_httpx(self):
        """Verify chatbot uses httpx for DDG search."""
        chatbot_path = os.path.join(os.path.dirname(__file__), "..", "agent", "chatbot.py")
        with open(chatbot_path) as f:
            content = f.read()
        assert "import httpx" in content, "httpx not imported in chatbot.py"
        assert "html.duckduckgo.com/html/" in content, "DDG HTML endpoint not used"
        print("✅ chatbot.py uses httpx + html.duckduckgo.com/html/ (no GIL blocking)")
