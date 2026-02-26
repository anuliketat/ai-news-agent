"""Backend tests for AI News Monitoring Agent API."""
import os
import time
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "").rstrip("/")
SECRET = "news_agent_2026_secret"
CHAT_ID = 805540771


@pytest.fixture(scope="module")
def client():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


@pytest.fixture(scope="module")
def auth_client():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json", "Authorization": f"Bearer {SECRET}"})
    return s


# --- Root ---
class TestRoot:
    def test_root_returns_service_info(self, client):
        r = client.get(f"{BASE_URL}/api/")
        assert r.status_code == 200
        data = r.json()
        assert data["service"] == "AI News Monitoring Agent"
        assert data["status"] == "running"
        assert "endpoints" in data
        print("PASS: root endpoint returns service info")


# --- Auth ---
class TestAuth:
    def test_trigger_without_auth_returns_401(self, client):
        r = client.post(f"{BASE_URL}/api/agent/trigger")
        assert r.status_code == 401
        print("PASS: trigger without auth returns 401")

    def test_trigger_with_wrong_token_returns_401(self, client):
        r = client.post(f"{BASE_URL}/api/agent/trigger", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401
        print("PASS: trigger with wrong token returns 401")


# --- Status ---
class TestStatus:
    def test_status_returns_valid_structure(self, client):
        r = client.get(f"{BASE_URL}/api/agent/status")
        assert r.status_code == 200
        data = r.json()
        assert "last_run" in data
        assert "pending_digest" in data
        print(f"PASS: status - last_run={data['last_run'] is not None}, pending={data['pending_digest'] is not None}")


# --- Runs ---
class TestRuns:
    def test_runs_returns_list(self, client):
        r = client.get(f"{BASE_URL}/api/agent/runs")
        assert r.status_code == 200
        data = r.json()
        assert "runs" in data
        assert "count" in data
        assert isinstance(data["runs"], list)
        print(f"PASS: runs returns {data['count']} runs")

    def test_runs_have_expected_fields(self, client):
        r = client.get(f"{BASE_URL}/api/agent/runs")
        data = r.json()
        if data["count"] > 0:
            run = data["runs"][0]
            assert "run_id" in run
            assert "status" in run
            assert "started_at" in run
            print(f"PASS: run fields valid - run_id={run['run_id']}, status={run['status']}")
        else:
            pytest.skip("No runs yet to validate fields")


# --- History ---
class TestHistory:
    def test_history_returns_digests(self, client):
        r = client.get(f"{BASE_URL}/api/agent/history")
        assert r.status_code == 200
        data = r.json()
        assert "digests" in data
        assert "count" in data
        print(f"PASS: history returns {data['count']} digests")


# --- Articles ---
class TestArticles:
    def test_articles_returns_list(self, client):
        r = client.get(f"{BASE_URL}/api/agent/articles")
        assert r.status_code == 200
        data = r.json()
        assert "articles" in data
        assert "count" in data
        print(f"PASS: articles returns {data['count']} articles")

    def test_articles_count_greater_than_zero(self, client):
        r = client.get(f"{BASE_URL}/api/agent/articles?limit=100")
        data = r.json()
        assert data["count"] > 0, "Expected articles in DB after previous run"
        print(f"PASS: articles count = {data['count']} > 0")

    def test_articles_have_required_fields(self, client):
        r = client.get(f"{BASE_URL}/api/agent/articles?limit=5")
        data = r.json()
        if data["count"] > 0:
            art = data["articles"][0]
            assert "title" in art
            assert "url" in art
            print(f"PASS: article fields ok - title={art['title'][:50]}")
        else:
            pytest.skip("No articles to validate")

    def test_articles_category_filter(self, client):
        r = client.get(f"{BASE_URL}/api/agent/articles?category=finance")
        assert r.status_code == 200
        data = r.json()
        print(f"PASS: category filter returns {data['count']} finance articles")


# --- Trigger (with auth) ---
class TestTrigger:
    def test_trigger_with_auth_returns_run_id(self, auth_client):
        r = auth_client.post(f"{BASE_URL}/api/agent/trigger")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "triggered"
        assert "run_id" in data
        assert len(data["run_id"]) > 0
        TestTrigger.run_id = data["run_id"]
        print(f"PASS: trigger returns run_id={data['run_id']}")

    def test_run_id_appears_in_runs_list(self, client):
        if not hasattr(TestTrigger, "run_id"):
            pytest.skip("No run_id from trigger test")
        # Wait a moment for DB insert
        time.sleep(2)
        r = client.get(f"{BASE_URL}/api/agent/runs")
        data = r.json()
        run_ids = [run["run_id"] for run in data["runs"]]
        assert TestTrigger.run_id in run_ids
        print(f"PASS: triggered run_id appears in runs list")


# --- Telegram Webhook ---
class TestTelegramWebhook:
    def _post(self, client, text):
        return client.post(f"{BASE_URL}/api/telegram/webhook", json={
            "message": {"chat": {"id": CHAT_ID}, "text": text}
        })

    def test_webhook_returns_ok(self, client):
        r = self._post(client, "/help")
        assert r.status_code == 200
        data = r.json()
        assert data.get("ok") is True
        print("PASS: webhook /help returns ok=True")

    def test_webhook_no_message_returns_ok(self, client):
        r = client.post(f"{BASE_URL}/api/telegram/webhook", json={})
        assert r.status_code == 200
        assert r.json().get("ok") is True
        print("PASS: webhook with empty body returns ok=True")

    def test_webhook_yes_returns_ok(self, client):
        r = self._post(client, "YES")
        assert r.status_code == 200
        assert r.json().get("ok") is True
        print("PASS: webhook YES returns ok=True")

    def test_webhook_no_returns_ok(self, client):
        r = self._post(client, "NO")
        assert r.status_code == 200
        assert r.json().get("ok") is True
        print("PASS: webhook NO returns ok=True")

    def test_webhook_details_returns_ok(self, client):
        r = self._post(client, "details 1")
        assert r.status_code == 200
        assert r.json().get("ok") is True
        print("PASS: webhook 'details 1' returns ok=True")

    def test_webhook_wrong_chat_id_returns_ok(self, client):
        r = client.post(f"{BASE_URL}/api/telegram/webhook", json={
            "message": {"chat": {"id": 9999999}, "text": "YES"}
        })
        assert r.status_code == 200
        assert r.json().get("ok") is True
        print("PASS: webhook wrong chat_id silently returns ok=True")

    def test_webhook_feedback_returns_ok(self, client):
        r = self._post(client, "feedback 1 too generic")
        assert r.status_code == 200
        assert r.json().get("ok") is True
        print("PASS: webhook feedback returns ok=True")


# --- Agent Stats Validation ---
class TestAgentStats:
    def test_runs_have_stats_after_previous_run(self, client):
        r = client.get(f"{BASE_URL}/api/agent/runs")
        data = r.json()
        if data["count"] == 0:
            pytest.skip("No completed runs yet")
        # Find a completed run
        completed = [run for run in data["runs"] if run.get("status") == "completed"]
        if not completed:
            print(f"INFO: No completed runs yet (statuses: {[r['status'] for r in data['runs']]})")
            pytest.skip("No completed runs yet")
        run = completed[0]
        stats = run.get("stats", {})
        print(f"Run stats: {stats}")
        assert stats.get("total_fetched", 0) > 0, "Expected total_fetched > 0"
        print(f"PASS: total_fetched={stats['total_fetched']}, after_dedup={stats.get('after_dedup')}, actionable={stats.get('actionable')}")
