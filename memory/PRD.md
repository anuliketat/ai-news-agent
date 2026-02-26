# AI News Monitoring Agent — PRD

## Original Problem Statement
Build an async LangGraph-based AI agent that monitors, validates, and sends credible personalized daily updates on Finance (UPI/credit card/banking), Tech Career (Data Science, MLE, AIOps, Agentic AI), and Government (Tax, subsidies, Hyderabad/Telangana schemes).

## User Persona
- Age 30, married, Hyderabad (Madhapur)
- Role: Data Scientist, salary ₹2.3L/month
- Lifestyle: food delivery, travel, dining, entertainment
- Goal: UPI cashback, credit card rewards optimisation, career growth in AI/DS, cost saving

## Architecture
- **Framework**: LangGraph 1.0.9 (async StateGraph)
- **LLM**: HuggingFace Llama 3.3-70B-Instruct (rule-based fallback when HF_TOKEN not set)
- **Database**: MongoDB (motor async driver)
- **Notifications**: Telegram Bot (webhook-based)
- **Scheduling**: GitHub Actions (9 AM + 6 PM IST via cron)
- **Translation**: deep-translator + langdetect (Google Translate, auto-detects non-English)
- **Cross-reference**: Google News RSS (free, no API key)

## LangGraph Pipeline
```
START
  → fetch_all_sources (async parallel: 14 RSS/API sources)
  → deduplicate (MongoDB last-7-days URL check)
  → validate_articles (HuggingFace Llama or rule-based, max 5 concurrent)
  → cross_reference_check (Google News RSS for unverified items)
  → filter_and_build_digest (UPI/CC priority boost, max 15 items)
  → send_approval_request (Telegram preview)
  → save_results (MongoDB)
END
```

## Data Sources (14 total)
- **Finance**: ET, ET Wealth, Moneycontrol, LiveMint, NDTV Profit, BankBazaar
- **Tech**: ArXiv CS.AI, HuggingFace Blog, TechCrunch, VentureBeat, HackerNews
- **Govt**: PIB India, The Hindu National, Indian Express

## Key Features Implemented

### Core Agent
- [x] LangGraph StateGraph with async nodes
- [x] Parallel source fetching (asyncio.gather, 14 sources)
- [x] MongoDB deduplication (last 7 days by URL)
- [x] LLM validation with fallback (credibility score, verification status)
- [x] Cross-reference via Google News RSS
- [x] Telegram webhook approval flow (YES/NO/details N/feedback N text)
- [x] GitHub Actions workflow (9 AM + 6 PM IST)

### Credibility Validation
- [x] Source type classification (official/news/community/research)
- [x] Validation status (verified/unverified/conflicting)
- [x] Credibility score (0-100)
- [x] Reasoning (1 sentence)
- [x] Cross-reference count

### User Enhancement Features (Feb 26, 2026)
- [x] **UPI/Credit Card Focus**: Strict keyword filtering for finance articles, only keeping banking/payment relevant news. 24 irrelevant articles filtered per run.
- [x] **Article Summaries**: LLM generates 2-3 sentence summary (rule-based extraction fallback). Shown as 📝 in digest.
- [x] **Auto-Translation**: langdetect + deep_translator; non-English articles (Hindi/Marathi/Telugu PIB articles etc.) auto-translated. [Auto-translated] tag shown in digest.
- [x] **UPI/CC Priority Boost**: Credit card/UPI articles sorted higher in digest (up to +30 score boost)
- [x] **Message Splitting**: Digest split at paragraph boundaries ≤3800 chars (handles Telegram 4096 limit)
- [x] **/refresh command**: Triggers on-demand agent run from Telegram; shows "🔄 Refreshing..." ack, guards against duplicate concurrent runs
- [x] **/status command**: Shows last run stats (fetched/verified/actionable/translated), pending digest status, total DB articles
- [x] **Bot command menu**: /refresh, /status, /help registered in Telegram's native command menu
- [x] **Error notifications**: Failed refresh runs notify user via Telegram

## API Endpoints
- `POST /api/agent/trigger` — Trigger agent run (requires Bearer auth)
- `GET /api/agent/status` — Last run status + pending digest
- `GET /api/agent/history` — Past 20 digests
- `GET /api/agent/runs` — Past 20 agent runs
- `GET /api/agent/articles?limit=N&category=X` — Fetched articles
- `POST /api/telegram/webhook` — Telegram webhook handler

## GitHub Actions Setup (for scheduling)
1. Add GitHub Secrets:
   - `AGENT_URL` = `https://credible-news-agent.preview.emergentagent.com`
   - `AGENT_SECRET_KEY` = `news_agent_2026_secret`
2. Workflow in `.github/workflows/scraper.yml` already configured

## Environment Variables
```
TELEGRAM_BOT_TOKEN=<your_token>
TELEGRAM_CHAT_ID=<your_chat_id>
HF_TOKEN=hf_YOUR_TOKEN_HERE   # Active — Llama 3.3-70B via router.huggingface.co
BACKEND_URL=https://credible-news-agent.preview.emergentagent.com
AGENT_SECRET_KEY=news_agent_2026_secret
MONGO_URL=mongodb://localhost:27017
DB_NAME=news_agent_db
```

## P0/P1/P2 Backlog

### P0 (Critical - needed for production)
- [x] ~~Add HF_TOKEN to enable real LLM validation~~ — DONE (router.huggingface.co)
- [ ] Test GitHub Actions scheduling with `AGENT_URL` and `AGENT_SECRET_KEY` secrets

### P1 (High value)
- [ ] Add Reddit async scraping (asyncpraw) for r/IndiaInvestments, r/CreditCardsIndia, r/MachineLearning
- [ ] Add RBI official RSS feed when URL is confirmed working
- [ ] Add CardInsider.com scraping for credit card offers
- [ ] LLM-generated summaries once HF_TOKEN is set (currently rule-based)
- [ ] User feedback loop: store feedback → use in future scoring
- [ ] /stats Telegram command: show past week's digest stats

### P2 (Nice to have)
- [ ] Playwright-based scraping for JS-rendered pages (bank websites)
- [ ] Conflicting info alert: immediate Telegram notification for conflicting reports
- [ ] Preference model trained on user feedback
- [ ] Export digest as PDF or email summary
- [ ] Per-source credibility tracking (permanently low-credibility sources blacklisted)

## Performance (measured Feb 26, 2026)
- Scraping: ~3 seconds for 14 sources (async parallel)
- Translation: ~3-5 seconds for 10-15 non-English articles
- Validation (rule-based): <1 second for 50 articles
- Cross-reference: ~4-6 seconds for 40 articles
- **Total pipeline: ~10-12 seconds** (well under 2-minute target)
