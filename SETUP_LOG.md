# SETUP_LOG — running record of exact commands

Purpose: the README writes itself from this file at the end. Every command
below was actually run on the build machine (Windows 11, Git Bash, Python
3.10.11) on the date shown. Copy-paste them in order on a clean clone.

---

## Day 1 — 2026-09-13 (foundations: platform + simulation + model layer)

### 0. Prerequisites

```bash
# Python 3.10+ (built and verified on 3.10.11)
python --version
```

> **Python version deviation (flagged):** the architecture doc targets 3.11+.
> The build machine only has 3.10.11 and 3.14.0; per decision, Day 1 builds on
> 3.10 with code kept 3.11-clean (`X | None` unions are fine at runtime under
> `from __future__ import annotations` and in Pydantic annotations).

### 1. Virtual environment + pinned dependencies

```bash
py -3.10 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip     # Windows path
# (Linux/macOS: python3 -m venv .venv && .venv/bin/pip install --upgrade pip)
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Verified versions installed: fastapi 0.115.6, uvicorn 0.32.1, sqlalchemy
2.0.36, aiosqlite 0.20.0, pydantic 2.10.4, pydantic-settings 2.7.0, httpx
0.28.1, numpy 1.26.4, scipy 1.11.4, typer 0.15.1, rich 13.9.4, pytest 8.3.4,
pytest-asyncio 0.25.0.

### 2. Environment file

```bash
cp .env.example .env
# defaults work as-is; edit OLLAMA_HOST / model tags only if yours differ
```

### 3. Start the mock platform (FastAPI + SQLite, port 8010)

```bash
.venv/Scripts/python.exe -m platform.main
# serves http://127.0.0.1:8010  — interactive docs at http://127.0.0.1:8010/docs
# port 8010 because 8000 is occupied by an unrelated dev server on this machine
```

On first boot it creates `platform.db` and seeds 3 channels
(`ch_shortform` QuickBites, `ch_discussion` DebateHall, `ch_professional`
LeadDesk) with distinct, engine-consumed character profiles.

### 4. Endpoint walk-through (second terminal)

```bash
BASE=http://127.0.0.1:8010

# health + channels
curl -s $BASE/health
curl -s $BASE/channels | python -m json.tool

# publish a post (evening slot on the short-form channel)
TODAY=$(date +%Y-%m-%d)
curl -s -X POST $BASE/posts -H "Content-Type: application/json" -d "{
  \"campaign_id\": \"camp_demo\",
  \"channel_id\": \"ch_shortform\",
  \"copy\": \"POV: your dorm just got an espresso upgrade and it cost less than your textbooks\",
  \"hashtags\": [\"#coffee\", \"#studentlife\", \"#budgetwins\"],
  \"cta\": \"Link in bio\",
  \"format\": \"video\",
  \"published_at\": \"${TODAY}T19:00:00Z\"
}"

# feed (filterable by channel / campaign)
curl -s "$BASE/feed?campaign_id=camp_demo"

# advance the simulated clock 24h -> engagement engine accrues the post
curl -s -X POST $BASE/simulate/tick -H "Content-Type: application/json" \
  -d '{"hours": 24, "campaign_id": "camp_demo"}'

# metrics + comments for the post (use the id POST returned)
PID=<post_id_from_publish_response>
curl -s $BASE/posts/$PID/metrics
curl -s $BASE/posts/$PID/comments

# Community Manager reply (Day 2 wires the agent; endpoint works today)
CID=$(curl -s $BASE/posts/$PID/comments | python -c "import sys,json;print(json.load(sys.stdin)['comments'][0]['id'])")
curl -s -X POST $BASE/posts/$PID/comments/$CID/reply \
  -H "Content-Type: application/json" \
  -d '{"reply_text": "Thanks for jumping in!"}'

# weekly aggregation (the Analytics Agent's raw data source)
curl -s $BASE/analytics/week/camp_demo/1
```

All ten routes verified live on Day 1: `/health`, `/channels`, `POST /posts`,
`GET /feed`, `GET /posts/{id}/metrics`, `GET /posts/{id}/comments`,
`POST /posts/{id}/comments/{id}/reply`, `POST /simulate/tick`,
`GET /analytics/week/{campaign_id}/{week}`, plus OpenAPI at `/docs`.

### 5. Tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
# 27 passed — 16 statistical/engine tests + 11 structured-output tests
```

### 6. Ollama + model layer (BLOCKED on Ollama install — see Day 1 summary)

```bash
# 1) Install Ollama: https://ollama.com/download  (Windows installer)
# 2) Verify daemon:
curl http://localhost:11434/api/tags
# 3) Pull the two models (7B ~4.7GB, 3B ~1.9GB):
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:3b-instruct
# 4) Run the live model-layer test (clean + corruption-injection + hostile cases):
.venv/Scripts/python.exe scripts/test_model_layer.py
# 5) Inspect the trace:
cat logs/agent_trace.jsonl | python -m json.tool --json-lines   # or just open it
```

Until Ollama is up, `scripts/test_model_layer.py` exits 2 with an actionable
message (verified on Day 1) — that IS the specified graceful-failure behavior.
Run it after installing; append the output below.

```
# (paste `python scripts/test_model_layer.py` output here after Ollama install)
```

### Day-1 notes for the write-up (captured live, not reconstructed)

- **stdlib shadow:** `import platform` inside SQLAlchemy resolves to OUR
  package from the repo root. Fix: `platform/__init__.py` re-exports stdlib
  names; `conftest.py` rebinds for pytest. Verified by a dedicated test.
- **tick semantics:** posts published before a tick's window accrue at
  publish-maturity with full exposure; posts published *inside* the window
  accrue with `exposure_fraction` (verified 0.674 live).
- **comment generation:** pools drawn via per-pool permutations so identical
  templates never repeat back-to-back.
