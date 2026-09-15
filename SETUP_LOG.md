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

---

## Day 2 — 2026-09-15 (agents, orchestration, reject loop end-to-end)

Built doc build-order steps 6-8: message bus, state machine, retry policy,
base agent, all 8 agents, and the compliance reject loop wired end to end.
Nothing publishes (Day 3).

### 1. Dependencies (one new pin)

```bash
.venv/Scripts/python.exe -m pip install jinja2==3.1.5
# (requirements.txt updated; a fresh venv gets it from requirements.txt)
```

### 2. Run the full test suite (Day 1 + Day 2 tests)

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
# Day-2 result: 63 passed in 8.5s (was 27 after Day 1; +36 new)
```

### 3. Run the Day-2 dry run (needs Ollama running)

```bash
# Ollama must be up (see Day 1 section 7 for install/pull commands):
curl http://localhost:11434/api/tags

# Full dry run with the interactive human gate:
.venv/Scripts/python.exe scripts/run_day2_dryrun.py

# Or headless (auto-approves at the human gate):
.venv/Scripts/python.exe scripts/run_day2_dryrun.py --auto-approve
```

What you should see: 7 stages printed (orchestrator -> strategy -> writer
3-pass -> creative -> compliance with 1 forced rejection + revision ->
human gate -> scheduler slots), then the per-campaign bus history summary.
The forced rejection: a banned phrase is injected into post 1's first
review; the rule prefilter rejects WITHOUT a model call; the Writer revises
with the verdict's reasons verbatim; re-review approves. If the injected
phrase ever survives, the script aborts itself (never ships a banned post).

When Ollama is down, the script exits 2 with install/start/pull/verify
instructions (verified — that is the specified graceful failure).

### 4. Inspect the agent conversation (the deliverable)

```bash
# The full conversation, chronologically interleaved:
cat logs/agent_trace.jsonl

# Just the bus handoffs:
grep '"event": "bus_message"' logs/agent_trace.jsonl | python -m json.tool --json-lines

# Just the rejection cycles (write-up gold):
grep -E '"message_type": "(compliance_verdict|posts_rejected)"' logs/agent_trace.jsonl
```

The bus ALSO persists every message to the `message_bus` table in
platform.db (survives restarts; replayable):

```bash
.venv/Scripts/python.exe -c "import asyncio; from orchestration.message_bus import MessageBus; print(asyncio.run(MessageBus().count()))"
```

### 5. Quick component spot-checks (all offline, no Ollama)

```bash
# State machine: legal walk + illegal transition error
.venv/Scripts/python.exe -m pytest tests/test_state_machine.py -v

# Bounded retry policy (the infinite-loop prevention)
.venv/Scripts/python.exe -m pytest tests/test_retry_policy.py -v

# Message bus durability (restart survival)
.venv/Scripts/python.exe -m pytest tests/test_message_bus.py -v

# All agents offline via scripted transports (no daemon needed)
.venv/Scripts/python.exe -m pytest tests/test_agents_offline.py -v
```

### Day-2 notes for the write-up (captured live, not reconstructed)

- **State machine fidelity fix:** the doc draws `content_drafted ->
  compliance_review` as an edge, with rejections returning from
  `compliance_review`. First draft had reject events fired from
  content_drafted; corrected to match the doc exactly, with
  `compliance_review` a re-entered ACTIVITY state.
- **Contract sync:** `models/campaign.py::CampaignStatus` expanded from the
  5-state Day-1 draft to the doc §8 lifecycle (10 states) — the state
  machine is the runtime source of truth; the Literal mirrors it.
- **Reject-loop accounting:** `Post`'s "pending with rejection history"
  state is now legal (validator relaxed) — a revised post re-enters review
  with `rejection_count` intact; that count is what the retry policy counts.
- **Hard-rule short-circuit:** a hard banned-term hit rejects with NO model
  call — "don't rely on the LLM alone for hard constraints" enforced both
  by priority and by saving the latency.
- **Two-stage analytics proven offline:** `kpi_performance` in the
  WeeklyReport is computed by Python; the model narrates and literally
  cannot write numbers into the KPI table (schema separation).

### Day 1+2 live verification — 2026-09-15 (Ollama installed, model layer proven live)

Ollama 0.34.0 installed via official installer; both models pulled:

```bash
ollama pull qwen2.5:3b-instruct   # 1.9 GB
ollama pull qwen2.5:7b-instruct   # 4.7 GB
ollama list                        # both must appear
curl http://localhost:11434/api/version   # {"version":"0.34.0"}
```

Two fixes made during live bring-up (both found BY the live test, kept):

1. `OLLAMA_TIMEOUT_SECONDS` (new setting, default 600): the client's old
   hardcoded 120s HTTP timeout died during the 7B's first-call weight load
   (measured: 56 s cold load + generation > 120 s on this CPU-only host).
   Config-driven now; `.env.example` documents it.
2. `llm/trace.py::log_llm_call` had a latent `NameError` (used `get_settings`
   without the function-local import its siblings have) — first live call
   surfaced it; fixed and covered by the full suite.

Verification runs (all green):

```bash
.venv/Scripts/python.exe -m pytest tests/ -q          # 63 passed
.venv/Scripts/python.exe scripts/test_model_layer.py  # 6/6 live cases OK
```

Model-layer live result summary:

- 7B + 3B: clean JSON -> OK attempt 1; corrupted first response -> retry with
  validation error appended -> OK attempt 2; hostile prose around JSON ->
  extracted OK attempt 1.
- Trace excerpt (logs/agent_trace.jsonl): `structured_output_retry ok=False
  attempt=1` then `llm_call p=194` (repair prompt) then `ok=True attempt=2`.

---

## Day 3 — 2026-09-15 (live publishing, tom week, real engagement, live analytics)

Build-order steps 9-11 plus closing Day-2 stubs: scheduler publishes to the
real platform, a simulated week accrues real engagement under hidden rules,
the Community Manager reads real comments and posts replies/ escalations, the
Analytics Agent produces a real weekly report from platform data (not
fixtures), and the strategy agent's memory retrieval is wired to a local
embedding store.

### 1. Run the full test suite (Day 1 + 2 + 3)

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
# Day-3 result: 88 passed in ~15s (was 63 after Day 2; +25 new)
```

New offline tests `tests/test_day3_wiring.py` cover: the shared RNG fix,
PlatformClient publish/tick/metrics/analytics via ASGITransport (no server),
MemoryStore store+retrieve round-trip, analytics key compat (`ends_in_question`
vs `copy_is_question`) and `kpi_vs_target`, scheduler day-spread, and the
escalation queue.

### 2. Start the platform (separate terminal)

```bash
.venv/Scripts/python.exe -m platform.main
# http://127.0.0.1:8010  (kept running for the whole Day-3 demo)
```

### 3. Run the full demo

```bash
# Requires Ollama up (Day 1 section 6). Full pipeline, auto-approve human gate:
.venv/Scripts/python.exe run_demo.py --brief seed_data\demo_brief.txt --auto-approve
```

What you should see (stages persistent across process restarts via
`logs/demo_checkpoint.json`, `--fresh` to restart):

1. orchestrator parses brief -> campaign id + routing plan
2. strategy fills audience / 3 pillars / KPIs
3. writer drafts -> self-critiques -> revises
4. creative briefs per post (asset type + description)
5. compliance: forced banned-phrase injection on post 1 -> reject -> revise ->
   approve loop
6. human gate: `--auto-approve` skips the y/n prompt
7. scheduler: 3 posts spread across the 7-day window at each channel's peak hour
8. publish: 3 POSTs to the live platform
9. simulate: ~33 six-hour ticks; each post accrues once in its window
10. community manager: real comment counts per post; replies/escalations via router
11. analytics: stage-1 stats (computed, logged as `analytics_computed_stats`)
    then stage-2 model narration; `logs/week1_campaign_<id>_report.json` written
12. memory: learnings embedded + stored (`memory_records`); week-2 retrieval wired
13. discovery scoring: agent findings table vs the 6 ground-truth rules
14. summary + trace + checkpoint cleared

Captured run transcript: `writeup/sample_run_transcript.md`.
Example scoring verdict from the live run: 2 of 6 rules found (honest sparse-week
result — see the transcript's "How to read the verdict").

### 4. Inspect outcomes

```bash
# Weekly report (the analytics deliverable):
cat logs/week1_campaign_*.json

# Agent conversation incl. analytics stages + bus handoffs:
cat logs/agent_trace.jsonl

# CLI viewers (read the platform DB directly; server may stay up):
.venv/Scripts/python.exe -m cli.viewer feed
.venv/Scripts/python.exe -m cli.viewer feed --campaign camp_9b7fb8c7
.venv/Scripts/python.exe -m cli.viewer trace camp_9b7fb8c7
.venv/Scripts/python.exe -m cli.viewer report camp_9b7fb8c7 1
.venv/Scripts/python.exe -m cli.viewer escalations --campaign camp_9b7fb8c7

# Discovery scoring script, standalone (auto-finds newest week*_report_*.json):
.venv/Scripts/python.exe scripts/score_analytics_discovery.py
```

### Day-3 notes for the write-up (captured live, not reconstructed)

- **Three landmines found by live runs + tests and fixed:**
  1. `platform/routes/metrics.py` missing `from math import sqrt` (NameError at
     weekly totals) — caught by the PlatformClient analytics roundtrip test.
  2. `compute_stats` key mismatch: platform emits `ends_in_question`, fixtures
     used `copy_is_question` — now both accepted.
  3. Engagement RNG was being re-seeded per accrual (all draws identical);
     replaced with a shared process-wide `get_rng()` advancing stream
     (seeded once from `SIM_SEED`), with per-tick seed override preserved for
     deterministic single-tick tests.
- **PlatformClient boundary:** agents now touch the platform ONLY through
  `platform/client.py` (httpx); unreachable platform raises
  `PlatformConnectionError` with a "run `python -m platform.main`" fix hint.
- **Two-stage analytics is real:** stage 1 `compute_stats` over live snapshots
  is logged as a DISTINCT `analytics_computed_stats` trace event before the
  model narrates; `kpi_vs_target` compares measured vs the campaign's own KPIs.
- **Fail-safe escalation observed:** with a 3B router in the loop, all 140
  comments escalated (router omitted -> "failing safe to human") — conservative
  by design; replies still proven offline in tests.
- **Checkpoint/resume proven live:** several stage bugs (jinja channel field,
  pillar-name drift, stale-resume guards) were fixed and the pipeline resumed
  mid-campaign instead of restarting — the resume path is the demoed behavior.
- **Memory is local-first:** char n-gram hashing -> 256-dim vector + cosine,
  zero external deps; Ollama `nomic-embed-text` documented as the upgrade path.
