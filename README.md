# Prodigal AI Task 1 — Multi-Agent Social Media AI Company (local, Ollama)

Status: **Day 1 of 4 complete** — mock platform + engagement simulation engine +
local-model layer (LLM calls pending Ollama install on this machine). Agents
arrive on Day 2. Exact commands: **[SETUP_LOG.md](SETUP_LOG.md)**.

## Quick start (Day 1 scope)

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
cp .env.example .env

.venv/Scripts/python.exe -m platform.main        # mock platform -> http://127.0.0.1:8010/docs
.venv/Scripts/python.exe -m pytest tests/ -v     # 27 tests (simulation + structured output)
.venv/Scripts/python.exe scripts/test_model_layer.py   # needs Ollama running (see SETUP_LOG §6)
```

## What's here

| Area | Where | Notes |
|---|---|---|
| Mock platform API | `platform/` | FastAPI + async SQLite; publish/feed/metrics/comments/reply/tick/analytics |
| Engagement simulator | `platform/simulation/` | 6 hidden rules (toggleable) + negative-binomial baseline |
| Data contracts | `models/` | Pydantic v2; each model's docstring states producer → consumer |
| Model layer | `llm/` | Ollama client (streaming, token accounting), structured-output retry loop, JSONL trace |
| Tests | `tests/` | Statistical direction tests (p<0.05, N=200/group) + retry-logic tests |

## Ground truth: the 6 planted engagement rules

1. **Time-window boost** — evening peak (per channel, 1.4–1.8x); 02:00–06:00 → 0.5x
2. **Question CTA** — copy ending in `?` → ~2.5x comments
3. **Hashtag curve** — 3–5 tags optimal; 0 tags −20%; 8+ tags −30%
4. **Channel length penalty** — professional >150 words −30%; short-form >60 words −60%; discussion exempt
5. **Novelty decay** — same format 3+ days running on a channel → compounding 15%/day
6. **Save-rate → follower boost** — save-rate >5% amplifies follower gain (second-order, hard to spot)

These are the Analytics Agent's targets; `tests/test_simulation_engine.py` is
the citable ground-truth table (each rule isolated, N=200 per group).
