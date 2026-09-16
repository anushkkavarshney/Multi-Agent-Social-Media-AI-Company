# Prodigal AI Task 1 — Multi-Agent Social Media AI Company (local, Ollama)

Status: **All 4 days complete** — mock platform + engagement simulation engine +
local-model agent layer (Ollama): orchestration, strategy, writer, creative,
compliance, scheduler, community manager, analytics, and the two-week memory
loop. A literal cold run (fresh unpack, README-only) was timed and logged —
see **[writeup/final_audit_report.md](writeup/final_audit_report.md)** and
**[writeup/week1_vs_week2_results.md](writeup/week1_vs_week2_results.md)**.
Exact setup commands: **[SETUP_LOG.md](SETUP_LOG.md)**.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
cp .env.example .env

.venv/Scripts/python.exe -m platform.main        # mock platform -> http://127.0.0.1:8010/docs
.venv/Scripts/python.exe -m pytest tests/ -v     # 91 tests (engine + agents + retries + wiring)
```

Requires Ollama running with both models pulled, as in the prior day-1 task:

```bash
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:3b-instruct
```

Run the two-week loop-closure demo (the headline Day-4 artifact):

```bash
.venv/Scripts/python.exe scripts/run_two_weeks.py --reject-once   # full
.venv/Scripts/python.exe scripts/run_two_weeks.py --reject-once --short  # skips writer self-critique
```

Week-1 and/or Week-2 reports land in `logs/week{1,2}_campaign_*_report.json`;
learnings go to the `memory_records` table and drive the Week-2 strategy via
cosine-similarity retrieval.

## What's here

| Area | Where | Notes |
|---|---|---|
| Mock platform API | `platform/` | FastAPI + async SQLite; publish/feed/metrics/comments/reply/tick/analytics |
| Engagement simulator | `platform/simulation/` | 6 hidden rules (toggleable) + negative-binomial baseline |
| Data contracts | `models/` | Pydantic v2; each model's docstring states producer → consumer |
| Model layer | `llm/` | Ollama client (streaming, token accounting), structured-output retry loop, JSONL trace |
| Agents | `agents/` | orchestrator, strategy, writer, creative, compliance, scheduler, community manager, analytics |
| Orchestration | `orchestration/` | state machine + persisted agent message bus (SQLite, replayable) |
| Memory | `memory/` | week-report embeddings → `memory_records`; retrieved by week-2 strategy |
| Tests | `tests/` | 91 tests: statistical direction tests (p<0.05) + agent/retry/wiring tests |

## Ground truth: the 6 planted engagement rules

1. **Time-window boost** — evening peak (per channel, 1.4–1.8x); 02:00–06:00 → 0.5x
2. **Question CTA** — copy ending in `?` → 2–3x comments, **channel-modulated** (discussion 3.0x, short-form 2.5x, professional 2.0x)
3. **Hashtag curve** — 3–5 tags optimal; 0 tags −20%; 8+ tags −30%
4. **Channel length penalty** — professional >150 words −30%; short-form >60 words −60%; discussion exempt
5. **Novelty decay** — same format 3+ days running on a channel → compounding 15%/day
6. **Save-rate → follower boost** — save-rate >5% amplifies follower gain (second-order, hard to spot)

These are the Analytics Agent's targets; `tests/test_simulation_engine.py` is
the citable ground-truth table (each rule isolated, N=200 per group).

## Honesty artifacts

- `writeup/sample_run_transcript.md` — a full Week-1 run, stages 1–13, pasted
  verbatim from stdin, including the 7B model failure + recovery.
- `writeup/week1_vs_week2_results.md` — the two-week report comparison: which
  week-1 findings survived week-2 projection, and which week-1 hypothesis was
  falsified by week-2 data.
- `writeup/final_audit_report.md` — the required audit: the 4 differentiators
  (DP variants, research, OPaaS, growth loop), each with a provenance-verified
  verdict and the real evidence behind it.