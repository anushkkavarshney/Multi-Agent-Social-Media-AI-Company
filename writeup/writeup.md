# Prodigal AI Task 1 — Write-Up

Multi-agent social media AI company, fully local on one Windows machine
(no hosted APIs), Ollama only. Four days of work: platform → core loop →
publish/analyze → loop closure. This document covers what the brief requires:
architecture, model choices, prompting techniques, hidden-rule documentation,
a sample run, and an honest failure-mode section.

The Day-1-to-Day-3 engineering log is **[SETUP_LOG.md](../SETUP_LOG.md)**.
A verbatim Day-3 run transcript is **[sample_run_transcript.md](sample_run_transcript.md)**.
The Day-4 before/after two-week comparison is **[week1_vs_week2_results.md](week1_vs_week2_results.md)**.
The required audit of the four main deliverables is **[final_audit_report.md](final_audit_report.md)**.

---

## 1. Architecture

```
                                                         ┌────────────────────┐
 brief ─► OrchestratorAgent ──► StrategyAgent ──► WriterAgent ──► CreativeAgent
   ▲                              │    ▲                        │
   │                              │    └── memory_retrieved ──► MemoryStore ──► WeeklyReport(week n)
   │                              ▼                        │
   │            ┌─── ComplianceAgent ◄── reject/revise loop │
   │            ▼                                         ▼
   │        human gate ──► SchedulerAgent ──► Mock Platform API
   │                                             │  publish / tick / comments
   │                                             ▼
   └──────────────────────────── AnalyticsAgent (week report) ◄─ CommunityManagerAgent
```

- **Orchestration:** a persisted **message bus** (SQLite `message_bus` table,
  append-only, ordered by seq) + a **state machine**
  (`draft → strategy_filled → content_drafted → compliance_review →
  pending_human_approval → scheduled → published → simulating → week_reviewed
  → strategy_filled (week 2)`). Both live in `orchestration/`. Routing
  decisions are first-class bus records, so the whole campaign conversation is
  machine-replayable.
- **Reject loop as a real state machine:** compliance rejections are events
  (`compliance_rejected`), each sends the post back to Writer for revision,
  bounded by `max_compliance_retries=3` (config), then auto-escalates to
  `needs_human_review`. A hard banned-term hit rejects with **no model call**
  (rule prefilter) — see failure modes.
- **Mocks are honest:** the platform simulates a week of engagement via 6
  planted ground-truth rules (documented below) + a negative-binomial noise
  baseline; the Analytics Agent never sees the rules, only API data.
- **Memory:** each WeeklyReport's patterns + recommendations are embedded
  (local char n-gram → 256-dim) and stored in `memory_records`. Week-2
  Strategy retrieves top-3 learnings by cosine similarity
  (`memory_retrieved` trace event) and must visibly incorporate ≥1 learning
  into a pillar or KPI.

## 2. Model choices and the split

| Role | Model | Why |
|---|---|---|
| Strategy / Writer / Creative / Compliance / Analytics narrative | `qwen2.5:7b-instruct` | the reasoning+generation work; measured 330s mean/call on this CPU-only host |
| Router / utterance classification (community manager) | `qwen2.5:3b-instruct` | cheap per-call classifier; 177s mean; a 7× lower-cost tier for high-volume routing |
| Embeddings (memory retrieval) | `nomic-embed-text` | local, no API |

All text outputs are forced through **JSON-schema grammar** (Ollama `format`)
and a **Pydantic structured-output layer** with a bounded retry loop
(`max_structured_retries=3`), so every agent consumes typed models, never raw
text. HTTP timeout default 1800s because grammar lookahead on a slow CPU can
exceed 10 minutes for a long stats block.

Quantified tradeoff (from the run trace): 41 × 7B calls ≈ 225 min total,
7 × 3B calls ≈ 21 min total. The 3B router is ~1.9× faster per call at ~1/6
the weight, and routes >140 comments per week; the 7B tier is reserved for
anything that produces artifacts (drafts, verdicts, reports).

## 3. Prompting techniques

- All prompts are **Jinja2 templates** (`llm/prompts/*.jinja`) with the
  data contract in the prompt (schema, field types, constraints) so the model
  and the validator agree.
- Strategy is given **the objective, channel profiles (incl. their content
  rules), and optionally memory learnings**; a hard prompt constraint forces
  any retrieved learning to be visibly used in ≥1 pillar/KPI.
- Analytics gets **precomputed statistics as numbers, never raw tables of
  dozens of rows** — the compute step is plain Python (`analytics_computed_stats`
  trace event) *before* the model narrates, so the narrative cannot invent
  figures that aren't in the stats. This is the write-up's proof of the
  "value the analysis loop" requirement.
- The writer gets the *rotation* (which pillar maps to which post) and is
  re-prompted with rejection reasons on each compliance cycle.

## 4. Hidden-rule documentation (the 6 planted rules)

Ground truth, documented in `README.md` and unit-tested in
`tests/test_simulation_engine.py` (each rule isolated, N≥200/group, p<0.05):

1. **Time-window boost** — evening peak per channel (1.4–1.8×); 02:00–06:00 → ×0.5
2. **Question CTA** — copy ending in `?` → 2–3× comments, **channel-modulated**
   (discussion 3.0× > short-form 2.5× > professional 2.0×)
3. **Hashtag curve** — 3–5 tags optimal; 0 tags −20%; 8+ tags −30%
4. **Channel length penalty** — professional >150 words −30%; short-form >60
   words −60%; discussion exempt
5. **Novelty decay** — same format ≥3 days running → compounding 15%/day reach decay
6. **Save-rate → follower boost** — save-rate >5% amplifies follower gain
   (second-order, intentionally hard to detect)

Discovery scoring methodology and a reconciled honest hit-rate are in
**[week1_vs_week2_results.md](week1_vs_week2_results.md)**.

## 5. Sample run

Full Day-3 transcript: **[sample_run_transcript.md](sample_run_transcript.md)**.
Day-4 two-week loop closure (the "strongest possible submission" path):
**[week1_vs_week2_results.md](week1_vs_week2_results.md)**.

## 6. Failure modes (real, not generic)

From the live trace (`logs/agent_trace.jsonl`):

- **`PostDraftList` schema collapse.** On 2026-09-15 the 7B writer emitted
  `{"posts": {"0": {...}}}` (a dict keyed by index) instead of a `list[PostDraft]`.
  12 pydantic validation errors → **3 timed retries** (10:18:48, 10:21:08,
  10:24:38, ~2.5 min apart) → `agent_model_failure` event, and an
  `agent_failed` bus message to the Orchestrator with the full validation
  detail. The run recovered after a later reproduce. The structured-output
  layer never silently accepts malformed output; it fails loudly with the
  schema + last error in the trace.
- **Banned-term prefilter.** Compliance rejects hard terms (e.g. "100%
  guaranteed") with **no model call** — a deterministic prefilter beats a
  model for the call where the model is worst. Emitted `posts_rejected` bus
  messages show the reason each cycle.
- **Router fail-safe.** The community-manager 3B router makes no decisions for
  ambiguous comments, which is safer than guessing: 0 replies and 140
  escalations in the Day-3 run — every "maybe reply" became a `needs_human`
  escalation instead of a model-generated reply. That is *by design*, stated in
  the transcript.
- **Statistics vs narrative.** The confidence gate
  (`analytics_confidence_downgrade`) labels any comparison with n=1 in a group
  `hypothesis_unverified` even if the direction looks real — the report says
  "observed_correlation" only at n≥2/group.

## 7. Honest limitations

- CPU-only Ollama: 7B calls average ~5.5 min; a full two-week run is a
  multi-hour job (see README for `--short`).
- 3 posts/week is a small sample; the discovery hit-rate (2–3 of 6 rules over
  two weeks) is the real, documented number, not a claim that the system finds
  every rule every week.
- The cold-run script is the reproducibility test the brief demands; its
  transcript and timing are in the final audit report.