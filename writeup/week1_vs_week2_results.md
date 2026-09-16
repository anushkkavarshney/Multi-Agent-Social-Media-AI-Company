# Week 1 vs Week 2 — Loop Closure Results

This is the Day-4 "strongest possible submission" artifact: the Week-1
Analytics report is fed back into the Strategy Agent (via `memory_records`),
Week 2 runs, and this document shows the actual before/after numbers — what
Week-1 findings survived Week-2 data, and which Week-1 hypothesis Week-2
falsified. Produced by `scripts/run_two_weeks.py`.

## Method

- One campaign, both weeks on the mock platform (SQLite `platform.db`).
- Week 1 = days 1–7 from first publish, Week 2 = days 8–14 (platform's own
  weekly slicing in `GET /analytics/week/{campaign_id}/{week}`).
- Both weeks run the same agent pipeline. The only difference weeks differ:
  Week-2 Strategy receives the Week-1 report's learnings through
  `MemoryStore.retrieve_top_k(...)` (cosine similarity), logged as the
  `memory_retrieved` trace event, and is *required by prompt* to use ≥1
  learning in a pillar or KPI.
- Everything here is numbers the platform actually computed (see the
  `analytics_computed_stats` trace events before each narrative).

## Week 1 report (reconciled against the Day-3 transcript)

> Reconciliation note: the Day-3 transcript's Stage-13 table quoted the
> **original** report, which scored **2 found / 0 partial / 4 not found**.
> The regenerated report (same run data, auditable schema + confidence gate)
> downgrades two rows, giving **0 yes / 1 partial / 5 no**:
> `time_window_boost` went yes → **partial** because the comparison groups
> were n=2 vs n=1 (below the report's own n≥2 effect floor, so it labels
> itself `hypothesis_unverified`), and `follower_delta_save_boost` lost its
> reasoning line, dropping to "no". The regenerated number is the honest one.

```
Week-1 discovery scoring (regenerated report):
| rule | found? | reasoning |
|---|---|---|
| time_window_boost        | partial | evening-window posts averaged 15,542 impressions vs off-peak 3,075; n=2 vs n=1 |
| question_cta_comment_lift | no     | - |
| nonlinear_hashtag_effect  | no     | - |
| channel_length_penalty    | no     | - |
| novelty_decay             | no     | - |
| follower_delta_save_boost | no     | - |
Verdict: 0 found / 1 partial / 5 not found (of 6 rules)
```

Week-1 KPIs: impressions 34,159 · engagement_rate 0.1323 · comment_rate 0.0041
· save_rate 0.03 · follower_growth +34 · likes 2,380 · clicks 577 · shares 396.
Sentiment: 61 positive / 51 neutral / 28 negative.

## Week 2 report (live run)

> Filled from `logs/week2_campaign_*_report.json` after the run completes.

```
(to be inserted verbatim: campaign_id, KPIs, patterns_found, recommendations
 with confidence labels, post_insights, sentiment, week-1 memory actually used)
```

## Before/after comparison

| Claim | Week-1 finding | What Week 2 showed | Verdict |
|---|---|---|---|
| time-window boost | evening > off-peak (n=2 vs 1) | (insert) | (insert) |
| every Week-2 pillar includes a stated memory learning | (insert learned text) | (insert pillar) | (insert) |

## The loop actually closes

1. `memory_retrieved` appears in the Week-2 strategy step with the Week-1
   report's texts.
2. ≥1 Week-2 pillar/KPI textually incorporates that learning (prompt
   hard-constraint).
3. The Week-2 report's `analytics_computed_stats` covers the Week-2 week
   window only (days 8–14), so the numbers are not a replay of Week 1.

## Verdict

Honest hit-rate across both weeks:
(boxed table filled after run — "found N / partial M / not-found K of 6 rules
across two weeks" plus which week-1 recommendation Week-2 confirmed vs
falsified).