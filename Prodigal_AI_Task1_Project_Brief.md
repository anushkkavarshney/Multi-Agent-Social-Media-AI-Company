# Prodigal AI — Task 1 Project Brief & Execution Plan
### Multi-Agent Social Media AI Company (Round 1 of 2)

**Deadline:** 4 calendar days from receipt
**Submit to:** surabhi@prodigalai.com
**Subject line:** `Task 1 Submission — <Your Full Name> — <College/Current Org>`

---

## 1. What This Task Actually Tests

Prodigal AI is not evaluating whether you can call an LLM. They are evaluating whether you can design a **system of agents** that cooperate, disagree, get reviewed, and produce output a human would trust — entirely on local compute, with no hosted APIs. The write-up is weighted as heavily as the code. Honesty about what broke is scored as a *positive* signal, not a negative one.

Automatic reject conditions:
- Any hosted LLM API anywhere in the pipeline (OpenAI, Anthropic, Gemini, Groq, etc.)
- Code that doesn't run from your own README instructions
- A submission that looks like one unedited AI-generated codebase with no sign of iteration

---

## 2. Inputs You're Given

- This brief (the spec above) — no dataset, no starter code, no API keys
- A single example of the kind of instruction the system must accept:
  > "We are launching a budget espresso machine for students. Run a two-week awareness campaign across our three channels, target price-sensitive 18–25 year olds, keep the tone playful, don't over-promise."
- Freedom to choose: language/framework, agent framework (or none), local model(s), hardware you already own

## 3. Inputs the System Itself Must Accept (at runtime)

- A **human brief** in loose natural language (like the espresso example) → this is the trigger for campaign creation
- Human **approve/reject** decisions at the review gate before publishing
- Read access to its own **mock platform's** feed, metrics, and comments (the system feeds itself its own simulated data)

## 4. Outputs You Must Produce

### A. From the running system (functional outputs)
1. A parsed **campaign object**: objective, target audience, channel mix, duration, content pillars/themes, post-by-post calendar, per-post copy + creative brief, and the KPIs it'll be judged on
2. **Published posts** on your own mock platform, at scheduled slots
3. **Simulated audience response**: impressions, likes, comments, shares, saves, click-throughs, follower delta, and actual simulated comment text
4. Community Manager **draft replies** to comments (with an escalation path for sensitive ones)
5. A **weekly analysis report**: performance vs. KPIs, top/bottom posts with hypotheses, cross-cutting patterns (format, length, timing, tone, channel, hashtags, CTA), comment sentiment themes, and *specific* next-week changes with evidence
6. Ideally: a **second week run** using the agent's own recommendations, with before/after numbers

### B. Submission deliverables (what you email)
1. **Code ZIP** (or GitHub repo link + ZIP regardless) — setup instructions, dependency file, seed/demo data included; no model weights, `venv/`, or `node_modules/`
2. **README** (repo root) — setup/run instructions, architecture, how to reproduce the demo
3. **PDF write-up** (heavily weighted) containing:
   - Architecture diagram: agent org + message flow
   - Model(s) used, hardware, quantisation, and *why*
   - Prompting/orchestration techniques and why (routing, structured output, self-critique, retries, tool use, memory, RAG, guardrails)
   - Mock platform's engagement model, including the hidden signals you planted
   - A full sample run: brief → campaign → week of posts → weekly report
   - **"What didn't work"** section — local-model failure modes, things you tried and abandoned, what you'd do with two more weeks
4. **Demo video**, 3–6 minutes, unlisted link (optional but strongly recommended)

---

## 5. Hard Requirements Checklist

| # | Requirement | Status |
|---|---|---|
| 1 | Ollama + local model(s), documented (model, quantisation, hardware) | ☐ |
| 2 | Zero hosted LLM APIs — including for embeddings if used | ☐ |
| 3 | Genuinely multi-agent: separate agents, separate prompts/tools (not one "act as a team" prompt) | ☐ |
| 4 | Self-built mock social platform with its own storage + API | ☐ |
| 5 | Campaign creation flow driven by a human brief | ☐ |
| 6 | Weekly analysis output: what worked, why, what to change | ☐ |
| 7 | Code runs from your own README, cold, on a clone | ☐ |

---

## 6. System Architecture to Build

### 6.1 Local Model Layer
- Thin abstraction over Ollama, not raw calls scattered everywhere
- Must show: streaming, retry-on-malformed-output, structured/JSON output enforcement, token accounting, graceful fallback when the model returns garbage
- Realistic sizing: a 7–8B model handled well beats a 70B model needed to compensate for weak engineering
- Consider a **two-model split** — small/fast model for routing/classification, larger model for generation — and justify the split in the write-up

### 6.2 Agent Organisation (suggested — deviate if you can defend it)

| Agent | Owns |
|---|---|
| Orchestrator / Chief of Staff | Parses brief, decomposes work, routes tasks, resolves conflicts, decides campaign is ready |
| Strategy Agent | Audience, channel mix, objectives, cadence, KPIs |
| Content Writer Agent | Post copy, hooks, CTAs, hashtags, per-channel tone |
| Creative Agent | Visual/creative brief per post (text brief or placeholder — no image gen required) |
| Scheduler / Publisher Agent | Approved content → calendar → pushes to mock platform |
| Community Manager Agent | Reads comments/DMs, drafts replies, escalates sensitive items |
| Brand & Compliance Agent | Reviews everything pre-publish; can reject and send back |
| Analytics Agent | Reads engagement data, finds patterns, writes weekly review |

**What they're really scoring here:** inter-agent communication design.
- How does work move between agents — shared state, or a message bus?
- What happens when Compliance rejects a post 3x in a row? (Must not loop forever — needs an escalation-to-human or max-retry exit)
- Is there memory of past campaigns, and do later decisions actually *use* it?
- Can a human override at any point?
- **Log/trace the agent conversation** — this is explicitly called out as one of the most valuable things to include.

### 6.3 Mock Social Media Platform
Minimum bar:
- Multiple channels with distinct character (e.g. short-form video, text/discussion, professional) where the *same* content performs differently
- Endpoints: publish post, fetch feed, fetch metrics per post, read comments
- Simulated engagement: impressions, likes, comments, shares, saves, click-throughs, follower delta
- Simulated comment text (not just numbers) — Community Manager needs real input
- Persistent storage (SQLite is fine)
- Minimal UI is a plus, not required — API + CLI viewer is acceptable

**Critical design point:** `random.randint(0,100)` engagement gives the Analytics Agent nothing real to discover — this will read as hollow. Plant **hidden, discoverable rules**, e.g.:
- Posts in certain time windows get disproportionate reach
- Posts ending in a question get more comments
- Hashtag count has a non-linear effect on reach
- Certain channels penalize long copy
- Novelty decay when the same format repeats 3 days running

Document these rules in the README as ground truth, then **honestly report** which ones the Analytics Agent found and which it missed.

### 6.4 Campaign Formation
- Human brief → campaign object (objective, audience, channel mix, duration, pillars, calendar, per-post copy + creative brief, KPIs)
- A real **human approve/reject gate** before anything publishes — must actually block execution, not just log an approval

### 6.5 Weekly Analysis & Improvement Loop
Report must cover:
- Performance vs. KPIs set at campaign start
- Top/bottom posts with a hypothesis for why
- Cross-cutting patterns: format, length, timing, tone, channel, hashtags, CTA
- Comment sentiment + recurring themes
- **Concrete, specific** next-week changes with evidence (e.g. "shift Channel B posts from 09:00 to 18:30, cut copy under 80 words, drop the third hashtag" — not "post more engaging content")

**Strongest possible submission:** feed the recommendations back into the Strategy Agent, run week two, show actual before/after numbers, and put them front and center in the README.

---

## 7. Evaluation Weights — Where to Spend Your 4 Days

| Area | Weight | Priority |
|---|---|---|
| Agent architecture (separation, communication, orchestration) | 25% | Highest — this is the core thesis of the task |
| Depth of the analysis loop (real insight in weekly report) | 20% | High |
| Quality of the mock platform (fair, non-trivial testbed) | 15% | High |
| Engineering quality (structure, readability, error handling, reproducibility) | 15% | High |
| Write-up quality and honesty about limitations | 15% | High |
| Local model handling (reliability on constrained hardware) | 10% | Medium |

**Implication:** architecture + analysis depth + platform quality = 60% of the score, and none of them require a bigger model — they require better design. This is where scoping time should go before polish.

---

## 8. Suggested 4-Day Timeline

- **Day 1 — Foundations:** Ollama model wrapper with structured-output + retry logic; mock platform schema (SQLite) with hidden engagement rules designed and written down first (design the rules before the agents, so you know what "success" looks like for Analytics); basic publish/feed/metrics/comments API.
- **Day 2 — Core agent loop:** Orchestrator, Strategy, Content Writer, Compliance with reject-and-revise loop (with a hard max-retry exit), human approval gate. Get one campaign end-to-end producing a calendar.
- **Day 3 — Publish, simulate, analyze:** Scheduler publishes to mock platform, simulate a week of engagement, Community Manager replies to comments, Analytics Agent produces the weekly report. This is the highest-weighted stretch (25% + 20% = 45%) — protect this day.
- **Day 4 — Loop closure + write-up + polish:** If time allows, feed Analytics recommendations back into Strategy and run week 2 for before/after numbers. Write the PDF (architecture diagram, model choices, prompting techniques, hidden-rule documentation, sample run, honest failure-mode section). Record the demo video. Clean README, test a cold clone-and-run yourself.

Cut scope, not corners. Four tightly-integrated agents that genuinely hand off work and get reviewed will beat eight thin agents with no real interaction.

---

## 9. How to Stand Out

Given your existing strengths (RAG/agentic orchestration work from SIH, LLM automation + evaluation experience at Prodigal itself, comfort with structured engineering deliverables), these are the highest-leverage differentiators:

1. **Design the hidden engagement rules like a real experiment, not flavor text.** Write them down *before* building the agents, treat them as ground truth, and in the write-up report a precise hit rate ("found 4/6, missed 2 — here's why") rather than vague success claims. This is explicitly what they say they want to read.
2. **Make the reject loop a real state machine.** Compliance → Writer revision → Compliance again, with a bounded retry count and a defined "escalate to human" exit. Show this in a log. Most candidates will either skip rejection entirely or let it loop unbounded — a clean bounded loop with logging is an easy differentiator.
3. **Actually close the loop (week 2).** The brief explicitly flags this as what separates "strongest submissions." Even a scoped-down version (fewer posts, shorter week) with real before/after numbers outperforms a polished single-week demo.
4. **Show the agent trace, not just the output.** A structured log/transcript of agents talking to each other — routing decisions, rejections, revisions — is called out by name as one of the most valuable artifacts. Treat this as a first-class deliverable, not a debug log.
5. **Be specific and numeric in the weekly report, matching their own example.** "Shift Channel B posts from 09:00 to 18:30, cut copy to under 80 words, drop the third hashtag" is the bar — vague recommendations will read as shallow regardless of how sophisticated the underlying pipeline is.
6. **Write the failure-mode section with real specificity.** Since you already do model evaluation work professionally, this section should be a strength: document actual malformed outputs you hit, what validation/retry logic you added in response, and where it still breaks. Generic "small models sometimes hallucinate" statements will read as filler.
7. **Justify your model split explicitly.** If you route with a small fast model and generate with a larger one, quantify the tradeoff (latency, quality, consistency) rather than asserting it.
8. **Treat the README as a reproducibility test, not documentation.** Actually clone your own repo into a clean environment and follow your own instructions before submitting — this is a stated automatic-reject condition if it fails.

---

## 10. Open Decisions to Make Early (and document as assumptions)

- Language/framework: custom orchestration vs. LangGraph/CrewAI/AutoGen — pick one and state the tradeoff
- Model(s): single generalist model vs. small-router + large-writer split
- Message passing: shared state store vs. explicit message bus between agents
- Memory: does the Orchestrator/Strategy agent persist and reuse insights across campaigns, or only within one run?
- Scope cut: if 4 agents fully working beats 8 half-wired ones, decide now which 4–6 are non-negotiable and which are stretch goals for Day 4
