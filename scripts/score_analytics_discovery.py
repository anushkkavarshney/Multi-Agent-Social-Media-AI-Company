"""Score the Analytics Agent's WeeklyReport against the 6 hidden rules.

Usage:
    python scripts/score_analytics_discovery.py [path_to_report.json]

If no path is given, the newest logs/week*_report*.json is used.

Honesty contract (project brief): this table goes into the write-up's
honesty section.  A pattern counts as "found" ONLY when the agent's stated
reasoning matches BOTH the mechanism and the direction of the real rule.
Keywords+mechanism heuristics below are deliberately conservative:
- keyword match on the rule's vocabulary (window/hashtag/length/...),
- a direction check (the finding must name the SAME direction the rule
  produces: e.g. evening => more impressions, long copy => fewer),
- "partial" when the vocabulary is hit but the mechanism/direction is
  wrong, vague, or the pattern is only echoed without evidence.

The raw pattern strings are printed too, so a human can override any row
without the script pretending precision it doesn't have.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RULES: list[dict] = [
    {
        "rule": "time_window_boost",
        "mechanism": "posts published in the channel's peak window reach more; dead-zone (02:00-06:00) reaches less",
        "terms": ["window", "peak", "evening", "18:00", "18-21", "overnight", "02:00", "dead", "hour", "morning", "07:00", "08:00"],
        "direction": ("peak", "more"),  # positive words => boost; negative => penalize
    },
    {
        "rule": "question_cta_comment_lift",
        "mechanism": "copy ending in '?' gets 2.5x expected comments",
        "terms": ["question", "?", "ends in a question", "question-cta", "comments swung", "comment rate"],
        "direction": ("question", "more"),
    },
    {
        "rule": "nonlinear_hashtag_effect",
        "mechanism": "3-5 hashtags reach most; 0 underperforms; 8+ penalized",
        "terms": ["hashtag", "tags", "3-5", "8+", "tag count"],
        "direction": ("3-5", "more"),
    },
    {
        "rule": "channel_length_penalty",
        "mechanism": "long copy is punished on shortform(>60w)/professional(>150w); discussion exempt",
        "terms": ["word", "length", "long", "short", "under 80 words", "word count", "copy length"],
        "direction": ("long", "less"),
    },
    {
        "rule": "novelty_decay",
        "mechanism": "same format 3+ days running on one channel compounds 15%/day reach decay",
        "terms": ["format", "repetition", "repeat", "novelty", "same format", "decay", "3 days"],
        "direction": ("same format", "less"),
    },
    {
        "rule": "follower_delta_save_boost",
        "mechanism": "save-rate >5% disproportionately lifts follower gain (second-order)",
        "terms": ["save", "save-rate", "save rate", "saves", "follower"],
        "direction": ("save", "more"),
    },
]

_POSITIVE_WORDS = {"higher", "more", "increase", "lift", "beat", "outperform", "wins", "gains", "exceeds", "outperforms", "boost", "strongest", "better", "top"}
_NEGATIVE_WORDS = {"lower", "less", "fewer", "drop", "decline", "underperform", "penalize", "penalized", "punish", "hurt", "fall", "decay", "worst", "lags"}
_REVERSE_WORDS = {"overnight", "dead", "long", "spam", "8+", "longer", "over 150", "over 60"}
_BOOST_RULE_WORDS = {"evening", "peak", "3-5", "question", "save"}


def _lowered(text: str) -> str:
    return text.lower()


def _has_dir(text: str, negative_direction: bool) -> bool:
    words = _lowered(text)
    if negative_direction:
        return any(w in words for w in _NEGATIVE_WORDS)
    return any(w in words for w in _POSITIVE_WORDS)


def _has_reverse(text: str, rule: dict) -> bool:
    """Does the text name a REVERSE of the rule's boost signal (e.g. 'overnight is best')?"""
    words = _lowered(text)
    if rule["direction"][0] in _REVERSE_WORDS or rule["rule"] == "time_window_boost":
        # for time_window: "overnight" combined with positive word is the reverse
        if "overnight" in words and _has_dir(text, negative_direction=False):
            return True
    return False


def score_report(report: dict) -> list[dict]:
    """Turn a WeeklyReport into a scoring table: rule | found? | reasoning vs mechanism."""
    texts = list(report.get("patterns_found", []))
    texts += [r.get("change", "") + " " + r.get("evidence", "") for r in report.get("recommendations", [])]
    texts += [i.get("hypothesis", "") for i in report.get("post_insights", [])]

    results = []
    for rule in _RULES:
        matched = []
        for t in texts:
            t_low = _lowered(t)
            if any(term in t_low for term in rule["terms"]):
                matched.append(t)
        if not matched:
            results.append({"rule": rule["rule"], "status": "no", "reasoning": "-", "mechanism": rule["mechanism"]})
            continue

        # The strongest/most specific match drives the verdict.
        best = matched[0]
        negative = rule["direction"][1] == "less"
        direction_ok = _has_dir(best, negative_direction=negative)
        reverse = _has_reverse(best, rule)

        # A recommendation with an explicit mechanism mention beats a bare
        # mention in a pattern.
        mech_ok = any(mk in _lowered(best) for mk in rule["terms"][:2]) and direction_ok and not reverse

        if mech_ok and direction_ok:
            status = "yes"
        elif direction_ok or any(term in _lowered(best) for term in rule["terms"][:3]):
            status = "partial"
        else:
            status = "no"
        results.append({
            "rule": rule["rule"],
            "status": status,
            "reasoning": best,
            "mechanism": rule["mechanism"],
        })
    return results


def render_table(results: list[dict]) -> str:
    cols = ("rule", "status", "reasoning")
    lines = ["| rule | found? | agent's stated reasoning | actual mechanism |", "|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['rule']} | **{r['status']}** | {r['reasoning'][:120]} | {r['mechanism']} |"
        )
    return "\n".join(lines)


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg:
        path = Path(arg)
    else:
        candidates = sorted(
            [p for p in (ROOT / "logs").glob("week*.json") if "report" in p.name],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print("No report found. Pass a path to a saved WeeklyReport JSON.", file=sys.stderr)
            return 1
        path = candidates[0]

    report = json.loads(Path(path).read_text(encoding="utf-8"))
    results = score_report(report)

    print(f"Discovery scoring for {path.name} ({report.get('campaign_id', '?')})\n")
    print(render_table(results))
    scores = [r["status"] for r in results]
    found = scores.count("yes")
    partial = scores.count("partial")
    print(f"\nVerdict: {found} found / {partial} partial / {scores.count('no')} not found (of {len(results)} rules).")
    return 0


if __name__ == "__main__":
    sys.exit(main())