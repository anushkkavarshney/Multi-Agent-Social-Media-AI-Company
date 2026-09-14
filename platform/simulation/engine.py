"""Engagement simulation engine.

Pipeline for one post (called by the /simulate/tick route):
    1. Draw a latent post "quality" q ~ Normal(1.0, 0.25)  — the unobserved
       factor that makes ALL metrics on a good post move together (this is the
       real statistical structure the Analytics Agent can discover).
    2. Compose the reach rules (time window, hashtags, length, novelty)
       multiplicatively (hidden_rules.apply_reach_multipliers).
    3. Expected impressions = BASE_MEAN * q * M, drawn from a NEGATIVE BINOMIAL
       (not uniform): NB has a heavy right tail, so a few posts go "viral" —
       exactly the structure engagement data has in reality, and exactly what a
       uniform randint would destroy (brief section 6.3 calls this out).
    4. Likes/shares/saves/clicks are binomial draws off impressions; comments
       are Poisson — with the question-CTA rule scaling the comment lambda.
    5. follower_delta: base Poisson scaled by quality, then rule 6 multiplies
       it when save-rate > 5% (the subtle second-order signal).
    6. Generate actual comment TEXT (Community Manager needs real input).

All randomness flows through one numpy Generator handed in by the caller —
the platform seeds it from settings.sim_seed so runs are reproducible.
"""

import numpy as np

from platform.simulation.hidden_rules import (
    apply_reach_multipliers,
    follower_delta_save_boost,
    question_cta_comment_lift,
)

# --- Baseline distribution parameters (tuned so a typical post reaches
#     ~3-8k impressions; the 1/scale NB draw gives a realistic heavy tail) ---
BASE_MEAN_IMPRESSIONS = 5000.0
QUALITY_MEAN = 1.0
QUALITY_SD = 0.25          # latent quality spread; drives cross-metric correlation
NB_DISPERSION = 5.0        # NB size parameter r: var = mean + mean^2/r -> heavy tail

# --- Engagement rate baselines (per impression, before quality scaling) ---
LIKE_RATE = 0.06
SHARE_RATE = 0.010
SAVE_RATE_BASE = 0.025
CLICK_RATE = 0.015
COMMENT_RATE = 0.004       # Poisson rate per impression -> expected comments
FOLLOWER_RATE = 0.001      # followers gained per impression, pre-boost
CTA_CLICK_UPLIFT = 1.5     # posts with an explicit CTA get 1.5x click rate


def _nb_draw(rng: np.random.Generator, mean: float) -> int:
    """Draw from NegativeBinomial(mean, dispersion=NB_DISPERSION).

    numpy's parameterization: mean = n(1-p)/p => p = n/(n+mean).
    With var = mean + mean^2/n, n=5 gives a heavy right tail (viral outliers).
    """
    if mean <= 0:
        return 0
    n = NB_DISPERSION
    p = n / (n + mean)
    return int(rng.negative_binomial(n, p))


def _clip_prob(p: float) -> float:
    """Binomial probabilities must live in [0, 0.9] (0.9 cap keeps draws sane
    even when quality outliers inflate a rate)."""
    return min(max(p, 0.0), 0.9)


def simulate_post_engagement(
    post: dict,
    channel_profile: dict,
    recent_history: dict,
    rng: np.random.Generator,
) -> dict:
    """Simulate cumulative engagement for one post; returns a metric dict.

    `post` needs: copy, hashtags (list), cta (str|None), format (str),
    published_at (datetime), channel_id (str). `channel_profile` is the parsed
    character_profile_json. `recent_history` carries format_history for the
    novelty rule. `rng` is a numpy Generator (reproducibility lives upstream).

    Returns metric fields plus `multipliers` — an observability record of every
    rule's contribution to THIS post (gold for debugging and for the write-up's
    ground-truth table).
    """
    # 1. Latent quality — one draw couples every downstream metric.
    quality = float(np.clip(rng.normal(QUALITY_MEAN, QUALITY_SD), 0.2, 3.0))

    # 2. Reach multipliers from the hidden rules (deterministic given inputs).
    M_reach = apply_reach_multipliers(post, channel_profile, recent_history)

    # 3. Impressions from a noisy NB baseline.
    # `exposure_fraction` (<1.0) scales expected reach for posts published
    # partway through a simulate-window — they simply had less time to accrue.
    exposure = float(post.get("exposure_fraction", 1.0))
    expected_impressions = BASE_MEAN_IMPRESSIONS * quality * M_reach * exposure
    impressions = _nb_draw(rng, expected_impressions)

    # 4. Engagement counts off impressions.
    likes = int(rng.binomial(impressions, _clip_prob(LIKE_RATE * quality)))
    shares = int(rng.binomial(impressions, _clip_prob(SHARE_RATE * quality)))
    saves = int(rng.binomial(impressions, _clip_prob(SAVE_RATE_BASE * quality)))

    click_rate = CLICK_RATE * quality * (CTA_CLICK_UPLIFT if post.get("cta") else 1.0)
    clicks = int(rng.binomial(impressions, _clip_prob(click_rate)))

    # Comments: Poisson with lambda scaled by quality AND the question rule.
    q_mult = question_cta_comment_lift(post, channel_profile, recent_history)
    comment_lambda = impressions * COMMENT_RATE * quality * q_mult
    n_comments = int(rng.poisson(comment_lambda)) if impressions > 0 else 0

    # 5. Follower delta with the second-order save-rate boost (rule 6).
    save_rate = (saves / impressions) if impressions > 0 else 0.0
    post_with_rate = {**post, "save_rate": save_rate}
    follower_mult = follower_delta_save_boost(post_with_rate, channel_profile, recent_history)
    follower_base = impressions * FOLLOWER_RATE * quality
    follower_delta = int(rng.poisson(max(follower_base * follower_mult, 0.0)))

    # Clamp comments to impressions (a comment implies an impression; the
    # Pydantic validator would otherwise rightly reject the snapshot).
    n_comments = min(n_comments, impressions)

    return {
        "impressions": impressions,
        "likes": likes,
        "comments": n_comments,
        "shares": shares,
        "saves": saves,
        "clicks": clicks,
        "follower_delta": follower_delta,
        "multipliers": {
            "reach_composite": M_reach,
            "question_comment_lift": q_mult,
            "follower_save_boost": follower_mult,
            "latent_quality": quality,
            "expected_impressions": expected_impressions,
        },
    }


# --- Simulated comment text -------------------------------------------------
# The Community Manager Agent reads these, so they must read like real replies.
# Templates are generic-but-natural and reference the post's own words loosely.

_POSITIVE = [
    "okay this is genuinely useful",
    "saving this one",
    "finally someone said it",
    "been looking for something like this",
    "great point about {topic}",
]

_QUESTION = [
    "wait but does it work with {topic}?",
    "what's the catch here?",
    "how does this compare to just doing it manually?",
    "any advice for total beginners?",
]

_NEGATIVE = [
    "this feels overhyped honestly",
    "tried something similar, didn't work for me",
    "kinda generic take",
    "not convinced tbh",
]

_SENSITIVE = [
    "why does everything have to be so expensive...",
    "cheaper alternative: just use the free tool lol",
    "support never replied to my ticket, fyi everyone",
]

_FAKE_HANDLES = [
    "night_owl_sam", "coffeeandcode", "priya.builds", "marcus_t", "the_real_deal",
    "anaonabudget", "studentlauncher", "dailygrindzoe", "kiranvv", "quietq",
]


class _Pool:
    """Draws template indexes without immediate repetition.

    A shuffled permutation per pass: identical templates can still recur
    (pools are small), just never back-to-back — repeated identical comments
    read as obviously fake to a human skimming the feed.
    """

    def __init__(self, size: int, rng: np.random.Generator):
        self.size = size
        self.rng = rng
        self._order = rng.permutation(size).tolist()

    def next(self) -> int:
        if not self._order:
            self._order = self.rng.permutation(self.size).tolist()
        return self._order.pop()


def generate_comment_texts(post: dict, n: int, rng: np.random.Generator) -> list[dict]:
    """Generate `n` comment dicts for a post: author, text, sentiment, sensitivity.

    Mix: ~45% positive / ~30% question (neutral) / ~15% negative / ~10% pool of
    sensitive comments (price complaints, competitor mentions, support gripes)
    flagged `is_sensitive=True` for the Community Manager's escalation path.
    """
    topic = _topic_word(post.get("copy", ""))
    pools = {
        "pos": _Pool(len(_POSITIVE), rng),
        "q": _Pool(len(_QUESTION), rng),
        "neg": _Pool(len(_NEGATIVE), rng),
        "sens": _Pool(len(_SENSITIVE), rng),
    }
    comments: list[dict] = []
    for _ in range(n):
        roll = rng.random()
        if roll < 0.10:
            text, sentiment, sensitive = _SENSITIVE[pools["sens"].next()], "negative", True
        elif roll < 0.55:
            text, sentiment, sensitive = _POSITIVE[pools["pos"].next()], "positive", False
        elif roll < 0.85:
            text, sentiment, sensitive = _QUESTION[pools["q"].next()], "neutral", False
        else:
            text, sentiment, sensitive = _NEGATIVE[pools["neg"].next()], "negative", False
        text = text.format(topic=topic)
        handle = _FAKE_HANDLES[int(rng.integers(0, len(_FAKE_HANDLES)))] + str(int(rng.integers(10, 99)))
        comments.append(
            {
                "author_handle": handle,
                "text": text,
                "sentiment": sentiment,
                "is_sensitive": sensitive,
            }
        )
    return comments


def _topic_word(copy: str) -> str:
    """Pull a plausible topic noun from the post copy for template filling.

    Heuristic: prefer a longer word (>=6 chars) — long words skew noun-ish and
    read naturally in "great point about X" — falling back to 'this' when the
    copy is too short to be informative. Comments stay grammatical no matter
    what the Writer Agent produced.
    """
    words = [w.strip(".,!?#;:").lower() for w in copy.split()]
    words = [w for w in words if w.isalpha()]
    long_words = sorted({w for w in words if len(w) >= 6}, key=len, reverse=True)
    return long_words[0] if long_words else "this"
