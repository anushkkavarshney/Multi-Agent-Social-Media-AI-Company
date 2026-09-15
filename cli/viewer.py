"""CLI viewer — rich terminal rendering for feeds, traces, and reports.

Run from the repo root:
    python -m cli.viewer feed [--channel ch_shortform] [--campaign camp_xxx]
    python -m cli.viewer trace camp_xxx
    python -m cli.viewer report camp_xxx 1
    python -m cli.viewer escalations [--campaign camp_xxx]

Every command reads directly from the platform DB (session_factory) so it
works whether or not the server is running.  Rich makes the tables readable
without JSON dumps.  Typer provides the CLI shell.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

app = typer.Typer(help="View platform data in the terminal (feeds, traces, reports).")
console = Console()


# ------------------------------------------------------------------ commands

@app.command()
def feed(
    channel: str | None = typer.Option(None, help="Filter by channel id."),
    campaign: str | None = typer.Option(None, help="Filter by campaign id."),
) -> None:
    """Show the latest posts (newest first) with metrics."""
    import asyncio
    asyncio.run(_feed(channel, campaign))


async def _feed(channel: str | None, campaign_id: str | None) -> None:
    from platform.db import get_session_factory, init_db
    from platform.schema import Channel, Metric, Post, as_utc
    from sqlalchemy import select, func

    await init_db()
    factory = get_session_factory()
    async with factory() as sess:
        channels = {c.id: c.name for c in (await sess.execute(select(Channel))).scalars().all()}
        q = select(Post).order_by(Post.published_at.desc()).limit(200)
        if channel:
            q = q.where(Post.channel_id == channel)
        if campaign_id:
            q = q.where(Post.campaign_id == campaign_id)
        posts = (await sess.execute(q)).scalars().all()

        table = Table(title="Platform Feed", show_lines=True)
        table.add_column("published_at", style="dim", width=20)
        table.add_column("channel", width=16)
        table.add_column("copy", max_width=45)
        table.add_column("tags", max_width=20)
        table.add_column("impressions", justify="right", width=12)
        table.add_column("likes", justify="right", width=8)
        table.add_column("comments", justify="right", width=8)
        table.add_column("saves", justify="right", width=8)
        table.add_column("campaign", style="dim", width=12)

        for p in posts:
            m = (await sess.execute(
                select(Metric).where(Metric.post_id == p.id).order_by(Metric.recorded_at.desc())
            )).scalars().first()
            ch_name = channels.get(p.channel_id, p.channel_id)
            table.add_row(
                as_utc(p.published_at).strftime("%Y-%m-%d %H:%M"),
                ch_name,
                p.copy[:45] + ("..." if len(p.copy) > 45 else ""),
                " ".join(list(p.hashtags_json or [])[:3]),
                str(m.impressions) if m else "-",
                str(m.likes) if m else "-",
                str(m.comments) if m else "-",
                str(m.saves) if m else "-",
                p.campaign_id[:12],
            )
    console.print(table)


@app.command()
def trace(campaign_id: str) -> None:
    """Show the agent message bus trace for a campaign."""
    import asyncio
    asyncio.run(_trace(campaign_id))


async def _trace(campaign_id: str) -> None:
    from orchestration.message_bus import MessageBus
    bus = MessageBus()
    msgs = await bus.history(campaign_id)

    table = Table(title=f"Agent Trace — {campaign_id}", show_lines=False)
    table.add_column("#", style="dim", width=5)
    table.add_column("time", style="dim", width=19)
    table.add_column("from", width=18, style="cyan")
    table.add_column("to", width=18, style="magenta")
    table.add_column("type", width=26, style="bold")
    table.add_column("payload snapshot", max_width=50)

    for i, m in enumerate(msgs, 1):
        payload_str = str(m.payload)[:50]
        table.add_row(
            str(i),
            m.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            m.from_agent,
            m.to_agent,
            m.message_type,
            payload_str,
        )
    console.print(table)
    console.print(f"\nTotal messages: {len(msgs)}")


@app.command()
def report(campaign_id: str, week: int = typer.Argument(...)) -> None:
    """Show a saved weekly report (from logs/) or fetch live from the platform."""
    report_path = ROOT / "logs" / f"week{week}_campaign_{campaign_id}_report.json"
    if report_path.exists():
        data = __import__("json").loads(report_path.read_text(encoding="utf-8"))
        _render_report(data)
    else:
        import asyncio
        asyncio.run(_fetch_report(campaign_id, week))


def _render_report(data: dict) -> None:
    """Render a WeeklyReport dict as rich panels and tables."""
    console.print(Panel(
        f"[bold]Campaign:[/] {data['campaign_id']}   [bold]Week:[/] {data['week_number']}",
        title="Weekly Report",
    ))

    # KPIs
    kpi_table = Table(title="KPI Performance", show_lines=False)
    kpi_table.add_column("metric", style="bold")
    kpi_table.add_column("measured", justify="right")
    for k, v in data.get("kpi_performance", {}).items():
        kpi_table.add_row(k, f"{v:,.4f}" if isinstance(v, float) else str(v))
    console.print(kpi_table)

    # Patterns
    console.print("\n[bold]Patterns Found:[/]")
    for p in data.get("patterns_found", []):
        console.print(f"  • {p}")

    # Recommendations
    console.print("\n[bold]Recommendations:[/]")
    for i, r in enumerate(data.get("recommendations", []), 1):
        console.print(f"  {i}. {r['change']}")
        console.print(f"     evidence      : {r['evidence']}")
        console.print(f"     expected effect: {r['expected_effect']}")

    # Post insights
    console.print("\n[bold]Post Insights:[/]")
    for pi in data.get("post_insights", []):
        rank = pi["performance_rank"]
        color = "green" if rank == "top" else "red" if rank == "bottom" else "dim"
        console.print(f"  [{color}]{rank:6s}[/] {pi['post_id']}: {pi['hypothesis'][:80]}")

    # Sentiment
    console.print(f"\n[bold]Sentiment:[/] {data.get('comment_sentiment_summary', '-')[:120]}")


async def _fetch_report(campaign_id: str, week: int) -> None:
    from platform.client import PlatformClient
    platform = PlatformClient()
    try:
        data = await platform.get_weekly(campaign_id, week)
        console.print(f"[dim]Live data from platform for week {week}[/]\n")
        _render_report({
            "campaign_id": campaign_id,
            "week_number": week,
            "kpi_performance": {"impressions": data["totals"]["impressions"],
                                "likes": data["totals"]["likes"],
                                "comments": data["totals"]["comments"]},
            "patterns_found": ["(platform patterns dict)", str(data.get("patterns", {}))[:200]],
            "post_insights": [],
            "recommendations": [],
            "comment_sentiment_summary": "(fetch saved report for full narrative)",
        })
    except Exception as exc:
        console.print(f"[red]Could not fetch: {exc}[/]")


@app.command()
def escalations(campaign: str | None = typer.Option(None)) -> None:
    """Show the needs_human_review escalation queue."""
    import asyncio
    asyncio.run(_escalations(campaign))


async def _escalations(campaign_id: str | None) -> None:
    from platform.db import get_session_factory, init_db
    from platform.schema import NeedsHumanReview, as_utc
    from sqlalchemy import select

    await init_db()
    factory = get_session_factory()
    async with factory() as sess:
        q = select(NeedsHumanReview).order_by(NeedsHumanReview.created_at.desc()).limit(100)
        if campaign_id:
            q = q.where(NeedsHumanReview.campaign_id == campaign_id)
        rows = (await sess.execute(q)).scalars().all()

    table = Table(title="Needs Human Review", show_lines=True)
    table.add_column("id", style="dim", width=16)
    table.add_column("campaign", width=14)
    table.add_column("post_id", width=16)
    table.add_column("comment_id", width=16)
    table.add_column("reason", max_width=60)
    table.add_column("time", width=19)

    for r in rows:
        table.add_row(
            r.id, r.campaign_id[:14], r.post_id, r.comment_id,
            r.reason[:60],
            as_utc(r.created_at).strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)
    console.print(f"\nTotal escalations: {len(rows)}")


if __name__ == "__main__":
    app()