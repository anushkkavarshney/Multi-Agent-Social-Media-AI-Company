"""HTTP client for the mock platform — the ONLY way agents touch the platform.

Day 2 wired agents to fixture data and selected the platform's own DB via
SQLAlchemy directly. Day 3 introduces the real boundary: Scheduler publishes
via POST /posts, Community Manager reads comments / posts replies, Analytics
reads the weekly aggregation — all through this client. Agents receive a
PlatformClient instance; run_demo passes one pointed at the running app, and
the offline tests pass one wrapping httpx.ASGITransport (no server needed).

Contract:
  - every method is async and returns the parsed JSON body of the endpoint
  - every method raises PlatformConnectionError with an actionable message
    when the platform is unreachable (so run_demo can tell the user to run
    `python -m platform.main` instead of dumping a stack trace)
  - the transport is injectable primarily for tests: PlatformClient(app=app)
    builds its own httpx.AsyncClient(transport=ASGITransport) and because the
    ASGI transport skips lifespan startup, the caller must call init_db()
    before exercising it.
"""

from __future__ import annotations

from typing import Any

import httpx

from config.settings import get_settings


class PlatformConnectionError(Exception):
    """Platform unreachable or returned a failure — carries a fix hint."""


class PlatformClient:
    def __init__(
        self,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        *,
        app=None,
    ) -> None:
        """Create a client.

        With `app` set (tests), a client is built over httpx.ASGITransport so
        the FastAPI app is called in-process; with `client` set, that client is
        used as-is (fully mocked transports). Otherwise a real client against
        `base_url` (defaults to settings.platform_base_url) is created.
        """
        from httpx import ASGITransport

        if app is not None:
            base = base_url or "http://test"
            self._client: httpx.AsyncClient = httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url=base
            )
        elif client is not None:
            self._client = client
        else:
            base = (base_url or get_settings().platform_base_url).rstrip("/")
            self._client = httpx.AsyncClient(base_url=base, timeout=30.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            resp = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise PlatformConnectionError(
                f"cannot reach the mock platform ({exc.__class__.__name__}).\n"
                "  Fix: run the platform in another terminal:\n"
                "       python -m platform.main   # http://127.0.0.1:8010\n"
            ) from exc
        if resp.status_code >= 400:
            detail = resp.text[:300]
            raise PlatformConnectionError(
                f"platform {method} {path} failed with {resp.status_code}: {detail}"
            )
        return resp.json()

    # --- platform read side -------------------------------------------------

    async def health(self) -> dict:
        return await self._request("GET", "/health")

    async def get_channels(self) -> list[dict]:
        data = await self._request("GET", "/channels")
        channels = data["channels"]
        # Flatten character_profile into the agent-facing contract (audience,
        # tone, and the engagement knobs) so prompt templates can address
        # channels as flat dicts.
        for ch in channels:
            profile = ch.pop("character_profile", None)
            if isinstance(profile, dict):
                ch.update(profile)
        return channels

    async def get_feed(self, channel: str | None = None, campaign_id: str | None = None) -> list[dict]:
        params: dict[str, str] = {}
        if channel:
            params["channel"] = channel
        if campaign_id:
            params["campaign_id"] = campaign_id
        data = await self._request("GET", "/feed", params=params)
        return data["posts"]

    # --- publish + engagement (Scheduler / Community Manager) ---------------

    async def publish_post(
        self,
        *,
        campaign_id: str,
        channel_id: str,
        copy: str,
        hashtags: list[str],
        cta: str | None,
        format: str,
        published_at: str,
    ) -> dict:
        return await self._request(
            "POST",
            "/posts",
            json={
                "campaign_id": campaign_id,
                "channel_id": channel_id,
                "copy": copy,
                "hashtags": hashtags,
                "cta": cta,
                "format": format,
                "published_at": published_at,
            },
        )

    async def tick(self, campaign_id: str, hours: float) -> dict:
        return await self._request(
            "POST", "/simulate/tick", json={"campaign_id": campaign_id, "hours": hours}
        )

    async def get_comments(self, post_id: str) -> list[dict]:
        data = await self._request("GET", f"/posts/{post_id}/comments")
        return data["comments"]

    async def post_reply(self, post_id: str, comment_id: str, reply_text: str) -> dict:
        return await self._request(
            "POST",
            f"/posts/{post_id}/comments/{comment_id}/reply",
            json={"reply_text": reply_text, "author": "brand_team"},
        )

    async def get_metrics(self, post_id: str) -> dict:
        return await self._request("GET", f"/posts/{post_id}/metrics")

    # --- analytics ----------------------------------------------------------

    async def get_weekly(self, campaign_id: str, week: int) -> dict:
        return await self._request("GET", f"/analytics/week/{campaign_id}/{week}")