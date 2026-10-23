"""
HTTP client for the NHL Web and Stats APIs.

Includes disk-based caching, rate limiting, and retry logic
so that re-runs don't hammer the API.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import (
    API_MAX_RETRIES,
    API_RATE_LIMIT_DELAY,
    API_REQUEST_TIMEOUT,
    CACHE_DIR,
    NHL_STATS_API,
    NHL_WEB_API,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class NHLAPIClient:
    """Thin wrapper around the two NHL API base URLs with caching."""

    def __init__(self, cache_dir: Path | None = None, cache_ttl_hours: float = 6.0):
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._cache_dir = cache_dir or CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_ts: float = 0.0
        self._cache_ttl_hours = cache_ttl_hours

    # ── public helpers ──────────────────────────────────────────────────
    def get_standings(self, date: str) -> dict[str, Any]:
        """GET /v1/standings/{date}  (e.g. '2024-04-18')."""
        return self._web(f"/standings/{date}")

    def get_standings_now(self) -> dict[str, Any]:
        return self._web("/standings/now")

    def get_club_stats_season(self, team: str) -> dict[str, Any]:
        return self._web(f"/club-stats-season/{team}")

    def get_team_schedule_season(
        self, team: str, season_id: str
    ) -> dict[str, Any]:
        return self._web(f"/club-schedule-season/{team}/{season_id}")

    def get_playoff_bracket(self, year: int) -> dict[str, Any]:
        return self._web(f"/playoff-bracket/{year}")

    def get_schedule(self, date: str) -> dict[str, Any]:
        return self._web(f"/schedule/{date}")

    def get_roster(self, team: str, season_id: str) -> dict[str, Any]:
        return self._web(f"/roster/{team}/{season_id}")

    # ── stats API (api.nhle.com/stats/rest/en) ──────────────────────────
    def get_team_stats(
        self,
        report: str = "summary",
        season_id: str | None = None,
        game_type: int = 2,
        **extra_params: Any,
    ) -> dict[str, Any]:
        """
        Fetch team-level stats from the stats API.

        report: summary | percentages | realtime | penalties | penaltykill |
                powerplay | faceoffpercentages | outshootoutshotby |
                goalsbyperiod | goalsagainstbystrength | goalsforbystrength |
                leadingtrailing | summaryshooting | shottype | scoretrailfirst
        game_type: 2 = regular season, 3 = playoffs
        """
        params: dict[str, Any] = {"isAggregate": "false", "isGame": "false"}
        if season_id:
            params["cayenneExp"] = (
                f"gameTypeId={game_type} and seasonId<={season_id} "
                f"and seasonId>={season_id}"
            )
        else:
            params["cayenneExp"] = f"gameTypeId={game_type}"
        params.update(extra_params)
        return self._stats(f"/team/{report}", params)

    def get_goalie_stats(
        self,
        report: str = "summary",
        season_id: str | None = None,
        game_type: int = 2,
        limit: int = -1,
        **extra_params: Any,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "isAggregate": "false",
            "isGame": "false",
            "limit": limit,
            "start": 0,
        }
        if season_id:
            params["cayenneExp"] = (
                f"gameTypeId={game_type} and seasonId<={season_id} "
                f"and seasonId>={season_id}"
            )
        else:
            params["cayenneExp"] = f"gameTypeId={game_type}"
        params.update(extra_params)
        return self._stats(f"/goalie/{report}", params)

    # ── internals ───────────────────────────────────────────────────────
    def _web(self, path: str) -> dict[str, Any]:
        return self._get(f"{NHL_WEB_API}{path}")

    def _stats(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{NHL_STATS_API}{path}"
        return self._get(url, params=params)

    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        stop=stop_after_attempt(API_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _get(
        self, url: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        cache_key = self._cache_key(url, params)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        self._rate_limit()
        logger.debug("GET %s  params=%s", url, params)
        resp = self._session.get(url, params=params, timeout=API_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        self._write_cache(cache_key, data)
        return data

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < API_RATE_LIMIT_DELAY:
            time.sleep(API_RATE_LIMIT_DELAY - elapsed)
        self._last_request_ts = time.monotonic()

    def _cache_key(self, url: str, params: dict[str, Any] | None) -> str:
        raw = url + json.dumps(params or {}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            import datetime
            mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
            age_hours = (datetime.datetime.now() - mtime).total_seconds() / 3600
            if age_hours > self._cache_ttl_hours:
                path.unlink()
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _write_cache(self, key: str, data: dict[str, Any]) -> None:
        path = self._cache_dir / f"{key}.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
