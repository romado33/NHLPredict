"""
Central configuration for the NHL Playoff Predictor.
All magic numbers, API URLs, and tunable parameters live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"

for d in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── NHL API ─────────────────────────────────────────────────────────────────
NHL_WEB_API = "https://api-web.nhle.com/v1"
NHL_STATS_API = "https://api.nhle.com/stats/rest/en"

API_REQUEST_TIMEOUT = 30  # seconds
API_RATE_LIMIT_DELAY = 0.5  # seconds between requests
API_MAX_RETRIES = 3

# ── Season Range ────────────────────────────────────────────────────────────
# 25 years: 2000-01 through 2024-25 (playoffs happen in spring of end-year)
FIRST_SEASON = 2001  # end-year of first season (2000-01)
LAST_HISTORICAL_SEASON = 2025  # end-year of last completed playoff season
CURRENT_SEASON_ID = "20252026"
CURRENT_SEASON_END_YEAR = 2026

# Season ID format: "20002001" for the 2000-01 season
def season_id(end_year: int) -> str:
    return f"{end_year - 1}{end_year}"

# ── Teams ───────────────────────────────────────────────────────────────────
# Current NHL team abbreviations (32 teams as of 2024-25)
# Historical teams included for older seasons
CURRENT_TEAMS = [
    "ANA", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH",
    "NJD", "NYI", "NYR", "OTT", "PHI", "PIT", "SJS", "SEA",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WSH", "WPG",
]

HISTORICAL_TEAMS = {
    "ATL": {"active_until": 2011, "relocated_to": "WPG"},
    "ARI": {"active_from": 2014, "active_until": 2024, "relocated_to": "UTA"},
    "PHX": {"active_until": 2014, "relocated_to": "ARI"},
    "HFD": {"active_until": 1997, "relocated_to": "CAR"},
}

# ── Model ───────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ── MoneyPuck ───────────────────────────────────────────────────────────────
MONEYPUCK_BASE = "https://moneypuck.com/moneypuck/playerData/seasonSummary"
# MoneyPuck uses start-year directories (e.g. 2024 for the 2024-25 season)
MONEYPUCK_TEAM_URL_TEMPLATE = (
    "https://moneypuck.com/moneypuck/playerData/seasonSummary/"
    "{start_year}/regular/teams.csv"
)
