"""Verify that current season data is fresh and accurate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.nhl_api_client import NHLAPIClient


def main():
    client = NHLAPIClient()

    # 1) Check standings freshness
    data = client.get_standings_now()
    standings = data.get("standings", [])
    date = standings[0].get("date", "unknown") if standings else "N/A"
    print(f"Standings date: {date}")
    print(f"Number of teams: {len(standings)}")
    print()

    header = f"{'Team':>4s}  {'GP':>3s}  {'W':>3s}  {'L':>3s}  {'OTL':>3s}  {'PTS':>4s}  {'PTS%':>6s}  {'GF':>4s}  {'GA':>4s}  {'DIFF':>5s}  {'Conf':>4s}  {'Div':>5s}"
    print(header)
    print("-" * len(header))

    for t in sorted(standings, key=lambda x: x.get("points", 0), reverse=True):
        abbrev = t.get("teamAbbrev", {}).get("default", "???")
        gp = t.get("gamesPlayed", 0)
        w = t.get("wins", 0)
        lo = t.get("losses", 0)
        otl = t.get("otLosses", 0)
        pts = t.get("points", 0)
        pctg = t.get("pointPctg", 0)
        gf = t.get("goalFor", 0)
        ga = t.get("goalAgainst", 0)
        diff = t.get("goalDifferential", 0)
        conf = t.get("conferenceAbbrev", "?")
        div = t.get("divisionAbbrev", "?")
        print(
            f"{abbrev:>4s}  {gp:3d}  {w:3d}  {lo:3d}  {otl:3d}  {pts:4d}  {pctg:6.3f}  {gf:4d}  {ga:4d}  {diff:+5d}  {conf:>4s}  {div:>5s}"
        )

    # 2) Verify NHL stats API data
    print("\n\nNHL Stats API verification:")
    stats = client.get_team_stats(report="summary", season_id="20252026", game_type=2)
    rows = stats.get("data", [])
    print(f"  Teams in stats API: {len(rows)}")
    if rows:
        sample = rows[0]
        print(f"  Sample team: {sample.get('teamFullName', '?')}")
        print(f"  Games played: {sample.get('gamesPlayed', '?')}")
        print(f"  Goals for/game: {sample.get('goalsForPerGame', '?')}")
        print(f"  Goals against/game: {sample.get('goalsAgainstPerGame', '?')}")
        print(f"  PP%: {sample.get('powerPlayPct', '?')}")
        print(f"  PK%: {sample.get('penaltyKillPct', '?')}")
        print(f"  Shot attempts % (5v5): {sample.get('satPercentage', '?')}")

    # 3) Verify MoneyPuck data
    print("\n\nMoneyPuck advanced stats verification:")
    import requests, io, pandas as pd
    mp_url = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/teams.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://moneypuck.com/data.htm",
    }
    resp = requests.get(mp_url, headers=headers, timeout=30)
    if resp.status_code == 200:
        mp = pd.read_csv(io.StringIO(resp.text))
        teams_all = mp[mp["situation"] == "all"] if "situation" in mp.columns else mp
        print(f"  Teams in MoneyPuck: {len(teams_all)}")
        print(f"  Columns: {len(mp.columns)}")
        if "xGoalsPercentage" in mp.columns:
            sample_row = teams_all.iloc[0]
            print(f"  Sample: {sample_row.get('team', '?')}")
            print(f"  xGoals%: {sample_row.get('xGoalsPercentage', '?')}")
            print(f"  Corsi%: {sample_row.get('corsiPercentage', '?')}")
            print(f"  Fenwick%: {sample_row.get('fenwickPercentage', '?')}")
    else:
        print(f"  MoneyPuck returned status {resp.status_code}")

    # 4) Check cache status
    print("\n\nCache status:")
    cache_dir = Path(__file__).resolve().parent.parent / "data" / "cache"
    cache_files = list(cache_dir.glob("*.json"))
    print(f"  Cached API responses: {len(cache_files)}")
    print(f"  NOTE: Delete data/cache/ to force fresh API calls")


if __name__ == "__main__":
    main()
