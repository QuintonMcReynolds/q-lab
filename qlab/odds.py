"""Optional live player-prop lines from The Odds API (https://the-odds-api.com).

The free tier gives 500 credits/month. Each event call costs
(#markets x #regions) credits, so scan a few markets at a time.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"

MARKETS = {
    "Passing Yards": "player_pass_yds",
    "Pass Completions": "player_pass_completions",
    "Pass Attempts": "player_pass_attempts",
    "Passing TDs": "player_pass_tds",
    "Interceptions": "player_pass_interceptions",
    "Rushing Yards": "player_rush_yds",
    "Rush Attempts": "player_rush_attempts",
    "Receptions": "player_receptions",
    "Receiving Yards": "player_reception_yds",
    "Rush + Rec Yards": "player_rush_reception_yds",
    "Anytime TD": "player_anytime_td",
}
MARKET_TO_STAT = {v: k for k, v in MARKETS.items()}

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(name: str) -> str:
    n = re.sub(r"[^a-z ]", "", str(name).lower().replace("-", " "))
    return re.sub(r"\s+", " ", _SUFFIX.sub("", n)).strip()


def fetch_events(api_key: str, days_ahead: int = 7) -> tuple[list[dict], str | None]:
    r = requests.get(f"{API}/events", params={"apiKey": api_key}, timeout=20)
    r.raise_for_status()
    cutoff = datetime.now(timezone.utc) + timedelta(days=days_ahead)
    events = [e for e in r.json()
              if datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) <= cutoff]
    return events, r.headers.get("x-requests-remaining")


def fetch_props(api_key: str, event_id: str, markets: list[str],
                bookmakers: str | None = None) -> tuple[pd.DataFrame, str | None]:
    params = {"apiKey": api_key, "markets": ",".join(markets),
              "oddsFormat": "american", "regions": "us"}
    if bookmakers:
        params["bookmakers"] = bookmakers
        params.pop("regions")
    r = requests.get(f"{API}/events/{event_id}/odds", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    rows = []
    for book in data.get("bookmakers", []):
        for market in book.get("markets", []):
            for o in market.get("outcomes", []):
                side = o.get("name")
                point = o.get("point")
                if market["key"] == "player_anytime_td":
                    if side != "Yes":
                        continue
                    side, point = "Over", 0.5
                rows.append(dict(
                    event_id=event_id, book=book["title"], market=market["key"],
                    stat=MARKET_TO_STAT.get(market["key"]), player=o.get("description"),
                    side=side, line=point, price=o.get("price"),
                    game=f"{data.get('away_team')} @ {data.get('home_team')}",
                ))
    return pd.DataFrame(rows), r.headers.get("x-requests-remaining")


def best_lines(props: pd.DataFrame) -> pd.DataFrame:
    """Collapse books into one row per player/stat/line with the best Over and Under price."""
    if props.empty:
        return props
    wide = []
    for (player, stat, line, game), g in props.groupby(["player", "stat", "line", "game"]):
        over = g[g["side"] == "Over"].sort_values("price", ascending=False).head(1)
        under = g[g["side"] == "Under"].sort_values("price", ascending=False).head(1)
        wide.append(dict(
            player=player, stat=stat, line=line, game=game,
            over_price=over["price"].iloc[0] if len(over) else None,
            over_book=over["book"].iloc[0] if len(over) else None,
            under_price=under["price"].iloc[0] if len(under) else None,
            under_book=under["book"].iloc[0] if len(under) else None,
            books=g["book"].nunique(),
        ))
    return pd.DataFrame(wide)
