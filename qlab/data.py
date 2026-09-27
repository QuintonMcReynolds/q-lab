"""Data loading from nflverse (free, public, updated nightly during the season)."""
from __future__ import annotations

from datetime import date

import pandas as pd

STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.parquet"
)
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

OFFENSE_POSITIONS = ["QB", "RB", "WR", "TE"]

KEEP_COLS = [
    "player_id", "player_display_name", "position", "headshot_url", "season", "week",
    "season_type", "game_id", "team", "opponent_team",
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds", "target_share",
]


def load_games() -> pd.DataFrame:
    games = pd.read_csv(GAMES_URL)
    games["gameday"] = pd.to_datetime(games["gameday"])
    return games


def current_season(games: pd.DataFrame, today: date | None = None) -> int:
    """Latest season whose first game has already kicked off (or is this week)."""
    today = pd.Timestamp(today or date.today())
    starts = games.groupby("season")["gameday"].min()
    started = starts[starts <= today + pd.Timedelta(days=7)]
    return int(started.index.max())


def load_player_weeks(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        try:
            df = pd.read_parquet(STATS_URL.format(season=season))
        except Exception:  # season file not published yet
            continue
        frames.append(df[[c for c in KEEP_COLS if c in df.columns]])
    if not frames:
        raise RuntimeError("Could not download any nflverse player stats.")
    return pd.concat(frames, ignore_index=True)


def prepare(stats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Keep offensive skill players and attach home/away + game date."""
    df = stats[stats["position"].isin(OFFENSE_POSITIONS)].copy()
    g = games[["game_id", "home_team", "gameday", "spread_line", "total_line"]]
    df = df.merge(g, on="game_id", how="left")
    df["is_home"] = df["team"] == df["home_team"]
    df = df.drop(columns=["home_team"])
    df = df.sort_values(["gameday", "season", "week"]).reset_index(drop=True)
    return df


def next_games(games: pd.DataFrame, season: int, today: date | None = None) -> pd.DataFrame:
    """One row per team: its next unplayed game this season, with Vegas context."""
    today = pd.Timestamp(today or date.today()).normalize()
    upcoming = games[
        (games["season"] == season)
        & games["result"].isna()
        & (games["gameday"] >= today - pd.Timedelta(days=1))
    ].sort_values("gameday")

    rows = []
    for _, g in upcoming.iterrows():
        spread = g["spread_line"]  # positive = home team favored
        total = g["total_line"]
        home_tt = (total + spread) / 2 if pd.notna(total) and pd.notna(spread) else None
        away_tt = (total - spread) / 2 if pd.notna(total) and pd.notna(spread) else None
        base = dict(game_id=g["game_id"], week=g["week"], gameday=g["gameday"],
                    gametime=g.get("gametime"), total=total)
        rows.append(dict(base, team=g["home_team"], opponent=g["away_team"], is_home=True,
                         spread=-spread if pd.notna(spread) else None, implied_total=home_tt))
        rows.append(dict(base, team=g["away_team"], opponent=g["home_team"], is_home=False,
                         spread=spread, implied_total=away_tt))
    if not rows:
        return pd.DataFrame(columns=["team", "opponent", "is_home", "week", "gameday",
                                     "spread", "total", "implied_total", "game_id"])
    out = pd.DataFrame(rows)
    return out.drop_duplicates("team", keep="first").set_index("team")
