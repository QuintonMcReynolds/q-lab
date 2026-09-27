"""Projection, matchup-grade and hit-rate logic. Pure pandas/numpy, no Streamlit."""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, floor, sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StatDef:
    cols: tuple[str, ...]
    positions: tuple[str, ...]
    kind: str          # "yards" | "count" | "rare"
    min_sd: float      # floor on the per-game standard deviation


STATS: dict[str, StatDef] = {
    "Passing Yards":      StatDef(("passing_yards",), ("QB",), "yards", 35),
    "Pass Completions":   StatDef(("completions",), ("QB",), "count", 3.5),
    "Pass Attempts":      StatDef(("attempts",), ("QB",), "count", 4.5),
    "Passing TDs":        StatDef(("passing_tds",), ("QB",), "rare", 0),
    "Interceptions":      StatDef(("passing_interceptions",), ("QB",), "rare", 0),
    "Rushing Yards":      StatDef(("rushing_yards",), ("QB", "RB", "WR"), "yards", 12),
    "Rush Attempts":      StatDef(("carries",), ("QB", "RB"), "count", 2.5),
    "Receptions":         StatDef(("receptions",), ("RB", "WR", "TE"), "count", 1.3),
    "Receiving Yards":    StatDef(("receiving_yards",), ("RB", "WR", "TE"), "yards", 14),
    "Rush + Rec Yards":   StatDef(("rushing_yards", "receiving_yards"), ("RB", "WR", "TE"), "yards", 18),
    "Pass + Rush Yards":  StatDef(("passing_yards", "rushing_yards"), ("QB",), "yards", 38),
    "Anytime TD":         StatDef(("rushing_tds", "receiving_tds"), ("RB", "WR", "TE", "QB"), "rare", 0),
}

GRADES = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "F"]


def stat_values(df: pd.DataFrame, stat: str) -> pd.Series:
    return df[list(STATS[stat].cols)].fillna(0).sum(axis=1)


# ---------------------------------------------------------------- player logs
def player_log(df: pd.DataFrame, player_id: str, stat: str) -> pd.DataFrame:
    log = df[df["player_id"] == player_id].copy()
    log["value"] = stat_values(log, stat)
    return log.sort_values("gameday").reset_index(drop=True)


def hit_rates(log: pd.DataFrame, line: float, season: int,
              opponent: str | None = None, is_home: bool | None = None) -> pd.DataFrame:
    """Over hit-rate splits, Prop Season style."""
    splits = {
        "Last 5": log.tail(5),
        "Last 10": log.tail(10),
        f"{season} season": log[log["season"] == season],
        f"{season - 1} season": log[log["season"] == season - 1],
    }
    if is_home is not None:
        label = "Home games" if is_home else "Away games"
        splits[label] = log[log["is_home"] == is_home].tail(16)
    if opponent:
        splits[f"vs {opponent} (career)"] = log[log["opponent_team"] == opponent]

    rows = []
    for name, part in splits.items():
        n = len(part)
        hits = int((part["value"] > line).sum())
        rows.append(dict(split=name, hits=hits, games=n,
                         hit_rate=hits / n if n else np.nan,
                         average=part["value"].mean() if n else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ matchup grades
def defense_table(df: pd.DataFrame, stat: str, position: str, season: int) -> pd.DataFrame:
    """How much of `stat` each defense allows per game to `position`.

    Blends this season with last season (this season weighted 1.0, last 0.35) so the
    table is usable in Week 1, and shrinks toward league average for small samples.
    """
    d = df[(df["position"] == position) & (df["season"].between(season - 1, season))].copy()
    d["value"] = stat_values(d, stat)
    per_game = (d.groupby(["season", "week", "opponent_team"], as_index=False)["value"].sum()
                  .rename(columns={"opponent_team": "defense"}))
    if per_game.empty:
        return pd.DataFrame(columns=["defense", "allowed_pg", "factor", "rank", "grade", "games"])

    per_game["w"] = np.where(per_game["season"] == season, 1.0, 0.35)
    league = np.average(per_game["value"], weights=per_game["w"])

    agg = per_game.groupby("defense").apply(
        lambda g: pd.Series({
            "allowed_pg": np.average(g["value"], weights=g["w"]),
            "eff_games": g["w"].sum(),
            "games_this_season": int((g["season"] == season).sum()),
        }),
        include_groups=False,
    ).reset_index()

    raw = agg["allowed_pg"] / league if league else 1.0
    shrink = agg["eff_games"] / (agg["eff_games"] + 4)
    agg["factor"] = 1 + (raw - 1) * shrink
    agg["rank"] = agg["factor"].rank(ascending=False, method="min").astype(int)  # 1 = softest
    pct = (agg["rank"] - 1) / max(len(agg) - 1, 1)
    agg["grade"] = [GRADES[min(int(p * len(GRADES)), len(GRADES) - 1)] for p in pct]
    agg["league_avg"] = league
    agg = agg.rename(columns={"games_this_season": "games"})
    return agg.sort_values("rank").reset_index(drop=True)


# ---------------------------------------------------------------- projection
@dataclass
class Projection:
    mean: float
    sd: float
    base: float
    matchup_factor: float
    games_used: int
    kind: str

    def prob_over(self, line: float) -> float:
        if self.kind == "rare":
            return poisson_over(self.mean, line)
        # Normal approximation; lines are x.5 so no continuity correction needed.
        z = (line - self.mean) / self.sd if self.sd > 0 else (np.inf if line >= self.mean else -np.inf)
        return 1 - normal_cdf(z)


def position_prior(df: pd.DataFrame, stat: str, position: str, season: int) -> float:
    """League-wide per-game average of `stat` for a position (last two seasons)."""
    d = df[(df["position"] == position) & df["season"].between(season - 1, season)]
    return float(stat_values(d, stat).mean()) if len(d) else 0.0


def project(log: pd.DataFrame, stat: str, season: int, matchup_factor: float = 1.0,
            max_games: int = 20, prior_mean: float | None = None,
            prior_games: float = 3.0) -> Projection | None:
    """Recency-weighted projection.

    For rare events (TDs, INTs) the average is shrunk toward `prior_mean` as if the
    player had `prior_games` extra games at the position average, so a player with no
    recent TDs still has a realistic, non-zero chance.
    """
    recent = log.tail(max_games)
    if recent.empty:
        return None
    n = len(recent)
    recency = 0.87 ** np.arange(n)[::-1]                      # newest game weight 1
    season_w = np.select([recent["season"] == season, recent["season"] == season - 1],
                         [1.0, 0.6], 0.3)
    w = recency * season_w
    vals = recent["value"].to_numpy(dtype=float)
    base = float(np.average(vals, weights=w))
    var = float(np.average((vals - base) ** 2, weights=w))
    if STATS[stat].kind == "rare" and prior_mean is not None:
        base = (base * w.sum() + prior_mean * prior_games) / (w.sum() + prior_games)

    # Apply 75% of the matchup effect; defenses regress more than players do.
    mean = max(base * (1 + 0.75 * (matchup_factor - 1)), 0.0)
    sdef = STATS[stat]
    sd = max(sqrt(var), sdef.min_sd, 0.3 * mean)
    return Projection(mean=mean, sd=sd, base=base, matchup_factor=matchup_factor,
                      games_used=n, kind=sdef.kind)


# ---------------------------------------------------------------- math/odds
def normal_cdf(z: float) -> float:
    if z == np.inf:
        return 1.0
    if z == -np.inf:
        return 0.0
    return 0.5 * (1 + erf(z / sqrt(2)))


def poisson_over(mu: float, line: float) -> float:
    """P(X > line) for X ~ Poisson(mu)."""
    k_max = floor(line)
    term = exp(-mu)
    cdf = term
    for k in range(1, k_max + 1):
        term *= mu / k
        cdf += term
    return float(max(0.0, 1 - cdf)) if k_max >= 0 else 1.0


def american_to_prob(odds: float) -> float:
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def american_to_decimal(odds: float) -> float:
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def prob_to_american(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def expected_value(p: float, odds: float) -> float:
    """Expected profit per $1 staked."""
    return p * (american_to_decimal(odds) - 1) - (1 - p)


def no_vig(over_odds: float, under_odds: float) -> tuple[float, float]:
    po, pu = american_to_prob(over_odds), american_to_prob(under_odds)
    return po / (po + pu), pu / (po + pu)


def verdict(p_over: float, ev_over: float | None, ev_under: float | None) -> tuple[str, str]:
    """Return (label, tone) where tone is 'over' | 'under' | 'pass'."""
    if ev_over is not None and ev_under is not None:
        if ev_over > 0.03 and ev_over >= ev_under:
            return f"OVER  (+{ev_over:.1%} EV)", "over"
        if ev_under > 0.03:
            return f"UNDER  (+{ev_under:.1%} EV)", "under"
        return "PASS: no edge at these odds", "pass"
    if p_over >= 0.57:
        return f"Lean OVER  ({p_over:.0%})", "over"
    if p_over <= 0.43:
        return f"Lean UNDER  ({1 - p_over:.0%})", "under"
    return "PASS: coin flip", "pass"


def suggest_line(values: pd.Series) -> float:
    """Default line: last-10 median snapped to x.5, like a book would post."""
    med = float(values.tail(10).median()) if len(values) else 0.5
    return max(floor(med) + 0.5, 0.5)
