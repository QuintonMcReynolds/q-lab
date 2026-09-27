import math

import pandas as pd
import pytest

from qlab import model, odds


def make_log(values, season=2026):
    return pd.DataFrame({
        "season": [season] * len(values),
        "week": range(1, len(values) + 1),
        "gameday": pd.date_range("2026-09-10", periods=len(values), freq="7D"),
        "opponent_team": ["KC", "BUF"] * (len(values) // 2) + ["KC"] * (len(values) % 2),
        "is_home": [True, False] * (len(values) // 2) + [True] * (len(values) % 2),
        "value": values,
    })


def test_odds_conversions():
    assert model.american_to_prob(-110) == pytest.approx(0.5238, abs=1e-4)
    assert model.american_to_prob(+150) == pytest.approx(0.4)
    assert model.american_to_decimal(-200) == pytest.approx(1.5)
    assert model.prob_to_american(0.6) == pytest.approx(-150)
    assert model.expected_value(0.5238, -110) == pytest.approx(0, abs=1e-3)
    o, u = model.no_vig(-110, -110)
    assert o == pytest.approx(0.5)


def test_poisson_over():
    mu = 1.2
    assert model.poisson_over(mu, 0.5) == pytest.approx(1 - math.exp(-mu))
    assert model.poisson_over(mu, -0.5) == 1.0


def test_projection_centers_on_recent_form():
    log = make_log([50] * 10 + [100] * 5)
    proj = model.project(log, "Receiving Yards", 2026)
    assert 70 < proj.mean < 100  # recency-weighted toward the last 5
    assert proj.prob_over(proj.mean - 0.5) > 0.5 > proj.prob_over(proj.mean + 30)


def test_matchup_factor_scales_projection():
    log = make_log([80] * 8)
    soft = model.project(log, "Receiving Yards", 2026, matchup_factor=1.2)
    tough = model.project(log, "Receiving Yards", 2026, matchup_factor=0.8)
    assert soft.mean > 80 > tough.mean


def test_hit_rates():
    log = make_log([10, 20, 30, 40, 50, 60])
    hr = model.hit_rates(log, 35.5, 2026, opponent="KC").set_index("split")
    assert hr.loc["Last 5", "hits"] == 3
    assert hr.loc["2026 season", "games"] == 6


def test_suggest_line_is_half_point():
    assert model.suggest_line(pd.Series([60, 70, 75, 80])) == 72.5


def test_name_normalization():
    assert odds.norm_name("Kenneth Walker III") == odds.norm_name("Kenneth Walker")
    assert odds.norm_name("Amon-Ra St. Brown") == "amon ra st brown"


def test_rare_stat_prior_prevents_zero_probability():
    log = make_log([0] * 12)
    no_prior = model.project(log, "Anytime TD", 2026)
    with_prior = model.project(log, "Anytime TD", 2026, prior_mean=0.3)
    assert no_prior.prob_over(0.5) == 0
    assert 0.02 < with_prior.prob_over(0.5) < 0.2
