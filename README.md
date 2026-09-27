# 🏈 PropLab: NFL Player Prop Research Tool

PropLab is a free, open-source NFL player-prop analyzer inspired by tools like Prop Season. It combines **historical hit rates**, **defense-vs-position matchup grades**, and a **recency-weighted projection model** to estimate the probability that a prop goes over or under, then compares that to sportsbook odds to find expected value (EV).

Built with Python, pandas, Streamlit, and Plotly. Data comes from [nflverse](https://github.com/nflverse) (free, updated nightly).

## Features

| Tab | What it does |
|---|---|
| 🔎 **Prop Analyzer** | Pick a player, stat, line, and odds. Get a projection, over/under probability, fair odds, EV, a verdict, a last-15-games chart vs. the line, hit-rate splits (L5 / L10 / season / home-away / vs. opponent), and the game log. |
| 📋 **Projections Board** | Projects every active player for this week's slate, with matchup grades. |
| 🛡️ **Matchup Grades** | Ranks all 32 defenses (A+ to F) by how much of a stat they allow to each position. |
| 📡 **Live Prop Scanner** | *(Optional)* Pulls real prop lines from [The Odds API](https://the-odds-api.com) across books and ranks every prop by model EV. |
| 🧾 **Bet Slip** | Collects picks, calculates parlay odds, payout, model hit chance, and EV. Exports to CSV. |

**Stats:** passing yards/completions/attempts/TDs/INTs, rushing yards/attempts, receptions, receiving yards, rush+rec yards, pass+rush yards, anytime TD.

## How the model works

1. **Base rate:** recency-weighted average of the player's last 20 games (each older game counts 13% less; last season ×0.6, two seasons ago ×0.3).
2. **Matchup adjustment:** opponent's "allowed to position" rate vs. league average, blended with last season early in the year and regressed toward average for small samples. 75% of the effect is applied.
3. **Probability:** normal distribution for yardage and volume stats (σ from the player's own variance, with floors). Poisson distribution for TDs and INTs, shrunk toward the position average so no one sits at 0%.
4. **Value:** compares model probability with the sportsbook's implied probability. EV = expected profit per $1.

## Run it locally

```bash
git clone https://github.com/QuintonMcReynolds/prop-lab.git
cd prop-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Optional live lines: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your free Odds API key. You can also paste the key into the Scanner tab.

## Deploy free (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app**, and pick this repo with `app.py`.
3. *(Optional)* Under **Advanced settings → Secrets**, add `ODDS_API_KEY = "..."`.

## Tests

```bash
pip install pytest && pytest -q
```

## Project structure

```
app.py               Streamlit UI
proplab/data.py      nflverse loading, schedule & Vegas context
proplab/model.py     projections, hit rates, matchup grades, odds math
proplab/odds.py      The Odds API client (optional)
tests/               unit tests for the model and odds math
```

## Disclaimer

PropLab is for research and entertainment. No model can guarantee results, and sportsbooks price in information (injuries, role changes, weather) that this model doesn't see. Only bet what you can afford to lose. If gambling stops being fun, call 1-800-GAMBLER.
