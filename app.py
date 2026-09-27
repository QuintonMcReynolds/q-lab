"""Q Lab: NFL player prop research tool (hit rates, matchup grades, projections)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qlab import data, model, odds

st.set_page_config(page_title="Q Lab · NFL Props", page_icon="🏈", layout="wide")

GREEN, RED, MUTED = "#22c55e", "#ef4444", "#71717a"

st.markdown(
    """
    <style>
      .block-container {padding-top: 3rem; max-width: 1300px;}
      .pl-card {background:#141416;border:1px solid #27272a;border-radius:12px;padding:14px 16px;}
      .pl-verdict {font-size:1.35rem;font-weight:700;letter-spacing:.3px;}
      .pl-over {color:#22c55e;} .pl-under {color:#ef4444;} .pl-pass {color:#a1a1aa;}
      .pl-grade {display:inline-block;min-width:44px;text-align:center;font-weight:800;
                 border-radius:8px;padding:2px 8px;font-size:1.1rem;}
      .pl-small {color:#a1a1aa;font-size:.85rem;}
      div[data-testid="stMetricValue"] {font-size:1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ data
@st.cache_data(ttl=3 * 3600, show_spinner="Loading NFL data from nflverse…")
def load_all():
    games = data.load_games()
    season = data.current_season(games)
    stats = data.load_player_weeks([season - 2, season - 1, season])
    df = data.prepare(stats, games)
    nxt = data.next_games(games, season)
    return df, nxt, season


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def defense(stat: str, position: str, season: int) -> pd.DataFrame:
    df, _, _ = load_all()
    return model.defense_table(df, stat, position, season)


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def player_index(season: int) -> pd.DataFrame:
    df, _, _ = load_all()
    recent = df[df["season"] >= season - 1]
    last = recent.sort_values("gameday").groupby("player_id").tail(1)
    idx = last[["player_id", "player_display_name", "position", "team", "season", "headshot_url"]]
    counts = recent.groupby("player_id").size().rename("games")
    idx = idx.join(counts, on="player_id")
    idx["label"] = idx["player_display_name"] + "  (" + idx["position"] + ", " + idx["team"] + ")"
    return idx.set_index("player_id")


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def volume(stat: str, season: int) -> pd.Series:
    """Recent per-game average, used to sort the player dropdown (busiest players first)."""
    df, _, _ = load_all()
    d = df[df["season"] >= season - 1].assign(v=lambda x: model.stat_values(x, stat))
    return d.groupby("player_id")["v"].apply(lambda v: v.tail(8).mean()).rename("vol")


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def prior(stat: str, position: str, season: int) -> float | None:
    if model.STATS[stat].kind != "rare":
        return None
    df, _, _ = load_all()
    return model.position_prior(df, stat, position, season)


df, NEXT, SEASON = load_all()
PLAYERS = player_index(SEASON)

if "slip" not in st.session_state:
    st.session_state.slip = []


def grade_color(grade: str) -> str:
    return {"A": "#16a34a", "B": "#65a30d", "C": "#ca8a04", "D": "#ea580c", "F": "#dc2626"}[grade[0]]


def grade_badge(grade: str) -> str:
    return f'<span class="pl-grade" style="background:{grade_color(grade)}22;color:{grade_color(grade)}">{grade}</span>'


def fmt_odds(o) -> str:
    if o is None or (isinstance(o, float) and np.isnan(o)):
        return "–"
    o = round(o)
    return f"+{o}" if o > 0 else str(o)


def matchup_for(team: str, position: str, stat: str):
    """Return (next-game row or None, defense row or None)."""
    game = NEXT.loc[team] if team in NEXT.index else None
    if game is None:
        return None, None
    table = defense(stat, position, SEASON).set_index("defense")
    return game, (table.loc[game["opponent"]] if game["opponent"] in table.index else None)


def add_to_slip(player, stat, side, line, price, p_side):
    st.session_state.slip.append(dict(player=player, stat=stat, side=side, line=line,
                                      odds=int(price), model_prob=round(p_side, 4)))
    st.toast(f"Added {player} {side} {line} {stat}")


# ------------------------------------------------------------------ header
left, right = st.columns([3, 2])
with left:
    st.markdown("## 🏈 Q Lab")
    st.caption("Hit rates · matchup grades · model projections for NFL player props")
with right:
    latest = df[df["season"] == SEASON]["week"].max()
    wk = int(NEXT["week"].min()) if len(NEXT) else None
    st.markdown(
        f'<div class="pl-card pl-small">Season <b>{SEASON}</b> · stats through <b>Week {latest}</b>'
        + (f' · upcoming <b>Week {wk}</b>' if wk else "")
        + "<br>Data: nflverse (updated nightly). For research/entertainment. Bet responsibly.</div>",
        unsafe_allow_html=True,
    )

tab_an, tab_board, tab_def, tab_scan, tab_slip = st.tabs(
    ["🔎 Prop Analyzer", "📋 Projections Board", "🛡️ Matchup Grades", "📡 Live Prop Scanner",
     f"🧾 Bet Slip ({len(st.session_state.slip)})"]
)

# ================================================================ ANALYZER
with tab_an:
    c1, c2, c3, c4, c5 = st.columns([3, 2, 1.2, 1, 1])
    stat = c2.selectbox("Stat", list(model.STATS), key="an_stat")
    sdef = model.STATS[stat]
    pool = PLAYERS[PLAYERS["position"].isin(sdef.positions)].join(volume(stat, SEASON))
    pool = pool.sort_values("vol", ascending=False)
    pid = c1.selectbox("Player", pool.index, format_func=lambda p: pool.loc[p, "label"], key="an_player")

    player = PLAYERS.loc[pid]
    log = model.player_log(df, pid, stat)
    default_line = 0.5 if stat == "Anytime TD" else model.suggest_line(log["value"])
    line = c3.number_input("Line", value=float(default_line), step=0.5, min_value=0.0,
                           key=f"line_{pid}_{stat}")
    over_px = c4.number_input("Over odds", value=-110, step=5, key="over_px")
    under_px = c5.number_input("Under odds", value=-110, step=5, key="under_px")

    game, drow = matchup_for(player["team"], player["position"], stat)
    factor = float(drow["factor"]) if drow is not None else 1.0
    proj = model.project(log, stat, SEASON, factor, prior_mean=prior(stat, player["position"], SEASON))

    if proj is None:
        st.info("No games found for this player.")
    else:
        p_over = proj.prob_over(line)
        p_under = 1 - p_over
        valid_odds = abs(over_px) >= 100 and abs(under_px) >= 100
        ev_o = model.expected_value(p_over, over_px) if valid_odds else None
        ev_u = model.expected_value(p_under, under_px) if valid_odds else None
        label, tone = model.verdict(p_over, ev_o, ev_u)

        # --- top row: player/matchup card + verdict
        a, b = st.columns([1.3, 2])
        with a:
            img = player["headshot_url"]
            opp_txt = "No upcoming game found"
            if game is not None:
                where = "vs" if game["is_home"] else "@"
                spread = f'{game["spread"]:+.1f}' if pd.notna(game["spread"]) else "–"
                itt = f'{game["implied_total"]:.1f}' if pd.notna(game["implied_total"]) else "–"
                opp_txt = (f'Week {int(game["week"])} {where} <b>{game["opponent"]}</b> · '
                           f'{pd.Timestamp(game["gameday"]):%a %b %d}<br>'
                           f'Spread {spread} · O/U {game["total"]} · Team total {itt}')
            grade_html = ""
            if drow is not None:
                grade_html = (f'<div style="margin-top:8px">Matchup {grade_badge(drow["grade"])} '
                              f'<span class="pl-small">{game["opponent"]} allows '
                              f'{drow["allowed_pg"]:.1f} {stat.lower()}/g to {player["position"]}s '
                              f'(#{int(drow["rank"])} softest)</span></div>')
            st.markdown(
                f'<div class="pl-card" style="display:flex;gap:14px;align-items:center">'
                + (f'<img src="{img}" onerror="this.style.display=\'none\'" style="width:84px;border-radius:10px">' if isinstance(img, str) else "")
                + f'<div><div style="font-size:1.2rem;font-weight:700">{player["player_display_name"]}</div>'
                  f'<div class="pl-small">{player["position"]} · {player["team"]}</div>'
                  f'<div style="margin-top:6px;font-size:.9rem">{opp_txt}</div>{grade_html}</div></div>',
                unsafe_allow_html=True,
            )
        with b:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Projection", f"{proj.mean:.1f}", f"{proj.mean - line:+.1f} vs line")
            m2.metric("Over prob", f"{p_over:.0%}", f"fair {fmt_odds(model.prob_to_american(p_over))}",
                      delta_color="off")
            m3.metric("EV Over", f"{ev_o:+.1%}" if ev_o is not None else "–")
            m4.metric("EV Under", f"{ev_u:+.1%}" if ev_u is not None else "–")
            st.markdown(f'<div class="pl-card"><span class="pl-small">Q Lab verdict</span><br>'
                        f'<span class="pl-verdict pl-{tone}">{label}</span>'
                        f'<div class="pl-small" style="margin-top:4px">Base {proj.base:.1f} × matchup '
                        f'{proj.matchup_factor:.2f} · σ {proj.sd:.1f} · {proj.games_used} games</div></div>',
                        unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            s1.button(f"➕ Over {line}", width="stretch", disabled=not valid_odds, on_click=add_to_slip,
                      args=(player["player_display_name"], stat, "Over", line, over_px, p_over))
            s2.button(f"➕ Under {line}", width="stretch", disabled=not valid_odds, on_click=add_to_slip,
                      args=(player["player_display_name"], stat, "Under", line, under_px, p_under))

        # --- chart
        recent = log.tail(15)
        labels = [f"{'' if h else '@'}{o}<br>{s % 100:02d} W{w}" for o, h, s, w in
                  zip(recent["opponent_team"], recent["is_home"], recent["season"], recent["week"])]
        colors = [GREEN if v > line else RED for v in recent["value"]]
        fig = go.Figure(go.Bar(x=labels, y=recent["value"], marker_color=colors,
                               text=recent["value"].round(0).astype(int), textposition="outside"))
        fig.add_hline(y=line, line_dash="dash", line_color="#e4e4e7",
                      annotation_text=f"Line {line}", annotation_position="top left")
        fig.add_hline(y=proj.mean, line_dash="dot", line_color="#60a5fa",
                      annotation_text=f"Proj {proj.mean:.1f}", annotation_position="bottom right")
        fig.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=30, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          title=dict(text=f"Last {len(recent)} games · {stat}", font=dict(size=14)),
                          yaxis=dict(gridcolor="#27272a"), showlegend=False)
        st.plotly_chart(fig, width="stretch")

        # --- hit rates + game log
        h, g = st.columns([1, 1.4])
        with h:
            st.markdown("**Hit rate (Over)**")
            hr = model.hit_rates(log, line, SEASON,
                                 opponent=game["opponent"] if game is not None else None,
                                 is_home=bool(game["is_home"]) if game is not None else None)
            hr["record"] = hr["hits"].astype(str) + "/" + hr["games"].astype(str)
            st.dataframe(
                hr[["split", "record", "hit_rate", "average"]], hide_index=True, width="stretch",
                column_config={
                    "split": "Split", "record": "Hits",
                    "hit_rate": st.column_config.ProgressColumn("Hit %", min_value=0, max_value=1, format="percent"),
                    "average": st.column_config.NumberColumn("Avg", format="%.1f"),
                })
        with g:
            st.markdown("**Game log**")
            cols = ["season", "week", "opponent_team", "is_home", "value", "targets", "carries", "attempts"]
            show = log[cols].iloc[::-1].head(20).copy()
            show["is_home"] = np.where(show["is_home"], "Home", "Away")
            show["result"] = np.where(show["value"] > line, "✅", "❌")
            st.dataframe(show, hide_index=True, width="stretch", height=260,
                         column_config={"opponent_team": "Opp", "is_home": "H/A", "value": stat})

# ================================================================ BOARD
with tab_board:
    b1, b2 = st.columns([2, 3])
    bstat = b1.selectbox("Stat", list(model.STATS), key="board_stat")
    bdef = model.STATS[bstat]
    min_games = b2.slider("Min games played this season", 1, 4, 1)

    @st.cache_data(ttl=3 * 3600, show_spinner="Projecting the slate…")
    def board(stat: str, min_games: int) -> pd.DataFrame:
        sdef = model.STATS[stat]
        cur = df[(df["season"] == SEASON) & df["position"].isin(sdef.positions)]
        counts = cur.groupby("player_id").size()
        rows = []
        for pid_, n in counts[counts >= min_games].items():
            info = PLAYERS.loc[pid_]
            if info["team"] not in NEXT.index:
                continue
            lg = model.player_log(df, pid_, stat)
            game_, drow_ = matchup_for(info["team"], info["position"], stat)
            f = float(drow_["factor"]) if drow_ is not None else 1.0
            pj = model.project(lg, stat, SEASON, f, prior_mean=prior(stat, info["position"], SEASON))
            if pj is None or pj.base <= 0:
                continue
            rows.append(dict(
                player=info["player_display_name"], pos=info["position"], team=info["team"],
                opp=("vs " if game_["is_home"] else "@ ") + game_["opponent"],
                grade=drow_["grade"] if drow_ is not None else "–",
                projection=pj.mean, l5=lg["value"].tail(5).mean(),
                season_avg=lg.loc[lg["season"] == SEASON, "value"].mean(),
                p_over_05=pj.prob_over(0.5) if sdef.kind == "rare" else np.nan,
            ))
        out = pd.DataFrame(rows)
        return out.sort_values("projection", ascending=False).reset_index(drop=True) if len(out) else out

    tbl = board(bstat, min_games)
    if tbl.empty:
        st.info("No upcoming games found (bye week or season over).")
    else:
        cfg = {
            "projection": st.column_config.NumberColumn("Proj", format="%.1f"),
            "l5": st.column_config.NumberColumn("L5 avg", format="%.1f"),
            "season_avg": st.column_config.NumberColumn(f"{SEASON} avg", format="%.1f"),
            "p_over_05": st.column_config.ProgressColumn("P(1+)", min_value=0, max_value=1, format="percent"),
            "grade": "Matchup",
        }
        cols = [c for c in tbl.columns if not (c == "p_over_05" and bdef.kind != "rare")]
        st.dataframe(tbl[cols], hide_index=True, width="stretch", height=560, column_config=cfg)
        st.caption("Matchup grade: A = defense allows the most of this stat to the position (good for Overs), "
                   "F = allows the least.")

# ================================================================ DEFENSE
with tab_def:
    d1, d2 = st.columns(2)
    dstat = d1.selectbox("Stat", list(model.STATS), key="def_stat")
    dpos = d2.selectbox("Vs position", list(model.STATS[dstat].positions), key="def_pos")
    tbl = defense(dstat, dpos, SEASON).copy()
    opp_of = {row["opponent"]: team for team, row in NEXT.iterrows()}
    tbl["facing"] = tbl["defense"].map(opp_of).fillna("–")
    fig = go.Figure(go.Bar(
        x=tbl["defense"], y=tbl["allowed_pg"],
        marker_color=[grade_color(g) for g in tbl["grade"]],
        hovertemplate="%{x}: %{y:.1f}/g<extra></extra>"))
    fig.add_hline(y=tbl["league_avg"].iloc[0], line_dash="dash", line_color="#a1a1aa",
                  annotation_text="League avg")
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      title=dict(text=f"{dstat} allowed per game to {dpos}s (softest → toughest)",
                                 font=dict(size=14)), yaxis=dict(gridcolor="#27272a"))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        tbl[["rank", "defense", "grade", "allowed_pg", "factor", "games", "facing"]],
        hide_index=True, width="stretch", height=420,
        column_config={"allowed_pg": st.column_config.NumberColumn("Allowed/g", format="%.1f"),
                       "factor": st.column_config.NumberColumn("Multiplier", format="%.2f"),
                       "games": f"{SEASON} games", "facing": "Next opp. offense"})
    st.caption(f"Blends {SEASON} (full weight) with {SEASON - 1} (35% weight) and regresses small samples "
               "toward league average.")

# ================================================================ SCANNER
with tab_scan:
    st.markdown("Pull **real sportsbook prop lines** and rank them by model edge. "
                "Needs a free key from [the-odds-api.com](https://the-odds-api.com) (500 credits/month).")
    try:
        default_key = st.secrets.get("ODDS_API_KEY", "")
    except Exception:  # no secrets.toml
        default_key = ""
    default_key = default_key or os.environ.get("ODDS_API_KEY", "")
    k1, k2 = st.columns([2, 3])
    api_key = k1.text_input("Odds API key", value=default_key, type="password")
    markets = k2.multiselect("Markets (each costs 1 credit per game)", list(odds.MARKETS),
                             default=["Receiving Yards", "Rushing Yards"])
    min_ev = st.slider("Show props with EV at least", -0.10, 0.20, 0.02, 0.01, format="%.2f")

    if st.button("📡 Scan this week's props", type="primary", disabled=not (api_key and markets)):
        try:
            events, remaining = odds.fetch_events(api_key)
            frames = []
            prog = st.progress(0.0, text="Fetching lines…")
            for i, ev in enumerate(events):
                p, remaining = odds.fetch_props(api_key, ev["id"], [odds.MARKETS[m] for m in markets])
                frames.append(p)
                prog.progress((i + 1) / len(events), text=f"{ev['away_team']} @ {ev['home_team']}")
            prog.empty()
            st.session_state.scan = (pd.concat(frames) if frames else pd.DataFrame(), remaining)
        except Exception as e:  # noqa: BLE001
            st.error(f"Odds API error: {e}")

    if "scan" in st.session_state:
        raw, remaining = st.session_state.scan
        if remaining:
            st.caption(f"API credits remaining: {remaining}")
        lines = odds.best_lines(raw)
        name_map = {odds.norm_name(n): pid_ for pid_, n in PLAYERS["player_display_name"].items()}
        results = []
        for _, r in lines.iterrows():
            pid_ = name_map.get(odds.norm_name(r["player"]))
            if pid_ is None or r["stat"] is None or pd.isna(r["line"]):
                continue
            info = PLAYERS.loc[pid_]
            lg = model.player_log(df, pid_, r["stat"])
            _, drow_ = matchup_for(info["team"], info["position"], r["stat"])
            pj = model.project(lg, r["stat"], SEASON, float(drow_["factor"]) if drow_ is not None else 1.0,
                               prior_mean=prior(r["stat"], info["position"], SEASON))
            if pj is None:
                continue
            po = pj.prob_over(r["line"])
            for side, price, book, p in [("Over", r["over_price"], r["over_book"], po),
                                         ("Under", r["under_price"], r["under_book"], 1 - po)]:
                if price is None or pd.isna(price):
                    continue
                results.append(dict(player=r["player"], stat=r["stat"], side=side, line=r["line"],
                                    odds=int(price), book=book, projection=pj.mean, model_prob=p,
                                    implied=model.american_to_prob(price),
                                    ev=model.expected_value(p, price),
                                    grade=drow_["grade"] if drow_ is not None else "–", game=r["game"]))
        res = pd.DataFrame(results)
        if res.empty:
            st.info("No props matched. Books may not have posted lines yet.")
        else:
            res = res[res["ev"] >= min_ev].sort_values("ev", ascending=False)
            st.dataframe(res, hide_index=True, width="stretch", height=520, column_config={
                "projection": st.column_config.NumberColumn("Proj", format="%.1f"),
                "model_prob": st.column_config.NumberColumn("Model %", format="percent"),
                "implied": st.column_config.NumberColumn("Book %", format="percent"),
                "ev": st.column_config.NumberColumn("EV", format="percent"),
            })
            st.caption("EV = expected profit per $1 at the best available price. "
                       "Treat big edges with suspicion: they often mean injury/role news the model doesn't know.")

# ================================================================ SLIP
with tab_slip:
    slip = st.session_state.slip
    if not slip:
        st.info("Add picks from the Prop Analyzer to build a slip.")
    else:
        sdf = pd.DataFrame(slip)
        sdf["ev"] = [model.expected_value(p, o) for p, o in zip(sdf["model_prob"], sdf["odds"])]
        edited = st.data_editor(
            sdf.assign(remove=False), hide_index=True, width="stretch",
            disabled=["player", "stat", "side", "line", "model_prob", "ev"],
            column_config={"model_prob": st.column_config.NumberColumn("Model %", format="percent"),
                           "ev": st.column_config.NumberColumn("EV", format="percent"),
                           "remove": st.column_config.CheckboxColumn("Remove")},
        )
        if edited["remove"].any():
            st.session_state.slip = edited[~edited["remove"]].drop(columns=["remove", "ev"]).to_dict("records")
            st.rerun()

        stake = st.number_input("Stake ($)", value=10.0, min_value=1.0, step=5.0)
        dec = float(np.prod([model.american_to_decimal(o) for o in edited["odds"]]))
        p_all = float(np.prod(edited["model_prob"]))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Parlay odds", fmt_odds((dec - 1) * 100 if dec >= 2 else -100 / (dec - 1)))
        c2.metric("Payout", f"${stake * dec:,.2f}")
        c3.metric("Model hit chance", f"{p_all:.1%}")
        c4.metric("Parlay EV", f"{p_all * dec - 1:+.1%}")
        st.caption("Parlay math assumes legs are independent. Same-game legs are usually correlated.")
        b1, b2 = st.columns(2)
        b1.download_button("⬇️ Export slip (CSV)", sdf.to_csv(index=False), "qlab_slip.csv",
                           width="stretch")
        if b2.button("🗑️ Clear slip", width="stretch"):
            st.session_state.slip = []
            st.rerun()
