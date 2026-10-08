#!/usr/bin/env python3
"""
Football score predictor for the NFL and college football (FBS)  --  v2

Feature groups (all pre-game information only, no leakage)
-----------------------------------------------------------
  base      margin-of-victory Elo, rolling/EWMA points, rest days, matchup-expected points
  advanced  EPA/play (pass, rush), success rate -- offense and defense, rolling.
            NFL: nflverse play-by-play.  CFB: CFBD PPA / success rate.
  qb        (NFL) starting-QB quality (shrunk EPA/dropback), QB changes vs. last game,
            and "starter listed Out/Doubtful" fallback for games where the starter
            isn't known yet.
  injuries  (NFL) weighted count of Out/Doubtful/Questionable players by position group.
  weather   temp, wind, dome flag. NFL: observed (history) / Open-Meteo forecast
            (upcoming).  CFB: CFBD /games/weather (Patreon tier only; skipped otherwise).
  roster    (CFB) returning production, transfer-portal net value, 247 talent composite;
            talent also sets each team's preseason Elo anchor.
  market    (NFL, opt-in via --use-market) Vegas spread/total as features.

Setup
-----
    pip install pandas numpy scikit-learn scipy requests pyarrow

Usage
-----
    python football_predictor.py backtest --league nfl
    python football_predictor.py ablate   --league nfl     # what does each group add?
    python football_predictor.py predict  --league nfl --days 7
    python football_predictor.py export   --league nfl --out data/nfl.json   # JSON for the website

    export CFBD_API_KEY=your_key        # free: https://collegefootballdata.com/key
    python football_predictor.py backtest --league cfb
    python football_predictor.py predict  --league cfb --days 7

Useful flags: --drop advanced,qb,injuries,weather,roster   --use-market
              --model ridge|gbm|ensemble   --start YEAR   --test-seasons N
First run downloads and caches data in ./data_cache (play-by-play is the slow part).
"""
import argparse
import io
import json
import math
import os
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class League:
    name: str
    k: float              # Elo K-factor
    hfa: float            # home-field advantage (Elo points)
    regress: float        # fraction of Elo pulled toward the team's base each offseason
    default_pts: float    # prior for avg points per team per game
    default_start: int    # first season to load
    default_warmup: int   # seasons excluded from training (Elo burn-in)
    default_sigma: float  # fallback residual std of margin
    talent_scale: float   # Elo points per 1 std of talent composite (preseason anchor)


LEAGUES = {
    "nfl": League("nfl", k=20, hfa=48, regress=0.33, default_pts=22.5,
                  default_start=2006, default_warmup=3, default_sigma=13.5, talent_scale=0.0),
    "cfb": League("cfb", k=30, hfa=60, regress=0.30, default_pts=28.0,
                  default_start=2010, default_warmup=2, default_sigma=17.0, talent_scale=40.0),
}

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
NFL_GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
CFBD = "https://api.collegefootballdata.com"
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
CACHE_DIR = "data_cache"

# Approximate stadium coordinates (home team -> lat, lon), used for weather forecasts.
NFL_STADIUMS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4009), "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839), "LA": (33.9535, -118.3392), "LAC": (33.9535, -118.3392),
    "LV": (36.0909, -115.1833), "MIA": (25.9580, -80.2389), "MIN": (44.9737, -93.2575),
    "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316), "SF": (37.4030, -121.9700), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9076, -76.8645),
}

ADV_STATS = ["epa", "pass_epa", "rush_epa", "succ"]
BASE_COLS = [
    "elo_diff", "home_elo", "away_elo", "is_neutral",
    "home_pf", "home_pa", "away_pf", "away_pa",
    "home_ewm_pf", "home_ewm_pa", "away_ewm_pf", "away_ewm_pa",
    "exp_home_pts", "exp_away_pts", "exp_margin", "exp_total",
    "home_rest", "away_rest", "rest_diff", "home_szn_games", "away_szn_games",
]
ADV_COLS = ([f"{sd}_{k}_{s}" for sd in ("home", "away") for k in ("off", "def") for s in ADV_STATS]
            + [f"edge_{s}" for s in ADV_STATS])
QB_COLS = ["home_qb_val", "away_qb_val", "qb_edge", "home_qb_delta", "away_qb_delta",
           "home_qb_n", "away_qb_n"]
INJ_GROUPS = ["qb", "ol", "skill", "def"]
INJ_COLS = [f"{sd}_inj_{g}" for sd in ("home", "away") for g in INJ_GROUPS] + ["inj_total_diff"]
WX_COLS = ["is_dome", "temp", "wind", "cold", "windy", "weather_known"]
ROSTER_COLS = ["home_ret_ppa", "away_ret_ppa", "home_ret_pass", "away_ret_pass",
               "home_portal_net", "away_portal_net", "home_talent", "away_talent",
               "ret_diff", "ret_pass_diff", "portal_diff", "talent_diff",
               "ret_diff_early", "ret_pass_diff_early", "portal_diff_early"]
ADJ_COLS = ([f"{sd}_adj{k}_{s}" for sd in ("home", "away") for k in ("off", "def") for s in ADV_STATS]
            + [f"edge_adj_{s}" for s in ADV_STATS]
            + ["home_adj_pf", "home_adj_pa", "away_adj_pf", "away_adj_pa", "exp_adj_margin", "exp_adj_total"])
TOV_COLS = ["home_tov_comm", "home_tov_forced", "away_tov_comm", "away_tov_forced", "tov_edge"]
INJW_COLS = [f"{sd}_injw_{g}" for sd in ("home", "away") for g in INJ_GROUPS] + ["injw_total_diff"]
NEW_GROUPS = ("adjusted", "turnovers", "injw", "injq")   # groups added in v3
MARKET_COLS = ["mkt_spread", "mkt_total"]

GAME_COLS = ["date", "season", "week", "game_id", "gametime", "home", "away", "home_pts", "away_pts",
             "neutral", "home_class", "away_class", "mkt_spread", "mkt_total", "mkt_ml_home", "mkt_ml_away",
             "roof", "temp", "wind", "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name"] + \
            [f"{sd}_{s}" for sd in ("h", "a") for s in ADV_STATS]


def current_season() -> int:
    t = pd.Timestamp.today()
    return t.year if t.month >= 3 else t.year - 1


def warn(msg: str):
    print(f"  [warn] {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Caching / HTTP helpers
# --------------------------------------------------------------------------- #


def cached(name, fetch, refresh=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name)
    if not refresh and os.path.exists(path):
        return pd.read_pickle(path)
    obj = fetch()
    if obj is not None:
        pd.to_pickle(obj, path)
    return obj


def download(url, timeout=180):
    r = requests.get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def cfbd_get(path, params, key, optional=False):
    """GET from CFBD. optional=True -> return None (with a warning) on auth/404 errors."""
    r = requests.get(CFBD + path, params=params,
                     headers={"Authorization": f"Bearer {key}"}, timeout=90)
    if optional and r.status_code in (401, 402, 403, 404):
        warn(f"{path} unavailable (HTTP {r.status_code}); skipping")
        return None
    r.raise_for_status()
    return r.json()


def _get(d: dict, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


class Context:
    """Side tables the feature builder needs."""

    def __init__(self):
        self.qb_log = {}          # game_id -> [(qb_id, n_dropbacks, epa_sum)]
        self.qb_out = {}          # (season, week, team) -> {gsis_id} of QBs Out/Doubtful
        self.box = None           # per game/side yards, plays, 3rd downs, turnovers (NFL)
        self.inj_counts = None    # DataFrame season, week, team, qb, ol, skill, def
        self.injw_counts = None   # same, but each injury weighted by the player's recent snap share
        self.snap_hist = None     # pfr_id -> (sorted season*100+week keys, snap shares)
        self.gsis2pfr = {}
        self.pval_hist = None     # gsis_id -> (sorted keys, skill EPA/game, defensive production/game)
        self.injq_counts = None   # injuries weighted by each player's recent PRODUCTION (quality)
        self.espn = {}            # team -> {"qb_depth": [names], "out": [players], "qb_out": bool} (live, NFL)
        self.qb_name_id = {}      # normalized QB name -> gsis id (from past starts)
        self.team_season = {}     # (season, team) -> dict(ret_ppa, ret_pass, portal_net, talent_z)
        self.talent_z = {}
        self.season_means = {}
        self.global_means = dict(ret_ppa=0.5, ret_pass=0.5, portal_net=0.0)

    def roster(self, season, team, cls):
        r = self.team_season.get((season, team))
        if r is not None:
            return r
        m = self.season_means.get(season, self.global_means)
        return dict(ret_ppa=m["ret_ppa"], ret_pass=m["ret_pass"], portal_net=0.0,
                    talent_z=0.0 if cls == "fbs" else -2.0)


# --------------------------------------------------------------------------- #
# NFL data
# --------------------------------------------------------------------------- #

PBP_COLS = ["game_id", "home_team", "away_team", "posteam", "pass", "rush", "epa", "success", "wp",
            "qb_dropback", "qb_kneel", "qb_spike", "passer_player_id", "rusher_player_id",
            "yards_gained", "third_down_converted", "third_down_failed", "interception", "fumble_lost"]


def load_nfl(start: int) -> pd.DataFrame:
    df = pd.read_csv(NFL_GAMES_URL)
    df = df[df["season"] >= start].copy()
    loc = df["location"] if "location" in df else pd.Series("Home", index=df.index)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["gameday"]),
        "season": df["season"].astype(int),
        "week": df["week"].astype(int),
        "game_id": df["game_id"],
        "gametime": df.get("gametime"),
        "home": df["home_team"].replace(TEAM_MAP),
        "away": df["away_team"].replace(TEAM_MAP),
        "home_pts": pd.to_numeric(df["home_score"], errors="coerce"),
        "away_pts": pd.to_numeric(df["away_score"], errors="coerce"),
        "neutral": loc.eq("Neutral"),
        "home_class": "nfl", "away_class": "nfl",
        "mkt_spread": pd.to_numeric(df.get("spread_line"), errors="coerce"),  # + = home favored
        "mkt_total": pd.to_numeric(df.get("total_line"), errors="coerce"),
        "mkt_ml_home": pd.to_numeric(df.get("home_moneyline"), errors="coerce"),
        "mkt_ml_away": pd.to_numeric(df.get("away_moneyline"), errors="coerce"),
        "roof": df.get("roof"),
        "temp": pd.to_numeric(df.get("temp"), errors="coerce"),
        "wind": pd.to_numeric(df.get("wind"), errors="coerce"),
        "home_qb_id": df.get("home_qb_id"),
        "away_qb_id": df.get("away_qb_id"),
        "home_qb_name": df.get("home_qb_name"),
        "away_qb_name": df.get("away_qb_name"),
    })
    return out


def summarize_pbp(p: pd.DataFrame):
    """-> (per game/side EPA & success table, per game/QB dropback table)"""
    p = p[p.posteam.notna() & p.epa.notna()]
    p = p[((p["pass"] == 1) | (p["rush"] == 1)) & (p.qb_kneel != 1) & (p.qb_spike != 1)].copy()
    p["side"] = np.where(p.posteam == p.home_team, "h", "a")

    w = p[(p.wp > 0.05) & (p.wp < 0.95)]  # drop garbage time
    keys = ["game_id", "side"]
    adv = pd.DataFrame({"epa": w.groupby(keys).epa.mean(),
                        "succ": w.groupby(keys).success.mean()})
    adv["pass_epa"] = w[w["pass"] == 1].groupby(keys).epa.mean()
    adv["rush_epa"] = w[w["rush"] == 1].groupby(keys).epa.mean()
    adv = adv.reset_index()

    db = p[p.qb_dropback == 1].copy()
    db["qb_id"] = db.passer_player_id.fillna(db.rusher_player_id)
    db = db[db.qb_id.notna()]
    qb = (db.groupby(["game_id", "qb_id"]).epa.agg(n="size", epa_sum="sum").reset_index())
    p["tov"] = ((p.interception == 1) | (p.fumble_lost == 1)).astype(int)
    box = (p.groupby(["game_id", "side"]).agg(plays=("yards_gained", "size"), yards=("yards_gained", "sum"),
                                              conv3=("third_down_converted", "sum"),
                                              fail3=("third_down_failed", "sum"), tov=("tov", "sum")).reset_index())
    return adv, qb, box


def load_nfl_pbp(games: pd.DataFrame, ctx: Context, start: int, end: int) -> pd.DataFrame:
    adv_frames, box_frames = [], []
    for yr in range(start, end + 1):
        def fetch(yr=yr):
            raw = download(f"{NFLVERSE}/pbp/play_by_play_{yr}.parquet")
            if raw is None:
                return None
            print(f"  downloaded play-by-play {yr}", file=sys.stderr)
            return summarize_pbp(pd.read_parquet(io.BytesIO(raw), columns=PBP_COLS))

        res = cached(f"nfl_pbp3_{yr}.pkl", fetch, refresh=yr >= end)
        if res is None:
            warn(f"no play-by-play for {yr}")
            continue
        adv, qb, box = res
        adv_frames.append(adv)
        box_frames.append(box)
        for r in qb.itertuples(index=False):
            ctx.qb_log.setdefault(r.game_id, []).append((r.qb_id, float(r.n), float(r.epa_sum)))
    ctx.box = pd.concat(box_frames, ignore_index=True) if box_frames else None
    if not adv_frames:
        return games
    adv = pd.concat(adv_frames)
    wide = adv.pivot(index="game_id", columns="side", values=ADV_STATS)
    wide.columns = [f"{side}_{stat}" for stat, side in wide.columns]
    return games.join(wide, on="game_id")


POS_GROUP = {"QB": "qb",
             "T": "ol", "G": "ol", "C": "ol", "OL": "ol", "OT": "ol", "OG": "ol",
             "WR": "skill", "TE": "skill", "RB": "skill", "FB": "skill", "HB": "skill",
             "DE": "def", "DT": "def", "DL": "def", "NT": "def", "LB": "def", "OLB": "def",
             "ILB": "def", "MLB": "def", "CB": "def", "S": "def", "DB": "def", "FS": "def", "SS": "def"}
STATUS_W = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.25}


SNAP_FIRST = 2012          # nflverse snap counts start in 2012, so weighted injuries start in 2013
DEFAULT_IMPORTANCE = 0.15  # snap share assumed for a player with no snap history (e.g. a rookie)


def load_nfl_snaps(ctx: Context, end: int):
    frames = []
    for yr in range(SNAP_FIRST, end + 1):
        def fetch(yr=yr):
            raw = download(f"{NFLVERSE}/snap_counts/snap_counts_{yr}.csv")
            if raw is None:
                return None
            return pd.read_csv(io.BytesIO(raw), usecols=["season", "week", "pfr_player_id", "offense_pct", "defense_pct"])

        d = cached(f"nfl_snap_{yr}.pkl", fetch, refresh=yr >= end)
        if d is not None:
            frames.append(d)

    def fetch_players():
        raw = download(f"{NFLVERSE}/players/players.csv")
        return None if raw is None else pd.read_csv(io.BytesIO(raw), usecols=["gsis_id", "pfr_id"], low_memory=False)

    pl = cached("nfl_players.pkl", fetch_players, refresh=True)   # refresh: new rookies appear every season
    if pl is None:
        pl = cached("nfl_players.pkl", fetch_players)
    if not frames or pl is None:
        warn("snap counts or player-id table unavailable; weighted injuries disabled")
        return
    sn = pd.concat(frames, ignore_index=True).dropna(subset=["pfr_player_id"])
    sn["key"] = sn.season.astype(int) * 100 + sn.week.astype(int)
    sn["share"] = sn[["offense_pct", "defense_pct"]].max(axis=1).fillna(0.0)
    sn = sn.sort_values("key")
    ctx.snap_hist = {pid: (g.key.to_numpy(), g.share.to_numpy()) for pid, g in sn.groupby("pfr_player_id")}
    pl = pl.dropna()
    ctx.gsis2pfr = dict(zip(pl.gsis_id, pl.pfr_id))


def snap_importance(ctx: Context, gsis, key):
    """Average snap share over the player's last 4 games BEFORE the report week (healthy-time role)."""
    h = ctx.snap_hist.get(ctx.gsis2pfr.get(gsis))
    if h is None:
        return DEFAULT_IMPORTANCE
    keys, shares = h
    i = int(np.searchsorted(keys, key, side="left"))
    return float(shares[max(0, i - 4):i].mean()) if i else DEFAULT_IMPORTANCE


PV_DEFAULT = {"skill": 0.2, "def": 0.3, "ol": 0.15}   # value assumed for a player with no history (rookie, backup)
PV_GAMES, PV_SHRINK = 6, 3


def load_nfl_pstats(ctx: Context, end: int):
    """Weekly player stats -> each player's recent production, for quality-weighted injuries."""
    want = ["player_id", "season", "week", "season_type", "position_group", "receiving_epa", "rushing_epa",
            "def_sacks", "def_tackles_for_loss", "def_qb_hits", "def_interceptions", "def_pass_defended",
            "def_tackles_solo", "def_fumbles_forced"]
    frames = []
    for yr in range(2009, end + 1):
        def fetch(yr=yr):
            raw = download(f"{NFLVERSE}/stats_player/stats_player_week_{yr}.csv")
            if raw is None:
                return None
            return pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in want, low_memory=False)
        d = cached(f"nfl_pstat_{yr}.pkl", fetch, refresh=yr >= end)
        if d is not None:
            frames.append(d)
    if not frames:
        warn("no player stats downloaded; quality-weighted injuries disabled")
        return
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.season_type == "REG") & d.position_group.isin(["RB", "WR", "TE", "DL", "LB", "DB"])].copy()
    for c in want[5:]:
        d[c] = pd.to_numeric(d.get(c), errors="coerce").fillna(0.0)
    d["skill"] = d.receiving_epa + d.rushing_epa
    d["defv"] = (d.def_sacks + 0.5 * d.def_tackles_for_loss + 0.25 * d.def_qb_hits + d.def_interceptions
                 + 0.3 * d.def_pass_defended + 0.1 * d.def_tackles_solo + 0.5 * d.def_fumbles_forced)
    d["key"] = d.season.astype(int) * 100 + d.week.astype(int)
    d = d.sort_values("key")
    ctx.pval_hist = {pid: (g.key.to_numpy(), g.skill.to_numpy(), g.defv.to_numpy()) for pid, g in d.groupby("player_id")}


def player_value(ctx: Context, gsis, grp, key):
    """Quality of a player BEFORE the report week: recent per-game production (offensive EPA for skill players,
    a defensive-play score for defenders) or snap share for linemen. Missed games are not counted."""
    if grp == "ol":
        return snap_importance(ctx, gsis, key) if ctx.snap_hist else PV_DEFAULT["ol"]
    h = (ctx.pval_hist or {}).get(gsis)
    if h is None:
        return PV_DEFAULT[grp]
    keys, sk, dv = h
    i = int(np.searchsorted(keys, key, side="left"))
    lo = max(0, i - PV_GAMES)
    n = i - lo
    if n == 0:
        return PV_DEFAULT[grp]
    x = float((sk if grp == "skill" else dv)[lo:i].mean())
    return max(x, 0.0) * n / (n + PV_SHRINK)


def load_nfl_injuries(ctx: Context, end: int):
    frames = []
    for yr in range(2009, end + 1):
        def fetch(yr=yr):
            raw = download(f"{NFLVERSE}/injuries/injuries_{yr}.csv")
            if raw is None:
                return None
            return pd.read_csv(io.BytesIO(raw), low_memory=False,
                               usecols=["season", "week", "team", "gsis_id", "position", "report_status"])

        d = cached(f"nfl_inj_{yr}.pkl", fetch, refresh=yr >= end)
        if d is not None:
            frames.append(d)
    if not frames:
        warn("no injury data downloaded")
        return
    d = pd.concat(frames, ignore_index=True)
    d["team"] = d.team.replace(TEAM_MAP)
    d["w"] = d.report_status.map(STATUS_W).fillna(0.0)
    d["grp"] = d.position.map(POS_GROUP)
    d = d[d.grp.notna() & (d.w > 0)].drop_duplicates(["season", "week", "team", "gsis_id"])
    counts = (d.pivot_table(index=["season", "week", "team"], columns="grp", values="w",
                            aggfunc="sum", fill_value=0.0).reset_index())
    for g in INJ_GROUPS:
        if g not in counts:
            counts[g] = 0.0
    counts["season"] = counts.season.astype(int)
    counts["week"] = counts.week.astype(int)
    ctx.inj_counts = counts
    if ctx.snap_hist:
        dd = d[d.season >= SNAP_FIRST + 1].copy()
        dd["imp"] = [snap_importance(ctx, gs, int(se) * 100 + int(wk))
                     for gs, se, wk in zip(dd.gsis_id, dd.season, dd.week)]
        dd["wi"] = dd.w * dd.imp
        cw = (dd.pivot_table(index=["season", "week", "team"], columns="grp", values="wi",
                             aggfunc="sum", fill_value=0.0).reset_index())
        for g in INJ_GROUPS:
            if g not in cw:
                cw[g] = 0.0
        cw["season"], cw["week"] = cw.season.astype(int), cw.week.astype(int)
        ctx.injw_counts = cw
    if ctx.pval_hist is not None:
        dq = d[(d.season >= 2010) & d.grp.isin(["skill", "ol", "def"])].copy()
        dq["val"] = [player_value(ctx, gs, g_, int(se) * 100 + int(wk))
                     for gs, g_, se, wk in zip(dq.gsis_id, dq.grp, dq.season, dq.week)]
        dq["wq"] = dq.w * dq.val
        cq = (dq.pivot_table(index=["season", "week", "team"], columns="grp", values="wq",
                             aggfunc="sum", fill_value=0.0).reset_index())
        for g in INJ_GROUPS:
            if g not in cq:
                cq[g] = 0.0
        cq["season"], cq["week"] = cq.season.astype(int), cq.week.astype(int)
        ctx.injq_counts = cq
    for r in d[(d.grp == "qb") & (d.w >= 0.75)].itertuples(index=False):
        ctx.qb_out.setdefault((int(r.season), int(r.week), r.team), set()).add(r.gsis_id)


ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ESPN_ABBR = {"WSH": "WAS", "LAR": "LA", "JAX": "JAC"}   # ESPN -> nflverse spellings when they differ
ESPN_OUT = {"out", "injured reserve", "ir", "suspension", "suspended", "pup", "physically unable to perform"}
ESPN_DOUBT = {"doubtful"}
ESPN_Q = {"questionable", "day-to-day"}
STARTER_SLOTS = {"qb": 1, "rb": 1, "te": 1, "wr": 3, "lt": 1, "lg": 1, "c": 1, "rg": 1, "rt": 1,
                 "de": 2, "dt": 2, "nt": 1, "lb": 3, "olb": 2, "ilb": 2, "mlb": 1, "cb": 3, "s": 2, "fs": 1, "ss": 1,
                 "ldt": 1, "rdt": 1, "lde": 1, "rde": 1, "wlb": 1, "slb": 1, "lcb": 1, "rcb": 1, "nb": 1}


def norm_name(n) -> str:
    n = re.sub(r"[^a-z ]", "", str(n or "").lower().replace(".", ""))
    return " ".join(w for w in n.split() if w not in ("jr", "sr", "ii", "iii", "iv"))


def _depth_lists(node, out):
    """Collect {position_key: [player names in depth order]} from ESPN's depth-chart JSON, whatever its nesting."""
    if isinstance(node, dict):
        pos = node.get("positions")
        if isinstance(pos, dict):
            for k, v in pos.items():
                ath = v.get("athletes") if isinstance(v, dict) else None
                if ath:
                    names = [a.get("displayName") or a.get("fullName") for a in ath if isinstance(a, dict)]
                    out.setdefault(str(k).lower(), [n for n in names if n])
        for v in node.values():
            _depth_lists(v, out)
    elif isinstance(node, list):
        for v in node:
            _depth_lists(v, out)


def parse_espn_team(depth_json, injury_list):
    """-> {"qb_depth": [...], "out": [{name,pos,status,slot,starter}], "qb_out": bool}"""
    depth = {}
    try:
        _depth_lists(depth_json, depth)
    except Exception:
        depth = {}
    where = {}                                    # normalized name -> (position key, slot)
    for k, names in depth.items():
        for i, n in enumerate(names):
            key = norm_name(n)
            if key not in where or i + 1 < where[key][1]:
                where[key] = (k, i + 1)
    players = []
    for it in injury_list or []:
        ath = it.get("athlete") or {}
        name = ath.get("displayName") or ath.get("fullName")
        status = str(it.get("status") or (it.get("type") or {}).get("description") or "").strip()
        if not name or not status:
            continue
        pos = str((ath.get("position") or {}).get("abbreviation") or "").upper()
        sl = status.lower()
        tier = "out" if sl in ESPN_OUT else "doubtful" if sl in ESPN_DOUBT else "questionable" if sl in ESPN_Q else None
        if tier is None:
            continue
        pk, slot = where.get(norm_name(name), (None, None))
        starter = bool(slot is not None and slot <= STARTER_SLOTS.get(pk or pos.lower(), 1))
        players.append({"name": name, "pos": pos, "status": status, "tier": tier, "slot": slot, "starter": starter})
    qbd = depth.get("qb", [])
    out_names = {norm_name(p["name"]) for p in players if p["tier"] in ("out", "doubtful")}
    return {"qb_depth": qbd, "out": players,
            "qb_out": bool(qbd) and norm_name(qbd[0]) in out_names}


def fetch_espn_report():
    """Current injuries + depth charts for all NFL teams from ESPN's public JSON. {} if unreachable."""
    try:
        def get(url):
            r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return r.json()
        teams = {}
        for t in get(f"{ESPN_SITE}/teams?limit=40")["sports"][0]["leagues"][0]["teams"]:
            t = t["team"]
            ab = ESPN_ABBR.get(t["abbreviation"], t["abbreviation"])
            teams[str(t["id"])] = TEAM_MAP.get(ab, ab)
        inj = {}
        for e in get(f"{ESPN_SITE}/injuries").get("injuries", []):
            inj[str(e.get("id"))] = e.get("injuries", [])
        rep = {}
        for tid, ab in teams.items():
            try:
                dj = get(f"{ESPN_SITE}/teams/{tid}/depthcharts")
            except Exception as e:
                warn(f"ESPN depth chart for {ab} unavailable: {e}")
                dj = {}
            rep[ab] = parse_espn_team(dj, inj.get(tid, []))
        n_out = sum(len(v["out"]) for v in rep.values())
        n_dep = sum(1 for v in rep.values() if v["qb_depth"])
        print(f"  ESPN report: {n_out} injured players, depth charts for {n_dep}/{len(rep)} teams", file=sys.stderr)
        return rep
    except Exception as e:
        warn(f"ESPN injury/depth data unavailable ({e}); using nflverse report only")
        return {}


def apply_forecasts(games: pd.DataFrame, days: int) -> pd.DataFrame:
    """Fill temp/wind for upcoming outdoor NFL games from the free Open-Meteo forecast API."""
    today = pd.Timestamp.today().normalize()
    need = (games.home_pts.isna() & games.temp.isna()
            & ~games.roof.isin(["dome", "closed"])
            & (games.date >= today - pd.Timedelta(days=1))
            & (games.date <= today + pd.Timedelta(days=min(days, 15))))
    cache, failed, n_ok = {}, set(), 0
    for c in ("gust", "pop"):
        if c not in games:
            games[c] = np.nan

    def fetch(home):
        lat, lon = NFL_STADIUMS[home]
        last = None
        for _try in range(2):                      # one retry for a flaky request
            try:
                r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=20, params=dict(
                    latitude=lat, longitude=lon,
                    hourly="temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability",
                    temperature_unit="fahrenheit", wind_speed_unit="mph",
                    timezone="UTC", forecast_days=16))
                r.raise_for_status()
                h = r.json()["hourly"]
                arr = lambda k: np.array([np.nan if v is None else v for v in h.get(k, [np.nan] * len(h["time"]))], float)
                return (pd.to_datetime(h["time"]), arr("temperature_2m"), arr("wind_speed_10m"),
                        arr("wind_gusts_10m"), arr("precipitation_probability"))
            except Exception as e:
                last = e
        raise last

    for idx in games.index[need]:
        g = games.loc[idx]
        if g.home not in NFL_STADIUMS or g.home in failed:
            continue
        try:
            if g.home not in cache:
                cache[g.home] = fetch(g.home)
            times, temps, winds, gusts, pops = cache[g.home]
            gt = g.gametime if isinstance(g.gametime, str) and ":" in g.gametime else "13:00"
            ko = (pd.Timestamp(f"{g.date:%Y-%m-%d} {gt}", tz="America/New_York")
                  .tz_convert("UTC").tz_localize(None))
            i = int(np.abs((times - ko).total_seconds()).argmin())
            games.loc[idx, "temp"], games.loc[idx, "wind"] = temps[i], winds[i]
            games.loc[idx, "gust"], games.loc[idx, "pop"] = gusts[i], pops[i]
            n_ok += 1
        except Exception as e:  # one team failing no longer stops the rest; that game falls back to climatology
            failed.add(g.home)
            warn(f"forecast failed for {g.home}: {e}")
    if need.any():
        print(f"  weather forecasts applied to {n_ok}/{int(need.sum())} upcoming outdoor games",
              file=sys.stderr)
    return games


# --------------------------------------------------------------------------- #
# CFB data
# --------------------------------------------------------------------------- #


def load_cfb(start: int, end: int, key: str, ctx: Context) -> pd.DataFrame:
    rows = []
    for yr in range(start, end + 1):
        def fetch(yr=yr):
            print(f"  downloading CFB games {yr}", file=sys.stderr)
            return cfbd_get("/games", {"year": yr, "seasonType": "both"}, key)

        for g in cached(f"cfb_games_{yr}.pkl", fetch, refresh=yr >= end):
            hc = _get(g, "homeClassification", "home_classification")
            ac = _get(g, "awayClassification", "away_classification")
            if hc != "fbs" and ac != "fbs":
                continue
            rows.append({
                "game_id": _get(g, "id", "gameId"),
                "date": _get(g, "startDate", "start_date"),
                "season": _get(g, "season"), "week": _get(g, "week"),
                "home": _get(g, "homeTeam", "home_team"), "away": _get(g, "awayTeam", "away_team"),
                "home_pts": _get(g, "homePoints", "home_points"),
                "away_pts": _get(g, "awayPoints", "away_points"),
                "neutral": bool(_get(g, "neutralSite", "neutral_site") or False),
                "home_class": hc or "other", "away_class": ac or "other",
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["date", "home", "away"])
    df["season"] = df["season"].astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")
    df = df.reset_index(drop=True)

    df = _cfb_advanced(df, start, end, key)
    df = _cfb_weather(df, start, end, key)
    df = _cfb_lines(df, start, end, key)
    _cfb_roster(ctx, start, end, key)
    return df


CFB_BOOKS = ("consensus", "DraftKings", "ESPN Bet", "Bovada", "teamrankings")   # preferred order


def pick_cfb_line(lines):
    """One book's line from CFBD's per-game list: first preferred book that has a spread, else the
    median across books.  -> (home_spread (+ = home favored), total, ml_home, ml_away) or Nones."""
    def num(x):
        try:
            v = float(x)
            return None if v != v else v
        except (TypeError, ValueError):
            return None
    ok = [l for l in (lines or []) if num(l.get("spread")) is not None]
    if not ok:
        return None, None, None, None
    by = {str(l.get("provider")): l for l in ok}
    chosen = next((by[b] for b in CFB_BOOKS if b in by), None)
    if chosen is not None:
        sp = -num(chosen["spread"])                         # CFBD: negative = home favored
        tot, mh = num(_get(chosen, "overUnder", "over_under")), num(_get(chosen, "homeMoneyline", "home_moneyline"))
        ma = num(_get(chosen, "awayMoneyline", "away_moneyline"))
        return sp, tot, mh, ma
    med = lambda v: float(np.median(v)) if v else None
    sp = -med([num(l["spread"]) for l in ok])
    tot = med([x for x in (num(_get(l, "overUnder", "over_under")) for l in ok) if x is not None])
    return sp, tot, None, None


def _cfb_lines(df, start, end, key):
    """Betting lines from CFBD /lines (spread, total, moneylines) -> mkt_* columns."""
    recs = []
    for yr in range(start, end + 1):
        def fetch(yr=yr):
            print(f"  downloading CFB betting lines {yr}", file=sys.stderr)
            return cfbd_get("/lines", {"year": yr, "seasonType": "both"}, key, optional=True)
        for g in cached(f"cfb_lines_{yr}.pkl", fetch, refresh=yr >= end) or []:
            sp, tot, mh, ma = pick_cfb_line(g.get("lines"))
            if sp is not None:
                recs.append({"game_id": _get(g, "id", "gameId"), "mkt_spread": sp, "mkt_total": tot,
                             "mkt_ml_home": mh, "mkt_ml_away": ma})
    if not recs:
        warn("no CFB betting lines available; CFB plays need them")
        return df
    ln = pd.DataFrame(recs).dropna(subset=["game_id"]).drop_duplicates("game_id")
    return df.drop(columns=["mkt_spread", "mkt_total", "mkt_ml_home", "mkt_ml_away"], errors="ignore") \
             .merge(ln, on="game_id", how="left")


def _cfb_advanced(df, start, end, key):
    recs = []
    for yr in range(start, end + 1):
        def fetch(yr=yr):
            print(f"  downloading CFB advanced stats {yr}", file=sys.stderr)
            return cfbd_get("/stats/game/advanced", {"year": yr}, key, optional=True)

        data = cached(f"cfb_adv_{yr}.pkl", fetch, refresh=yr >= end)
        for r in data or []:
            off = r.get("offense") or {}
            recs.append({"game_id": _get(r, "gameId", "game_id"), "team": r.get("team"),
                         "epa": off.get("ppa"), "succ": off.get("successRate"),
                         "pass_epa": (off.get("passingPlays") or {}).get("ppa"),
                         "rush_epa": (off.get("rushingPlays") or {}).get("ppa")})
    if not recs:
        warn("no CFB advanced stats; 'advanced' group disabled")
        return df
    adv = pd.DataFrame(recs).dropna(subset=["game_id", "team"]).drop_duplicates(["game_id", "team"])
    for side, tcol in (("h", "home"), ("a", "away")):
        sub = adv.rename(columns={s: f"{side}_{s}" for s in ADV_STATS}).rename(columns={"team": tcol})
        df = df.merge(sub, on=["game_id", tcol], how="left")
    return df


def _cfb_weather(df, start, end, key):
    recs = []
    for yr in range(start, end + 1):
        def fetch(yr=yr):
            return cfbd_get("/games/weather", {"year": yr}, key, optional=True)

        data = cached(f"cfb_wx_{yr}.pkl", fetch, refresh=yr >= end)
        if data is None:
            return df  # Patreon-only endpoint; stop trying
        for r in data:
            recs.append({"game_id": _get(r, "id", "gameId"),
                         "roof": "dome" if _get(r, "gameIndoors", "game_indoors") else "outdoors",
                         "temp": _get(r, "temperature"), "wind": _get(r, "windSpeed", "wind_speed")})
    if not recs:
        return df
    wx = pd.DataFrame(recs).dropna(subset=["game_id"]).drop_duplicates("game_id")
    wx["temp"] = pd.to_numeric(wx["temp"], errors="coerce")
    wx["wind"] = pd.to_numeric(wx["wind"], errors="coerce")
    return df.drop(columns=["roof", "temp", "wind"], errors="ignore").merge(wx, on="game_id", how="left")


def _cfb_roster(ctx: Context, start, end, key):
    for yr in range(start, end + 1):
        def get(path, name, yr=yr):
            def fetch():
                return cfbd_get(path, {"year": yr}, key, optional=True)
            return cached(f"cfb_{name}_{yr}.pkl", fetch, refresh=yr >= end)

        ret, portal, talent = get("/player/returning", "ret"), get("/player/portal", "portal"), \
            get("/talent", "talent")
        table = {}

        if ret:
            d = pd.DataFrame([{"team": r.get("team"),
                               "ret_ppa": pd.to_numeric(_get(r, "percentPPA", "percent_ppa"), errors="coerce"),
                               "ret_pass": pd.to_numeric(_get(r, "percentPassingPPA", "percent_passing_ppa"),
                                                         errors="coerce")} for r in ret]).dropna(subset=["team"])
            for c in ("ret_ppa", "ret_pass"):  # API may return percent (0-100) or fraction
                if d[c].max() > 1.5:
                    d[c] = d[c] / 100.0
            for r in d.itertuples(index=False):
                table.setdefault(r.team, {}).update(ret_ppa=r.ret_ppa, ret_pass=r.ret_pass)
            ctx.season_means[yr] = dict(ret_ppa=float(d.ret_ppa.mean()), ret_pass=float(d.ret_pass.mean()),
                                        portal_net=0.0)

        if portal:
            p = pd.DataFrame([{"origin": r.get("origin"), "dest": r.get("destination"),
                               "rating": pd.to_numeric(r.get("rating"), errors="coerce")} for r in portal])
            p["val"] = p.rating.fillna(0.75)
            net = (p.dropna(subset=["dest"]).groupby("dest").val.sum()
                   .sub(p.dropna(subset=["origin"]).groupby("origin").val.sum(), fill_value=0.0))
            for team, v in net.items():
                table.setdefault(team, {})["portal_net"] = float(v)

        if talent:
            t = pd.DataFrame([{"team": _get(r, "school", "team"),
                               "talent": pd.to_numeric(r.get("talent"), errors="coerce")} for r in talent])
            t = t.dropna()
            if len(t) > 5 and t.talent.std() > 0:
                z = (t.talent - t.talent.mean()) / t.talent.std()
                for team, zz in zip(t.team, z):
                    table.setdefault(team, {})["talent_z"] = float(zz)

        m = ctx.season_means.get(yr, ctx.global_means)
        for team, v in table.items():
            full = dict(ret_ppa=m["ret_ppa"], ret_pass=m["ret_pass"], portal_net=0.0, talent_z=0.0)
            full.update({k: x for k, x in v.items() if x is not None and not pd.isna(x)})
            ctx.team_season[(yr, team)] = full
            ctx.talent_z[(yr, team)] = full["talent_z"]


# --------------------------------------------------------------------------- #
# Feature engineering (strictly pre-game)
# --------------------------------------------------------------------------- #


class TeamState:
    def __init__(self, base_elo, window, adv_window):
        self.base0 = base_elo
        self.base = base_elo
        self.elo = base_elo
        self.season = None
        self.last_date = None
        self.pf = deque(maxlen=window)
        self.pa = deque(maxlen=window)
        self.ewm_pf = None
        self.ewm_pa = None
        self.season_g = 0
        self.off = {s: deque(maxlen=adv_window) for s in ADV_STATS}
        self.dfn = {s: deque(maxlen=adv_window) for s in ADV_STATS}  # stats ALLOWED by the defense
        self.aoff = {s: deque(maxlen=adv_window) for s in ADV_STATS}   # opponent-adjusted offense
        self.adef = {s: deque(maxlen=adv_window) for s in ADV_STATS}   # opponent-adjusted stats allowed
        self.apf = deque(maxlen=adv_window)
        self.apa = deque(maxlen=adv_window)
        self.tcom = deque(maxlen=16)   # turnovers committed per game
        self.tfor = deque(maxlen=16)   # turnovers forced per game
        self.last_qb = None


class Builder:
    WINDOW = 5
    SHRINK = 2.0
    EWM_ALPHA = 0.25
    ADV_WINDOW = 8
    ADV_SHRINK = 3.0
    QB_K = 200.0          # pseudo-dropbacks of prior
    QB_PRIOR = -0.05      # EPA/dropback prior for an unproven QB
    BACKUP_VAL = -0.12    # assumed value of an unknown replacement QB
    TOV_SHRINK = 8.0      # pseudo-games of league-average turnovers (turnover luck rarely repeats)

    def __init__(self, lg: League, ctx: Context, groups: set, talent_elo: bool = True):
        self.lg, self.ctx, self.groups = lg, ctx, groups
        self.talent_elo = talent_elo and lg.talent_scale > 0 and bool(ctx.talent_z)
        self.teams = {}
        self.pts_sum, self.pts_n = lg.default_pts * 50, 50
        self.adv_sum = {s: 0.0 for s in ADV_STATS}
        self.adv_cnt = {s: 0 for s in ADV_STATS}
        self.qb_stats = {}  # qb_id -> [n, epa_sum]
        self.tov_sum, self.tov_n = 1.3 * 50, 50
        self.tov_map = {}
        if ctx.box is not None:
            self.tov_map = {(r.game_id, r.side): r.tov for r in ctx.box.itertuples(index=False)}

    @property
    def avg(self):
        return self.pts_sum / self.pts_n

    def adv_mean(self, s):
        return self.adv_sum[s] / self.adv_cnt[s] if self.adv_cnt[s] else 0.0

    # ---- team bookkeeping
    def _team(self, name, season, cls):
        t = self.teams.get(name)
        if t is None:
            t = TeamState(1500.0 if cls in ("nfl", "fbs") else 1300.0, self.WINDOW, self.ADV_WINDOW)
            self.teams[name] = t
            first = True
        else:
            first = False
        if t.season != season:
            if self.talent_elo:
                t.base = t.base0 + self.lg.talent_scale * self.ctx.talent_z.get((season, name), 0.0)
            if first:
                t.elo = t.base
            elif t.season is not None:
                t.elo = (1 - self.lg.regress) * t.elo + self.lg.regress * t.base
            t.season = season
            t.season_g = 0
        return t

    def _roll(self, t):
        n, k, avg = len(t.pf), self.SHRINK, self.avg
        pf = (sum(t.pf) + k * avg) / (n + k)
        pa = (sum(t.pa) + k * avg) / (n + k)
        return (pf, pa, t.ewm_pf if t.ewm_pf is not None else avg,
                t.ewm_pa if t.ewm_pa is not None else avg)

    @staticmethod
    def _rest(t, date):
        return 14.0 if t.last_date is None else float(min((date - t.last_date).days, 14))

    # ---- feature groups
    def _base(self, g, ht, at):
        hpf, hpa, hef, hea = self._roll(ht)
        apf, apa, aef, aea = self._roll(at)
        hfa = 0.0 if g.neutral else self.lg.hfa
        hr, ar = self._rest(ht, g.date), self._rest(at, g.date)
        f = {"elo_diff": ht.elo + hfa - at.elo, "home_elo": ht.elo, "away_elo": at.elo,
             "is_neutral": float(g.neutral),
             "home_pf": hpf, "home_pa": hpa, "away_pf": apf, "away_pa": apa,
             "home_ewm_pf": hef, "home_ewm_pa": hea, "away_ewm_pf": aef, "away_ewm_pa": aea,
             "exp_home_pts": (hpf + apa) / 2, "exp_away_pts": (apf + hpa) / 2,
             "home_rest": hr, "away_rest": ar, "rest_diff": hr - ar,
             "home_szn_games": float(ht.season_g), "away_szn_games": float(at.season_g)}
        f["exp_margin"] = f["exp_home_pts"] - f["exp_away_pts"]
        f["exp_total"] = f["exp_home_pts"] + f["exp_away_pts"]
        return f

    def _shrunk(self, dq, s):
        n, k = len(dq), self.ADV_SHRINK
        return (sum(dq) + k * self.adv_mean(s)) / (n + k)

    def _adv(self, ht, at):
        f = {}
        for sd, t in (("home", ht), ("away", at)):
            for s in ADV_STATS:
                f[f"{sd}_off_{s}"] = self._shrunk(t.off[s], s)
                f[f"{sd}_def_{s}"] = self._shrunk(t.dfn[s], s)
        for s in ADV_STATS:
            f[f"edge_{s}"] = (f[f"home_off_{s}"] + f[f"away_def_{s}"]) - \
                             (f[f"away_off_{s}"] + f[f"home_def_{s}"])
        return f

    def _adj(self, ht, at):
        f = {}
        for sd, t in (("home", ht), ("away", at)):
            for s in ADV_STATS:
                f[f"{sd}_adjoff_{s}"] = self._shrunk(t.aoff[s], s)
                f[f"{sd}_adjdef_{s}"] = self._shrunk(t.adef[s], s)
        for s in ADV_STATS:
            f[f"edge_adj_{s}"] = (f[f"home_adjoff_{s}"] + f[f"away_adjdef_{s}"]) - \
                                 (f[f"away_adjoff_{s}"] + f[f"home_adjdef_{s}"])
        avg, k = self.avg, self.SHRINK

        def lvl(dq):
            return (sum(dq) + k * avg) / (len(dq) + k)
        f["home_adj_pf"], f["home_adj_pa"] = lvl(ht.apf), lvl(ht.apa)
        f["away_adj_pf"], f["away_adj_pa"] = lvl(at.apf), lvl(at.apa)
        he = (f["home_adj_pf"] + f["away_adj_pa"]) / 2
        ae = (f["away_adj_pf"] + f["home_adj_pa"]) / 2
        f["exp_adj_margin"], f["exp_adj_total"] = he - ae, he + ae
        return f

    def _tov(self, ht, at):
        m, k = self.tov_sum / self.tov_n, self.TOV_SHRINK

        def lv(dq):
            return (sum(dq) + k * m) / (len(dq) + k)
        f = {"home_tov_comm": lv(ht.tcom), "home_tov_forced": lv(ht.tfor),
             "away_tov_comm": lv(at.tcom), "away_tov_forced": lv(at.tfor)}
        f["tov_edge"] = (f["home_tov_forced"] - f["home_tov_comm"]) - (f["away_tov_forced"] - f["away_tov_comm"])
        return f

    def _qb_value(self, qb):
        if qb is None:
            return self.BACKUP_VAL, 0.0
        n, s = self.qb_stats.get(qb, (0.0, 0.0))
        return (s + self.QB_K * self.QB_PRIOR) / (n + self.QB_K), n

    def _qb(self, g, ht, at):
        f = {}
        for sd, t, team, cur in (("home", ht, g.home, g.home_qb_id), ("away", at, g.away, g.away_qb_id)):
            rep = self.ctx.espn.get(team)
            upcoming = pd.isna(g.home_pts)
            if pd.isna(cur) or (upcoming and rep and rep["qb_depth"]):   # starter unknown, or ESPN knows better
                if rep and rep["qb_depth"]:
                    # ESPN depth chart: first QB on the chart who isn't Out/Doubtful/IR
                    outn = {norm_name(p["name"]) for p in rep["out"] if p["tier"] in ("out", "doubtful")}
                    nm = next((n for n in rep["qb_depth"] if norm_name(n) not in outn), None)
                    cur = self.ctx.qb_name_id.get(norm_name(nm)) if nm else None
                else:             # fall back: assume last starter unless the nflverse report has him Out/Doubtful
                    cur = t.last_qb
                    if cur is not None and cur in self.ctx.qb_out.get((g.season, g.week, team), ()):
                        cur = None
            val, n = self._qb_value(cur)
            prev = self._qb_value(t.last_qb)[0] if t.last_qb is not None else val
            f[f"{sd}_qb_val"] = val
            f[f"{sd}_qb_delta"] = 0.0 if cur == t.last_qb else val - prev
            f[f"{sd}_qb_n"] = min(n, 1000.0) / 1000.0
        f["qb_edge"] = f["home_qb_val"] - f["away_qb_val"]
        return f

    def _roster(self, g, ht, at):
        h = self.ctx.roster(g.season, g.home, g.home_class)
        a = self.ctx.roster(g.season, g.away, g.away_class)
        early = max(0.0, 1.0 - 0.5 * (ht.season_g + at.season_g) / 6.0)
        f = {"home_ret_ppa": h["ret_ppa"], "away_ret_ppa": a["ret_ppa"],
             "home_ret_pass": h["ret_pass"], "away_ret_pass": a["ret_pass"],
             "home_portal_net": h["portal_net"], "away_portal_net": a["portal_net"],
             "home_talent": h["talent_z"], "away_talent": a["talent_z"],
             "ret_diff": h["ret_ppa"] - a["ret_ppa"], "ret_pass_diff": h["ret_pass"] - a["ret_pass"],
             "portal_diff": h["portal_net"] - a["portal_net"], "talent_diff": h["talent_z"] - a["talent_z"]}
        f["ret_diff_early"] = f["ret_diff"] * early
        f["ret_pass_diff_early"] = f["ret_pass_diff"] * early
        f["portal_diff_early"] = f["portal_diff"] * early
        return f

    # ---- updates after a game is played
    def _update(self, g, ht, at):
        hp, ap = float(g.home_pts), float(g.away_pts)
        hfa = 0.0 if g.neutral else self.lg.hfa
        diff = ht.elo + hfa - at.elo
        exp_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        actual = 1.0 if hp > ap else (0.0 if hp < ap else 0.5)
        winner_diff = diff if hp > ap else -diff
        mult = math.log(abs(hp - ap) + 1) * 2.2 / max(0.5, winner_diff * 0.001 + 2.2)
        delta = self.lg.k * mult * (actual - exp_home)
        ht.elo += delta
        at.elo -= delta

        if "adjusted" in self.groups:   # compare each result with what the opponent usually allows / scores
            avg = self.avg
            hpf_l, hpa_l = self._roll(ht)[:2]
            apf_l, apa_l = self._roll(at)[:2]
            ht.apf.append(hp - (apa_l - avg))
            ht.apa.append(ap - (apf_l - avg))
            at.apf.append(ap - (hpa_l - avg))
            at.apa.append(hp - (hpf_l - avg))
        a = self.EWM_ALPHA
        for t, pf, pa in ((ht, hp, ap), (at, ap, hp)):
            t.pf.append(pf)
            t.pa.append(pa)
            t.ewm_pf = pf if t.ewm_pf is None else (1 - a) * t.ewm_pf + a * pf
            t.ewm_pa = pa if t.ewm_pa is None else (1 - a) * t.ewm_pa + a * pa
            t.season_g += 1
            t.last_date = g.date
        self.pts_sum += hp + ap
        self.pts_n += 2

        if "advanced" in self.groups:
            for s in ADV_STATS:
                hv, av = getattr(g, f"h_{s}"), getattr(g, f"a_{s}")
                if "adjusted" in self.groups:   # all reads happen before this game's raw stats are appended
                    m = self.adv_mean(s)
                    if not pd.isna(hv):
                        ht.aoff[s].append(hv - (self._shrunk(at.dfn[s], s) - m))
                        at.adef[s].append(hv - (self._shrunk(ht.off[s], s) - m))
                    if not pd.isna(av):
                        at.aoff[s].append(av - (self._shrunk(ht.dfn[s], s) - m))
                        ht.adef[s].append(av - (self._shrunk(at.off[s], s) - m))
                if not pd.isna(hv):
                    ht.off[s].append(hv)
                    at.dfn[s].append(hv)
                    self.adv_sum[s] += hv
                    self.adv_cnt[s] += 1
                if not pd.isna(av):
                    at.off[s].append(av)
                    ht.dfn[s].append(av)
                    self.adv_sum[s] += av
                    self.adv_cnt[s] += 1

        if "turnovers" in self.groups:
            th, ta = self.tov_map.get((g.game_id, "h")), self.tov_map.get((g.game_id, "a"))
            if th is not None and ta is not None:
                ht.tcom.append(th)
                ht.tfor.append(ta)
                at.tcom.append(ta)
                at.tfor.append(th)
                self.tov_sum += th + ta
                self.tov_n += 2

        if "qb" in self.groups:
            for qb, n, e in self.ctx.qb_log.get(g.game_id, ()):
                rec = self.qb_stats.setdefault(qb, [0.0, 0.0])
                rec[0] += n
                rec[1] += e
            if not pd.isna(g.home_qb_id):
                ht.last_qb = g.home_qb_id
            if not pd.isna(g.away_qb_id):
                at.last_qb = g.away_qb_id

    def run(self, games: pd.DataFrame) -> pd.DataFrame:
        feats = []
        for g in games.itertuples(index=False):
            ht = self._team(g.home, g.season, g.home_class)
            at = self._team(g.away, g.season, g.away_class)
            f = self._base(g, ht, at)
            if "advanced" in self.groups:
                f.update(self._adv(ht, at))
            if "adjusted" in self.groups:
                f.update(self._adj(ht, at))
            if "turnovers" in self.groups:
                f.update(self._tov(ht, at))
            if "qb" in self.groups:
                f.update(self._qb(g, ht, at))
            if "roster" in self.groups:
                f.update(self._roster(g, ht, at))
            feats.append(f)
            if not (pd.isna(g.home_pts) or pd.isna(g.away_pts)):
                self._update(g, ht, at)
        return pd.DataFrame(feats, index=games.index)


def injury_features(games: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for sd in ("home", "away"):
        m = games[["season", "week", sd]].merge(
            inj, how="left", left_on=["season", "week", sd], right_on=["season", "week", "team"])
        for g in INJ_GROUPS:
            out[f"{sd}_inj_{g}"] = m[g].fillna(0.0).to_numpy()
    out = pd.DataFrame(out, index=games.index)
    out["inj_total_diff"] = (out[[f"home_inj_{g}" for g in INJ_GROUPS]].sum(axis=1)
                             - out[[f"away_inj_{g}" for g in INJ_GROUPS]].sum(axis=1))
    return out


def injw_features(games: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for sd in ("home", "away"):
        m = games[["season", "week", sd]].merge(
            inj, how="left", left_on=["season", "week", sd], right_on=["season", "week", "team"])
        cov = (games.season >= SNAP_FIRST + 1).to_numpy()      # earlier seasons have no snap history -> unknown
        for g in INJ_GROUPS:
            v = m[g].to_numpy(dtype=float)
            out[f"{sd}_injw_{g}"] = np.where(cov, np.nan_to_num(v, nan=0.0), np.nan)
    out = pd.DataFrame(out, index=games.index)
    out["injw_total_diff"] = (out[[f"home_injw_{g}" for g in INJ_GROUPS]].sum(axis=1, min_count=1)
                              - out[[f"away_injw_{g}" for g in INJ_GROUPS]].sum(axis=1, min_count=1))
    return out


INJQ_COLS = [f"{sd}_injq_{g}" for sd in ("home", "away") for g in ("skill", "ol", "def")] + ["injq_total_diff"]


def injq_features(games: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    out = {}
    cov = (games.season >= 2010).to_numpy()
    for sd in ("home", "away"):
        m = games[["season", "week", sd]].merge(
            inj, how="left", left_on=["season", "week", sd], right_on=["season", "week", "team"])
        for g in ("skill", "ol", "def"):
            v = m[g].to_numpy(dtype=float)
            out[f"{sd}_injq_{g}"] = np.where(cov, np.nan_to_num(v, nan=0.0), np.nan)
    out = pd.DataFrame(out, index=games.index)
    out["injq_total_diff"] = (out[[f"home_injq_{g}" for g in ("skill", "ol", "def")]].sum(axis=1, min_count=1)
                              - out[[f"away_injq_{g}" for g in ("skill", "ol", "def")]].sum(axis=1, min_count=1))
    return out


def weather_features(games: pd.DataFrame):
    roof = games["roof"].astype("object")
    indoor = roof.isin(["dome", "closed"])
    temp = pd.to_numeric(games["temp"], errors="coerce")
    wind = pd.to_numeric(games["wind"], errors="coerce")
    known = (temp.notna() & wind.notna()) | indoor
    if known.mean() < 0.2:
        return None
    out_mask = ~indoor
    month = games["date"].dt.month
    month_temp = temp[out_mask].groupby(month[out_mask]).mean()  # climatology fallback
    t = temp.copy()
    t[indoor] = 70.0
    miss = t.isna()
    t[miss] = month[miss].map(month_temp).fillna(60.0)
    w = wind.copy()
    w[indoor] = 0.0
    w = w.fillna(float(wind[out_mask].median()) if wind[out_mask].notna().any() else 7.0)
    return pd.DataFrame({"is_dome": indoor.astype(float), "temp": t, "wind": w,
                         "cold": (45.0 - t).clip(lower=0), "windy": (w - 10.0).clip(lower=0),
                         "weather_known": known.astype(float)}, index=games.index)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def make_model(kind: str):
    ridge = make_pipeline(SimpleImputer(), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 20)))
    gbm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=300,
                                        min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    if kind == "ridge":
        return ridge
    if kind == "gbm":
        return gbm
    return VotingRegressor([("ridge", ridge), ("gbm", gbm)])


class ScorePredictor:
    """Predicts margin (home - away) and total points, then derives scores."""

    def __init__(self, kind: str, default_sigma: float):
        self.kind, self.sigma = kind, default_sigma

    def fit(self, X, margin, total):
        n = len(X)
        if n >= 400:  # out-of-sample residual std from a time-ordered holdout
            cut = int(n * 0.8)
            m = make_model(self.kind).fit(X.iloc[:cut], margin.iloc[:cut])
            self.sigma = float(np.std(margin.iloc[cut:] - m.predict(X.iloc[cut:])))
        self.margin_model = make_model(self.kind).fit(X, margin)
        self.total_model = make_model(self.kind).fit(X, total)
        return self

    def predict(self, X) -> pd.DataFrame:
        m, t = self.margin_model.predict(X), self.total_model.predict(X)
        return pd.DataFrame({"pred_margin": m, "pred_total": t, "pred_home": (t + m) / 2,
                             "pred_away": (t - m) / 2, "home_win_prob": norm.cdf(m / self.sigma)},
                            index=X.index)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


MANUAL_ALIAS = {"WSH": "WAS", "LAR": "LA", "JAX": "JAC", "LVR": "LV", "OAK": "LV", "SD": "LAC", "STL": "LA"}


def apply_manual_lines(games, path="data/manual_lines.json"):
    """Lines the site owner pasted from another source (e.g. Action Network) for UNPLAYED games.
    Each entry: {"home","away","favorite","spread" (points the favorite lays, e.g. 3.5),"total","ml_home","ml_away",
    "week"?,"season"?,"source"?}. Only the fields given are replaced. Finished games are never touched."""
    if not os.path.exists(path):
        return games
    try:
        rows = json.load(open(path))
    except Exception as e:
        warn(f"manual lines file unreadable ({e}); ignoring it")
        return games
    ab = lambda t: MANUAL_ALIAS.get(str(t).strip().upper(), str(t).strip().upper())
    hit = 0
    for r in rows:
        h, a = ab(r.get("home")), ab(r.get("away"))
        m = (games.home == h) & (games.away == a) & games.home_pts.isna()
        if r.get("week") is not None:
            m &= games.week == r["week"]
        if r.get("season") is not None:
            m &= games.season == r["season"]
        if m.sum() != 1:
            warn(f"manual line for {a} @ {h} matched {int(m.sum())} unplayed games; skipped")
            continue
        i = games.index[m][0]
        if r.get("spread") is not None:
            fav, sp = ab(r.get("favorite", "")), abs(float(r["spread"]))
            if sp and fav not in (h, a):
                warn(f"manual line for {a} @ {h}: favorite '{r.get('favorite')}' is neither team; skipped")
                continue
            games.loc[i, "mkt_spread"] = 0.0 if not sp else (sp if fav == h else -sp)
        if r.get("total") is not None:
            games.loc[i, "mkt_total"] = float(r["total"])
        if r.get("ml_home") is not None:
            games.loc[i, "mkt_ml_home"] = float(r["ml_home"])
        if r.get("ml_away") is not None:
            games.loc[i, "mkt_ml_away"] = float(r["ml_away"])
        hit += 1
    print(f"  manual lines applied to {hit}/{len(rows)} games ({path})", file=sys.stderr)
    return games


def prepare(args, forecast_days=None):
    lg = LEAGUES[args.league]
    start = args.start or lg.default_start
    end = current_season()
    ctx = Context()
    if args.league == "nfl":
        games = load_nfl(start)
        games = load_nfl_pbp(games, ctx, start, end)
        ws_ = getattr(args, "with_set", set())
        if "injw" in ws_ or "injq" in ws_:   # only needed when weighted / quality-weighted injuries are enabled
            load_nfl_snaps(ctx, end)
        if "injq" in ws_:
            load_nfl_pstats(ctx, end)
        load_nfl_injuries(ctx, end)
        nm = pd.concat([games[["home_qb_name", "home_qb_id"]].set_axis(["n", "i"], axis=1),
                        games[["away_qb_name", "away_qb_id"]].set_axis(["n", "i"], axis=1)]).dropna()
        ctx.qb_name_id = {norm_name(n): i for n, i in zip(nm.n, nm.i)}   # later starts overwrite earlier
        if forecast_days is not None:
            ctx.espn = fetch_espn_report()
            games = apply_forecasts(games.sort_values("date").reset_index(drop=True), forecast_days)
    else:
        key = os.environ.get("CFBD_API_KEY")
        if not key:
            sys.exit("Set CFBD_API_KEY (free key: https://collegefootballdata.com/key)")
        games = load_cfb(start, end, key, ctx)
    for c in GAME_COLS:
        if c not in games:
            games[c] = np.nan
    games = games.sort_values("date", kind="stable").reset_index(drop=True)
    games = apply_manual_lines(games)
    return lg, games, ctx


EPAR_K, EPAR_REG, EPAR_PLAYS = 0.12, 0.6, 65.0     # update speed, offseason carry-over, plays per game (EPA -> points)
EPAR_COLS = [f"{sd}_epar_{k}" for sd in ("home", "away") for k in ("off", "def")] + ["epar_diff", "epar_total"]


def epar_features(games):
    """Opponent-adjusted EPA power rating, updated game by game (pre-game values only, so no leakage).
    A team's offense is rated by how much more EPA/play it gets than expected from the defense it faced, and its
    defense likewise. Expressed in points per game."""
    g = games.sort_values("date", kind="stable")
    off, dfn, season, out, n, sm = {}, {}, {}, {}, 0, 0.0
    for i, x in g.iterrows():
        for t in (x.home, x.away):
            if season.get(t) != x.season:
                if t in season:
                    off[t] *= EPAR_REG
                    dfn[t] *= EPAR_REG
                else:
                    off[t] = dfn[t] = 0.0
                season[t] = x.season
        mu = sm / n if n else 0.0
        oh, dh, oa, da = off[x.home], dfn[x.home], off[x.away], dfn[x.away]
        P = EPAR_PLAYS
        out[i] = (oh * P, dh * P, oa * P, da * P, ((oh - dh) - (oa - da)) * P, (oh + oa + dh + da) * P)
        if pd.notna(x.h_epa) and pd.notna(x.a_epa):          # learn only from games already played
            eh, ea = x.h_epa - (mu + oh + da), x.a_epa - (mu + oa + dh)
            off[x.home] += EPAR_K * eh
            dfn[x.away] += EPAR_K * eh
            off[x.away] += EPAR_K * ea
            dfn[x.home] += EPAR_K * ea
            sm += x.h_epa + x.a_epa
            n += 2
    return pd.DataFrame.from_dict(out, orient="index", columns=EPAR_COLS)


def featurize(lg, games, ctx, talent_elo=True):
    groups = set()
    if games[["h_epa", "a_epa"]].notna().any().any():
        groups.add("advanced")
    if games["home_qb_id"].notna().any():
        groups.add("qb")
    if ctx.team_season:
        groups.add("roster")
    if "advanced" in groups:
        groups.add("adjusted")
    if ctx.box is not None:
        groups.add("turnovers")
    feats = Builder(lg, ctx, groups, talent_elo).run(games)
    gc = {"base": BASE_COLS}
    if "advanced" in groups:
        gc["advanced"] = ADV_COLS
    if "adjusted" in groups:
        gc["adjusted"] = ADJ_COLS
    if "advanced" in groups:
        feats = feats.join(epar_features(games))
        gc["epar"] = EPAR_COLS
    if "turnovers" in groups:
        gc["turnovers"] = TOV_COLS
    if "qb" in groups:
        gc["qb"] = QB_COLS
    if "roster" in groups:
        gc["roster"] = ROSTER_COLS
    if ctx.inj_counts is not None:
        feats = feats.join(injury_features(games, ctx.inj_counts))
        gc["injuries"] = INJ_COLS
    if ctx.injw_counts is not None:
        feats = feats.join(injw_features(games, ctx.injw_counts))
        gc["injw"] = INJW_COLS
    if ctx.injq_counts is not None:
        feats = feats.join(injq_features(games, ctx.injq_counts))
        gc["injq"] = INJQ_COLS
    wx = weather_features(games)
    if wx is not None:
        feats = feats.join(wx)
        gc["weather"] = WX_COLS
    if games["mkt_spread"].notna().any():
        feats["mkt_spread"], feats["mkt_total"] = games["mkt_spread"], games["mkt_total"]
        gc["market"] = MARKET_COLS
    return feats, gc


def select_cols(gc, include):
    cols = list(gc["base"])
    for g in include:
        if g in gc:
            cols += gc[g]
    return cols


def chosen_groups(gc, args):
    inc = [g for g in gc if g not in ("base", "market") and g not in args.drop_set
           and (g not in NEW_GROUPS or g in args.with_set)]   # v3 groups are opt-in: they did not beat the v2 model
    if args.use_market and "market" in gc:
        inc.append("market")
    return inc


def acc(pred, actual):
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    m = actual != 0
    return float(np.mean((pred[m] > 0) == (actual[m] > 0)))


def score_block(d):
    out = {"games": len(d),
           "margin_MAE": np.abs(d.actual_margin - d.pred_margin).mean(),
           "total_MAE": np.abs(d.actual_total - d.pred_total).mean(),
           "win_acc": acc(d.pred_margin, d.actual_margin),
           "elo_only_acc": acc(d.elo_diff, d.actual_margin),
           "home_always_acc": acc(np.ones(len(d)), d.actual_margin)}
    mk = d.mkt_spread.notna()
    if mk.any():
        out["vegas_margin_MAE"] = np.abs(d.actual_margin[mk] - d.mkt_spread[mk]).mean()
        out["vegas_total_MAE"] = np.abs(d.actual_total[mk] - d.mkt_total[mk]).mean()
        out["vegas_acc"] = acc(d.mkt_spread[mk], d.actual_margin[mk])
    return out


def run_backtest(lg, games, feats, cols, kind, n_test, warmup):
    played = games.home_pts.notna() & games.away_pts.notna()
    margin, total = games.home_pts - games.away_pts, games.home_pts + games.away_pts
    seasons = sorted(games.loc[played, "season"].unique())
    first_train = seasons[0] + warmup
    frames = []
    for s in seasons[-n_test:]:
        tr = played & (games.season < s) & (games.season >= first_train)
        te = played & (games.season == s)
        if tr.sum() < 200 or te.sum() == 0:
            continue
        model = ScorePredictor(kind, lg.default_sigma).fit(feats.loc[tr, cols], margin[tr], total[tr])
        pr = model.predict(feats.loc[te, cols])
        pr["season"] = s
        pr["actual_margin"], pr["actual_total"] = margin[te], total[te]
        pr["elo_diff"] = feats.loc[te, "elo_diff"]
        pr["mkt_spread"], pr["mkt_total"] = games.loc[te, "mkt_spread"], games.loc[te, "mkt_total"]
        pr["mkt_ml_home"], pr["mkt_ml_away"] = games.loc[te, "mkt_ml_home"], games.loc[te, "mkt_ml_away"]
        frames.append(pr)
    if not frames:
        sys.exit("Not enough data to backtest.")
    return pd.concat(frames)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_backtest(args):
    lg, games, ctx = prepare(args)
    feats, gc = featurize(lg, games, ctx, talent_elo="roster" not in args.drop_set)
    inc = chosen_groups(gc, args)
    allr = run_backtest(lg, games, feats, select_cols(gc, inc), args.model, args.test_seasons,
                        args.warmup if args.warmup is not None else lg.default_warmup)
    rows = {int(s): score_block(d) for s, d in allr.groupby("season")}
    rows["ALL"] = score_block(allr)
    print(f"\n{lg.name.upper()} walk-forward backtest  (model={args.model})")
    print(f"feature groups: base + {', '.join(inc) if inc else '(none)'}\n")
    print(pd.DataFrame(rows).T.round(3).to_string())
    print("\nMAE = mean absolute error in points (lower is better). win_acc = straight-up winner accuracy.")


def cmd_ablate(args):
    args.with_set = set(NEW_GROUPS)   # the ablation always evaluates the opt-in groups
    lg, games, ctx = prepare(args)
    feats, gc = featurize(lg, games, ctx)
    feats_nt = featurize(lg, games, ctx, talent_elo=False)[0] if "roster" in gc else feats
    avail = [g for g in gc if g not in ("base", "market")]
    wu = args.warmup if args.warmup is not None else lg.default_warmup
    configs = [("base only", [])] + [(f"base + {g}", [g]) for g in avail] + [("ALL groups", avail),
               ("PREVIOUS model (no new groups)", [g for g in avail if g not in NEW_GROUPS])] + \
              [(f"ALL - {g}", [x for x in avail if x != g]) for g in avail]
    rows, vegas = {}, None
    for name, inc in configs:
        print(f"  running: {name}", file=sys.stderr)
        f = feats if "roster" in inc else feats_nt
        r = score_block(run_backtest(lg, games, f, select_cols(gc, inc), args.model, args.test_seasons, wu))
        rows[name] = {k: r[k] for k in ("margin_MAE", "total_MAE", "win_acc")}
        vegas = vegas or {k: r[k] for k in ("vegas_margin_MAE", "vegas_total_MAE", "vegas_acc") if k in r}
    print(f"\n{lg.name.upper()} feature ablation  (model={args.model}, "
          f"last {args.test_seasons} seasons, walk-forward)\n")
    print(pd.DataFrame(rows).T.round(3).to_string())
    if vegas:
        print("\nVegas closing line for reference: " + ", ".join(f"{k}={v:.3f}" for k, v in vegas.items()))


def cmd_predict(args):
    lg, games, ctx = prepare(args, forecast_days=args.days if args.league == "nfl" else None)
    feats, gc = featurize(lg, games, ctx, talent_elo="roster" not in args.drop_set)
    inc = chosen_groups(gc, args)
    cols = select_cols(gc, inc)
    played = games.home_pts.notna() & games.away_pts.notna()
    wu = args.warmup if args.warmup is not None else lg.default_warmup
    tr = played & (games.season >= games.loc[played, "season"].min() + wu)

    today = pd.Timestamp.today().normalize()
    up = (~played) & (games.date >= today - pd.Timedelta(days=1)) & \
         (games.date <= today + pd.Timedelta(days=args.days))
    if not up.any():
        sys.exit(f"No unplayed games found in the next {args.days} days.")

    margin, total = games.home_pts - games.away_pts, games.home_pts + games.away_pts
    model = ScorePredictor(args.model, lg.default_sigma).fit(feats.loc[tr, cols], margin[tr], total[tr])
    pr = model.predict(feats.loc[up, cols])
    out = games.loc[up, ["date", "away", "home", "neutral"]].join(pr)
    if "weather" in gc:
        out = out.join(feats.loc[up, ["is_dome", "temp", "wind", "weather_known"]])
    out = out.sort_values("date")

    print(f"\n{lg.name.upper()} predictions (model={args.model}; groups: base + "
          f"{', '.join(inc) if inc else '(none)'}; margin sigma ~ {model.sigma:.1f})\n")
    for _, r in out.iterrows():
        fav = (f"{r.home} -{r.pred_margin:.1f}" if r.pred_margin >= 0 else f"{r.away} -{-r.pred_margin:.1f}")
        extra = " (neutral)" if r.neutral else ""
        if "is_dome" in out and r.is_dome == 0:
            tag = "" if r.weather_known else " est."  # est. = climatology fallback, not a forecast
            extra += f" [{r.temp:.0f}F, {r.wind:.0f}mph{tag}]"
        print(f"{r.date:%Y-%m-%d}  {r.away:>22} {r.pred_away:5.1f}  @  {r.pred_home:5.1f} {r.home:<22} | "
              f"{fav:<26} | total {r.pred_total:5.1f} | home win {r.home_win_prob:4.0%}{extra}")
    fname = f"predictions_{lg.name}.csv"
    out.to_csv(fname, index=False)
    print(f"\nSaved {fname}")


# ---- Betting logic: turns model-vs-Vegas gaps into picks, and grades the same rules historically
EDGE_PLAY, EDGE_STRONG = 3.0, 4.0   # points of disagreement with Vegas -> play / strong play (1.5u)
WIN_UNITS = 100 / 110               # profit per 1u risked at -110
ML_BAND = 2.0                       # Vegas spread within this many points (either way) -> pick the moneyline instead


def ml_pay(odds):
    """Profit per 1u risked on an American moneyline."""
    return odds / 100 if odds > 0 else 100 / -odds


def wilson(w, n, z=1.96):
    if n == 0:
        return None, None
    p, d = w / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def american_prob(ml):
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def make_picks(home, away, pred_margin, pred_total, v_spread, v_total, ml_home=None, ml_away=None):
    """v_spread = Vegas expected home margin (+ = home favored)."""
    out = {"sim_spread": None, "sim_total": None, "spread_play": None, "total_play": None}
    if pd.notna(v_spread):
        e = pred_margin - v_spread
        team, line = (home, -v_spread) if e > 0 else (away, v_spread)
        odds = {home: ml_home, away: ml_away}
        if abs(v_spread) <= ML_BAND and pd.notna(odds[home]) and pd.notna(odds[away]):
            # a near pick'em: bet the moneyline on the side the model likes
            team = home if e > 0 else away
            out["sim_spread"] = {"team": team, "line": None, "ml": float(odds[team]), "edge": abs(e)}
            if abs(e) >= EDGE_PLAY:
                out["spread_play"] = {"team": team, "line": None, "ml": float(odds[team]), "kind": "ml",
                                      "edge": abs(e), "units": 1.5 if abs(e) >= EDGE_STRONG else 1.0}
        else:
            out["sim_spread"] = {"team": team, "line": line, "edge": abs(e)}
            if abs(e) >= EDGE_PLAY:
                out["spread_play"] = {"team": team, "line": line, "edge": abs(e),
                                      "units": 1.5 if abs(e) >= EDGE_STRONG else 1.0}
    if pd.notna(v_total):
        e = pred_total - v_total
        side = "Over" if e > 0 else "Under"
        out["sim_total"] = {"side": side, "line": v_total, "edge": abs(e)}
        if abs(e) >= EDGE_PLAY:
            out["total_play"] = {"side": side, "line": v_total, "edge": abs(e),
                                 "units": 1.5 if abs(e) >= EDGE_STRONG else 1.0}
    return out


def rec(res, units=None, pay=None):
    res = np.asarray(res, float)
    u = np.ones(len(res)) if units is None else np.asarray(units, float)
    pw = np.full(len(res), WIN_UNITS) if pay is None else np.asarray(pay, float)
    w, l = int((res == 1).sum()), int((res == -1).sum())
    lo, hi = wilson(w, w + l)
    net = float((u * np.where(res == 1, pw, np.where(res == -1, -1.0, 0.0))).sum())
    return {"w": w, "l": l, "p": int((res == 0).sum()), "win_pct": w / (w + l) if w + l else None,
            "lo": lo, "hi": hi, "units": net, "staked": float(u[res != 0].sum())}


def build_record(allr):
    """Walk-forward record of the betting rules (every game was predicted before it was played)."""
    d = allr[allr.mkt_spread.notna()].copy()
    es = d.pred_margin - d.mkt_spread
    s_res = np.where(es > 0, 1, -1) * np.sign(d.actual_margin - d.mkt_spread)
    s_u = np.where(es.abs() >= EDGE_STRONG, 1.5, 1.0)
    t = d[d.mkt_total.notna()]
    et = t.pred_total - t.mkt_total
    t_res = np.where(et > 0, 1, -1) * np.sign(t.actual_total - t.mkt_total)
    t_u = np.where(et.abs() >= EDGE_STRONG, 1.5, 1.0)
    sp, tp = (es.abs() >= EDGE_PLAY).to_numpy(), (et.abs() >= EDGE_PLAY).to_numpy()
    return {
        "official_all": rec(np.concatenate([s_res[sp], t_res[tp]]), np.concatenate([s_u[sp], t_u[tp]])),
        "official_spread": rec(s_res[sp], s_u[sp]), "official_total": rec(t_res[tp], t_u[tp]),
        "sim_spread_all": rec(s_res), "sim_over": rec(t_res[(et > 0).to_numpy()]),
        "sim_under": rec(t_res[(et <= 0).to_numpy()]),
    }


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return None if pd.isna(o) else round(float(o), 3)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _kickoff(r):
    gt = r.gametime
    if isinstance(gt, str) and ":" in gt:  # NFL: Eastern time -> UTC
        return (pd.Timestamp(f"{r.date:%Y-%m-%d} {gt}", tz="America/New_York")
                .tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
    if r.date.hour or r.date.minute:       # CFB: already UTC
        return r.date.strftime("%Y-%m-%dT%H:%M:%SZ")
    return r.date.strftime("%Y-%m-%d")


def _gid(x):
    if isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x):
        return str(int(x))
    return str(x)


def _before_kickoff(k, now):
    return pd.Timestamp(k) > now if len(k) > 10 else pd.Timestamp(k).date() > now.date()


def grade_pick(e, hp, ap):
    if e["type"] == "ml":
        mg = (hp - ap) if e["team"] == e["home"] else (ap - hp)
        res = 1 if mg > 0 else -1 if mg < 0 else 0
        return res, e["units"] * (ml_pay(e["ml"]) if res == 1 else -1.0 if res == -1 else 0.0)
    if e["type"] == "spread":
        d = ((hp - ap) if e["team"] == e["home"] else (ap - hp)) + e["line"]
    else:
        tot = hp + ap
        d = tot - e["line"] if e["side"] == "Over" else e["line"] - tot
    res = 1 if d > 0 else -1 if d < 0 else 0
    return res, e["units"] * (WIN_UNITS if res == 1 else -1.0 if res == -1 else 0.0)


PICK_TYPES = ("spread",)   # official top plays are spread plays only (near pick'ems are played on the moneyline)
TOP_N, LOCK_DAYS = 5, 3   # official plays per week; a week's plays lock this many days before its first kickoff


def _utc(k):
    t = pd.Timestamp(k)
    return t.tz_localize("UTC") if t.tzinfo is None else t


def spread_only_backfill(path, allr, games, season, now):
    """The official top plays are spread plays only. Total plays already in this season's log are removed, and any
    finished week left short of TOP_N is topped up with the plays the rules would have made on spreads (marked
    replay, built from the model's pre-game predictions and graded at the final line)."""
    if not os.path.exists(path):
        return
    log = json.load(open(path))
    gone = [e for e in log if e.get("type") == "total"]
    if not gone:
        return
    log = [e for e in log if e.get("type") != "total"]
    meta = games.assign(_g=games.game_id.map(_gid)).drop_duplicates("_g").set_index("_g")
    weeks = {(e["season"], e["week"]) for e in gone if e.get("season") is not None}
    for (sn, w) in sorted(weeks):
        g_w = games[(games.season == sn) & (games.week == w)]
        if g_w.empty or not (g_w.home_pts.notna().all() and g_w.away_pts.notna().all()):
            continue                                              # unfinished weeks refill live
        have = [e for e in log if (e.get("season"), e.get("week")) == (sn, w)]
        used = {e["game_id"] for e in have}
        d = allr[(allr.season == sn) & allr.mkt_spread.notna()]
        d = d[games.loc[d.index, "week"] == w]
        cand = []
        for idx, r in d.iterrows():
            g = games.loc[idx]
            if _gid(g.game_id) in used:
                continue
            p = make_picks(g.home, g.away, r.pred_margin, r.pred_total, r.mkt_spread, r.mkt_total,
                           r.mkt_ml_home, r.mkt_ml_away)["spread_play"]
            if not p:
                continue
            e = {"id": f"{_gid(g.game_id)}|spread", "type": p.get("kind") or "spread", "ml": p.get("ml"),
                 "game_id": _gid(g.game_id), "league": "nfl", "season": int(sn), "week": int(w), "kickoff": _kickoff(g),
                 "away": g.away, "home": g.home, "team": p.get("team"), "side": p.get("side"), "line": p["line"],
                 "edge": p["edge"], "units": p["units"], "pred_away": r.pred_away, "pred_home": r.pred_home,
                 "status": "graded", "replay": True, "logged_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "home_pts": float(g.home_pts), "away_pts": float(g.away_pts)}
            e["result"], net = grade_pick(e, e["home_pts"], e["away_pts"])
            e["net"] = round(net, 3)
            cand.append(e)
        cand.sort(key=lambda e: -e["edge"])
        log += _clean(cand[:max(TOP_N - len(have), 0)])
    log.sort(key=lambda e: (e.get("season") or 0, e.get("week") or 0, e["kickoff"]))
    with open(path, "w") as f:
        json.dump(log, f, indent=1)
    print(f"  spread-only: removed {len(gone)} total plays from the log and topped up finished weeks", file=sys.stderr)


def update_pick_log(path, league, cands, games, now):
    """Keep the log to the top TOP_N plays per week. Plays are added once, before kickoff, at the line when
    first posted, and never edited. Returns (live record dict, {play id: official-play details with rank})."""
    log = json.load(open(path)) if os.path.exists(path) else []
    meta = games.assign(_g=games.game_id.map(_gid)).drop_duplicates("_g").set_index("_g")

    def wk(e):
        if e.get("season") is not None and e.get("week") is not None:
            return (e["season"], e["week"])
        if e["game_id"] in meta.index:
            return (int(meta.at[e["game_id"], "season"]), int(meta.at[e["game_id"], "week"]))
        return (None, None)

    # 1) cap at TOP_N per week; only plays whose games haven't kicked off can ever be removed
    by = defaultdict(list)
    for e in log:
        by[wk(e)].append(e)
    drop = set()
    for es in by.values():
        pend = sorted([e for e in es if e["status"] == "pending" and _before_kickoff(e["kickoff"], now)],
                      key=lambda e: -e["edge"])
        room = TOP_N - (len(es) - len(pend))
        drop |= {e["id"] for e in pend[max(room, 0):]}
    log = [e for e in log if e["id"] not in drop]

    # 2) lock each week's top plays once its first kickoff is within LOCK_DAYS
    have = {e["id"] for e in log}
    cnt = defaultdict(int)
    for e in log:
        cnt[wk(e)] += 1
    byw = defaultdict(list)
    for u in cands:
        byw[(u["season"], u["week"])].append(u)
    for k, us in byw.items():
        if _utc(min(u["kickoff"] for u in us)) - now > pd.Timedelta(days=LOCK_DAYS) or cnt[k] >= TOP_N:
            continue
        pool = [(p["edge"], typ, u, p) for u in us if _before_kickoff(u["kickoff"], now)
                for typ in PICK_TYPES for p in [u.get(f"{typ}_play")]
                if p and f"{u['game_id']}|{typ}" not in have]
        pool.sort(key=lambda x: -x[0])
        for edge, typ, u, p in pool[:TOP_N - cnt[k]]:
            log.append(_clean({"id": f"{u['game_id']}|{typ}", "type": p.get("kind") or typ, "ml": p.get("ml"), "game_id": u["game_id"], "league": league,
                               "season": u["season"], "week": u["week"], "kickoff": u["kickoff"],
                               "away": u["away"], "home": u["home"], "team": p.get("team"), "side": p.get("side"),
                               "line": p["line"], "edge": p["edge"], "units": p["units"],
                               "pred_away": u["pred_away"], "pred_home": u["pred_home"], "status": "pending",
                               "logged_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")}))

    # 2a) a pending play keeps its side but its LINE follows the market until kickoff, then freezes
    cur = {u["game_id"]: u for u in cands}
    ov_ids = set()
    _ovp = os.path.join(os.path.dirname(path) or ".", "line_overrides.json")
    if os.path.exists(_ovp):
        ov_ids = {o["id"] for o in json.load(open(_ovp))}
    for e in log:
        u = cur.get(e["game_id"])
        if e["status"] != "pending" or u is None or e["id"] in ov_ids or not _before_kickoff(e["kickoff"], now):
            continue
        new = None
        if e["type"] == "spread" and pd.notna(u.get("mkt_spread")):
            new = float(-u["mkt_spread"] if e["team"] == e["home"] else u["mkt_spread"])
        elif e["type"] == "total" and pd.notna(u.get("mkt_total")):
            new = float(u["mkt_total"])
        elif e["type"] == "ml":
            o_ = u.get("mkt_ml_home") if e["team"] == e["home"] else u.get("mkt_ml_away")
            if pd.notna(o_) and float(o_) != e.get("ml"):
                e.setdefault("ml_first", e.get("ml"))
                e["ml"] = float(o_)
                e["line_updated_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        if new is not None and new != e["line"]:
            e.setdefault("line_first", e["line"])
            e["line"], e["line_updated_at"] = new, now.strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"  line follows market: {e['id']} {e['line_first']} -> {new}", file=sys.stderr)

    # 2b) manual line corrections from data/line_overrides.json: [{"id", "line", "side"?, "note"?}].
    #     The original line is kept on the entry (line_orig) and the site marks the pick as edited.
    ov_path = os.path.join(os.path.dirname(path) or ".", "line_overrides.json")
    ovs = {o["id"]: o for o in json.load(open(ov_path))} if os.path.exists(ov_path) else {}
    for e in log:
        o = ovs.get(e["id"])
        if not o or e.get("line") == float(o["line"]):
            continue
        if o.get("side") and e.get("side") != o["side"]:
            warn(f"override for {e['id']} skipped: logged side is {e.get('side')}, not {o['side']}")
            continue
        if e["type"] == "spread" and (e["line"] > 0) != (float(o["line"]) > 0):
            warn(f"override for {e['id']} skipped: sign differs from logged line {e['line']}")
            continue
        e["line_orig"] = e.get("line_orig", e["line"])
        e["line"] = float(o["line"])
        e["line_note"] = o.get("note") or "line edited by site owner"
        if e["status"] == "graded":           # re-grade at the new line
            res, net = grade_pick(e, e["home_pts"], e["away_pts"])
            e.update(result=res, net=round(net, 3))
        print(f"  line override applied: {e['id']} {e['line_orig']} -> {e['line']}", file=sys.stderr)

    # 3) grade finished plays
    fin = meta
    for e in log:
        if e["status"] == "graded" or e["game_id"] not in fin.index:
            continue
        hp, ap = fin.at[e["game_id"], "home_pts"], fin.at[e["game_id"], "away_pts"]
        if pd.isna(hp) or pd.isna(ap):
            continue
        res, net = grade_pick(e, float(hp), float(ap))
        e.update(status="graded", home_pts=float(hp), away_pts=float(ap), result=res, net=round(net, 3))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=1)

    done = sorted([e for e in log if e["status"] == "graded"], key=lambda e: e["kickoff"])
    pend = sorted([e for e in log if e["status"] == "pending"], key=lambda e: e["kickoff"])

    def sub(t=None):
        xs = [e for e in done if t in (None, e["type"])]
        return rec([e["result"] for e in xs], [e["units"] for e in xs],
                   [ml_pay(e["ml"]) if e["type"] == "ml" else WIN_UNITS for e in xs])
    series, cum = [], 0.0
    for e in done:
        cum += e["net"]
        day = e["kickoff"][:10]
        if series and series[-1][0] == day:
            series[-1][1] = round(cum, 2)
        else:
            series.append([day, round(cum, 2)])
    live = {"record": {"all": sub(), "spread": sub("spread"), "ml": sub("ml"), "total": sub("total")}, "series": series,
            "picks": pend + done[::-1][:50], "pending": len(pend),
            "since": min([e["logged_at"][:10] for e in log], default=None)}
    # most recent week whose games have ALL finished: its official plays become "last week"
    wdone = {k: bool(g.home_pts.notna().all() and g.away_pts.notna().all())
             for k, g in games.groupby(["season", "week"])}
    byw_ = defaultdict(list)
    for e in log:
        byw_[wk(e)].append(e)
    fin_weeks = [k for k, es in byw_.items() if k[0] is not None and wdone.get((k[0], k[1])) and
                 all(e["status"] == "graded" for e in es)]
    if fin_weeks:
        k = max(fin_weeks)
        es = sorted(byw_[k], key=lambda e: e["kickoff"])
        ppay = lambda xs: [ml_pay(e["ml"]) if e["type"] == "ml" else WIN_UNITS for e in xs]
        def rw(t=None):
            xs = [e for e in es if t in (None, e["type"])]
            return rec([e["result"] for e in xs], [e["units"] for e in xs], ppay(xs))
        live["last_week"] = {"season": k[0], "week": k[1], "picks": es,
                             "record": {"all": rw(), "spread": rw("spread"), "ml": rw("ml"), "total": rw("total")}}
    arch = []
    for k in sorted(fin_weeks, reverse=True):
        es = sorted(byw_[k], key=lambda e: e["kickoff"])
        ppay = lambda xs: [ml_pay(e["ml"]) if e["type"] == "ml" else WIN_UNITS for e in xs]
        arch.append({"season": k[0], "week": k[1], "picks": es,
                     "record": rec([e["result"] for e in es], [e["units"] for e in es], ppay(es))})
    live["archive"] = arch
    official, byk = {}, defaultdict(list)
    for e in log:
        byk[wk(e)].append(e)
    for es in byk.values():
        for r, e in enumerate(sorted(es, key=lambda e: -e["edge"]), 1):
            official[e["id"]] = {"team": e.get("team"), "side": e.get("side"), "line": e["line"],
                                 "kind": e["type"] if e["type"] == "ml" else None, "ml": e.get("ml"),
                                 "edge": e["edge"], "units": e["units"], "rank": r, "of": TOP_N, "wk": list(wk(e))}
    return live, official


def update_all_log(path, league, upcoming, games, now):
    """Live record of EVERY game's model pick (straight up, spread side, over/under side), not just the official
    top plays. A game is logged once, within LOCK_DAYS of kickoff and before it starts, at the line then posted,
    and never edited. Spread/total picks are flat 1u at -110. Returns {su, spread, total, pending, since}."""
    log = json.load(open(path)) if os.path.exists(path) else []
    have = {e["id"] for e in log}
    for u in upcoming:
        if not _before_kickoff(u["kickoff"], now) or _utc(u["kickoff"]) - now > pd.Timedelta(days=LOCK_DAYS):
            continue
        base = {"game_id": u["game_id"], "league": league, "kickoff": u["kickoff"], "away": u["away"],
                "home": u["home"], "units": 1.0, "status": "pending", "logged_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
        rows = [("su", {"team": u["home"] if u["pred_margin"] > 0 else u["away"], "line": None, "side": None})]
        if u.get("sim_spread"):
            ss = u["sim_spread"]
            if ss.get("ml") is not None:
                rows.append(("ml", {"team": ss["team"], "line": None, "side": None, "ml": ss["ml"]}))
            else:
                rows.append(("spread", {"team": ss["team"], "line": ss["line"], "side": None}))
        if u.get("sim_total"):
            rows.append(("total", {"team": None, "line": u["sim_total"]["line"], "side": u["sim_total"]["side"]}))
        for typ, extra in rows:
            if f"{u['game_id']}|{typ}" not in have:
                log.append(_clean({"id": f"{u['game_id']}|{typ}", "type": typ, **base, **extra}))
    meta = games.assign(_g=games.game_id.map(_gid)).drop_duplicates("_g").set_index("_g")
    for e in log:
        if e["status"] == "graded" or e["game_id"] not in meta.index:
            continue
        hp, ap = meta.at[e["game_id"], "home_pts"], meta.at[e["game_id"], "away_pts"]
        if pd.isna(hp) or pd.isna(ap):
            continue
        hp, ap = float(hp), float(ap)
        if e["type"] == "su":
            mg = (hp - ap) if e["team"] == e["home"] else (ap - hp)
            res = 1 if mg > 0 else -1 if mg < 0 else 0
        else:
            res, _ = grade_pick(e, hp, ap)
        e.update(status="graded", home_pts=hp, away_pts=ap, result=res)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=1)

    def sub(t):
        return rec([e["result"] for e in log if e["type"] == t and e["status"] == "graded"], None)
    return {"su": sub("su"), "spread": sub("spread"), "total": sub("total"),
            "pending": len({e["game_id"] for e in log if e["status"] == "pending"}),
            "since": min([e["logged_at"][:10] for e in log], default=None)}


def all_pick_blocks(d):
    """Model's pick in EVERY game of a walk-forward backtest frame -> {su, spread, total} records.
    Spread/total are flat 1u at -110; straight up is win/loss only."""
    d = d[d.actual_margin.notna()]
    su = np.sign(d.actual_margin) * np.where(d.pred_margin > 0, 1, -1)
    out = {"su": rec(su)}
    m = d[d.mkt_spread.notna()]
    es = m.pred_margin - m.mkt_spread
    out["spread"] = rec(np.where(es > 0, 1, -1) * np.sign(m.actual_margin - m.mkt_spread), np.ones(len(m)))
    t = d[d.mkt_total.notna()]
    et = t.pred_total - t.mkt_total
    out["total"] = rec(np.where(et > 0, 1, -1) * np.sign(t.actual_total - t.mkt_total), np.ones(len(t)))
    return out


def play_frame(allr, games, spreads_only=True):
    """Every official play the rules would have made, graded (for the backtest)."""
    d = allr[allr.mkt_spread.notna()].copy()
    d["date"] = games.loc[d.index, "date"]
    d["week"] = games.loc[d.index, "week"]
    es = d.pred_margin - d.mkt_spread
    sp = d[es.abs() >= EDGE_PLAY].copy()
    sp["edge"] = es[sp.index].abs()
    sp["type"] = "spread"
    sp["res"] = np.where(es[sp.index] > 0, 1, -1) * np.sign(sp.actual_margin - sp.mkt_spread)
    sp["units"] = np.where(es[sp.index].abs() >= EDGE_STRONG, 1.5, 1.0)
    sp["pay"] = WIN_UNITS
    # near pick'ems (spread within ML_BAND) become moneyline plays on the model's side
    hasml = sp.mkt_ml_home.notna() & sp.mkt_ml_away.notna() & (sp.mkt_spread.abs() <= ML_BAND)
    if hasml.any():
        m = sp[hasml]
        pick_home = (es[m.index] > 0).to_numpy()
        sp.loc[m.index, "type"] = "ml"
        sp.loc[m.index, "res"] = np.where(pick_home, 1, -1) * np.sign(m.actual_margin)
        odds = np.where(pick_home, m.mkt_ml_home, m.mkt_ml_away)
        sp.loc[m.index, "pay"] = np.where(odds > 0, odds / 100, 100 / np.abs(odds))
    t = d[d.mkt_total.notna()]
    et = t.pred_total - t.mkt_total
    tp = t[et.abs() >= EDGE_PLAY].copy()
    tp["edge"] = et[tp.index].abs()
    tp["type"] = "total"
    tp["res"] = np.where(et[tp.index] > 0, 1, -1) * np.sign(tp.actual_total - tp.mkt_total)
    tp["units"] = np.where(et[tp.index].abs() >= EDGE_STRONG, 1.5, 1.0)
    tp["pay"] = WIN_UNITS
    cols = ["date", "season", "week", "type", "res", "units", "edge", "pay"]
    out = pd.concat([sp[cols]] if spreads_only else [sp[cols], tp[cols]])
    out = out.sort_values("edge", ascending=False).groupby(["season", "week"]).head(TOP_N).sort_values("date")
    out["net"] = out.units * np.where(out.res == 1, out.pay, np.where(out.res == -1, -1.0, 0.0))
    return out


def replay_weeks(allr, games, season, skip_weeks):
    """Official plays the rules WOULD have made for finished weeks that were never logged live, built from the
    model's pre-game predictions (walk-forward) and graded. Returned in the same shape as live log entries."""
    out = []
    d = allr[(allr.season == season) & allr.mkt_spread.notna()]
    d = d[games.loc[d.index, "week"].isin(
        [w for w, g in games[games.season == season].groupby("week") if g.home_pts.notna().all() and g.away_pts.notna().all()])]
    byw = defaultdict(list)
    for idx, r in d.iterrows():
        g = games.loc[idx]
        w = int(g.week)
        if (season, w) in skip_weeks or pd.isna(g.home_pts) or pd.isna(g.away_pts):
            continue
        pk = make_picks(g.home, g.away, r.pred_margin, r.pred_total, r.mkt_spread, r.mkt_total,
                        r.mkt_ml_home, r.mkt_ml_away)
        for typ in PICK_TYPES:
            p = pk[f"{typ}_play"]
            if not p:
                continue
            e = {"id": f"{_gid(g.game_id)}|{typ}", "type": p.get("kind") or typ, "ml": p.get("ml"), "game_id": _gid(g.game_id),
                 "season": int(season), "week": w, "kickoff": _kickoff(g), "away": g.away, "home": g.home,
                 "team": p.get("team"), "side": p.get("side"), "line": p["line"], "edge": p["edge"], "units": p["units"],
                 "status": "graded", "replay": True, "home_pts": float(g.home_pts), "away_pts": float(g.away_pts)}
            e["result"], net = grade_pick(e, e["home_pts"], e["away_pts"])
            e["net"] = round(net, 3)
            byw[w].append(e)
    for w, es in byw.items():
        es = sorted(es, key=lambda e: -e["edge"])[:TOP_N]
        es.sort(key=lambda e: e["kickoff"])
        out.append({"season": int(season), "week": w, "replay": True, "picks": _clean(es),
                    "record": rec([e["result"] for e in es], [e["units"] for e in es],
                                  [ml_pay(e["ml"]) if e["type"] == "ml" else WIN_UNITS for e in es])})
    return sorted(out, key=lambda a: -a["week"])


def hist_calibration(allr, games):
    """Out-of-sample hit rate of every play with a 3+ point gap, by gap size and bet type."""
    global TOP_N
    old = TOP_N
    TOP_N = 999
    try:
        pf = play_frame(allr, games, spreads_only=False)
    finally:
        TOP_N = old
    out = {}
    for name, sel in (("spread", pf.type != "total"), ("total", pf.type == "total")):
        d, rows = pf[sel], []
        for lo, hi in ((EDGE_PLAY, 4.0), (4.0, 5.0), (5.0, 99.0)):
            x = d[(d.edge >= lo) & (d.edge < hi)]
            w, l = int((x.res == 1).sum()), int((x.res == -1).sum())
            rows.append({"lo": lo, "hi": hi, "w": w, "l": l, "win_pct": w / (w + l) if w + l else None})
        out[name] = rows
    return out


def backtest_block(pf):
    def three(g):
        r = lambda x: rec(x.res, x.units, x.pay)
        return {"all": r(g), "spread": r(g[g.type == "spread"]), "ml": r(g[g.type == "ml"]),
                "total": r(g[g.type == "total"])}
    ser = pf.groupby(pf.date.dt.strftime("%Y-%m-%d")).net.sum().cumsum()
    return {"overall": three(pf), "by_season": {str(int(k)): three(g) for k, g in pf.groupby("season")},
            "series": [[k, round(float(v), 2)] for k, v in ser.items()]}


STAT_DIR = {"ppg": 1, "papg": -1, "ppg_l3": 1, "papg_l3": -1, "ypp": 1, "ypp_allowed": -1, "epa_off": 1, "epa_def": -1,
            "succ_off": 1, "succ_def": -1, "pass_epa_off": 1, "pass_epa_def": -1, "rush_epa_off": 1,
            "rush_epa_def": -1, "third_off": 1, "third_def": -1, "tov_margin": 1}   # +1 = higher is better


def team_stats(games, ctx, season):
    """Season-to-date team stats and league ranks (1 = best). 'def' stats are what the defense allowed."""
    g = games[(games.season == season) & games.home_pts.notna() & games.away_pts.notna()].copy()
    if g.empty:
        return {}, 0
    g["_g"] = g.game_id.map(_gid)

    def frame(me, you, tcol, ocol):
        d = pd.DataFrame({"game_id": g["_g"].values, "side": me, "team": g[tcol].values, "date": g["date"].values,
                          "pf": g[f"{tcol}_pts"].values, "pa": g[f"{ocol}_pts"].values})
        for s_ in ADV_STATS:
            d[f"off_{s_}"] = g[f"{me}_{s_}"].values
            d[f"def_{s_}"] = g[f"{you}_{s_}"].values
        return d
    L = pd.concat([frame("h", "a", "home", "away"), frame("a", "h", "away", "home")], ignore_index=True)
    L = L.sort_values("date", kind="stable")
    has_box = ctx.box is not None
    if has_box:
        b = ctx.box
        L = L.merge(b.add_prefix("o_").rename(columns={"o_game_id": "game_id", "o_side": "side"}),
                    on=["game_id", "side"], how="left")
        ob = b.copy()
        ob["side"] = ob.side.map({"h": "a", "a": "h"})
        L = L.merge(ob.add_prefix("d_").rename(columns={"d_game_id": "game_id", "d_side": "side"}),
                    on=["game_id", "side"], how="left")
    rows = {}
    for team, d in L.groupby("team"):
        r = {"games": len(d), "w": int((d.pf > d.pa).sum()), "l": int((d.pf < d.pa).sum()),
             "t": int((d.pf == d.pa).sum()), "ppg": d.pf.mean(), "papg": d.pa.mean(),
             "ppg_l3": d.pf.tail(3).mean(), "papg_l3": d.pa.tail(3).mean(),
             "epa_off": d.off_epa.mean(), "epa_def": d.def_epa.mean(),
             "succ_off": d.off_succ.mean(), "succ_def": d.def_succ.mean(),
             "pass_epa_off": d.off_pass_epa.mean(), "pass_epa_def": d.def_pass_epa.mean(),
             "rush_epa_off": d.off_rush_epa.mean(), "rush_epa_def": d.def_rush_epa.mean()}
        if has_box and d.o_plays.notna().any() and d.o_plays.sum() > 0:
            r["ypp"] = d.o_yards.sum() / d.o_plays.sum()
            r["ypp_allowed"] = d.d_yards.sum() / d.d_plays.sum()
            r["third_off"] = d.o_conv3.sum() / max(d.o_conv3.sum() + d.o_fail3.sum(), 1)
            r["third_def"] = d.d_conv3.sum() / max(d.d_conv3.sum() + d.d_fail3.sum(), 1)
            r["tov_margin"] = d.d_tov.mean() - d.o_tov.mean()
        rows[team] = r
    T = pd.DataFrame(rows).T
    cls = pd.concat([games[["home", "home_class"]].set_axis(["t", "c"], axis=1),
                     games[["away", "away_class"]].set_axis(["t", "c"], axis=1)])
    pool = T.index.isin(set(cls[cls.c.isin(["nfl", "fbs"])].t))
    ranks = {k: T.loc[pool, k].astype(float).rank(ascending=(dr < 0), method="min")
             for k, dr in STAT_DIR.items() if k in T}
    out = {}
    for team, r in rows.items():
        sd = {}
        for k in STAT_DIR:
            if k in r and pd.notna(r[k]):
                rk = ranks[k].get(team)
                sd[k] = {"v": float(r[k]), "r": int(rk) if rk is not None and pd.notna(rk) else None}
        out[team] = {"games": r["games"], "record": f"{r['w']}-{r['l']}" + (f"-{r['t']}" if r["t"] else ""), "s": sd}
    return out, int(pool.sum())


def last_qbs(games):
    out = {}
    for r in games[games.home_pts.notna()].itertuples():
        if isinstance(r.home_qb_name, str):
            out[r.home] = r.home_qb_name
        if isinstance(r.away_qb_name, str):
            out[r.away] = r.away_qb_name
    return out


def team_card(side, team, f, ts, qbn, ctx_espn=None):
    c = dict(ts.get(team, {"games": 0, "record": "0-0", "s": {}}))
    c["elo"], c["rest"] = f.get(f"{side}_elo"), f.get(f"{side}_rest")
    if f"{side}_qb_val" in f.index:
        c["qb"] = {"name": qbn.get(team), "val": f[f"{side}_qb_val"], "flag": bool(f[f"{side}_qb_delta"] <= -0.05)}
    rep = (ctx_espn or {}).get(team)
    if rep:
        qbn_ = None
        if rep["qb_depth"]:
            outn = {norm_name(p["name"]) for p in rep["out"] if p["tier"] in ("out", "doubtful")}
            qbn_ = next((n for n in rep["qb_depth"] if norm_name(n) not in outn), None)
        c["report"] = {"qb_out": rep["qb_out"], "qb_starter": qbn_, "qb_depth1": (rep["qb_depth"] or [None])[0],
                       "players": [{k: p[k] for k in ("name", "pos", "status", "tier", "starter")}
                                   for p in sorted(rep["out"], key=lambda p: (p["tier"] != "out", not p["starter"], p["tier"]))]}
    if rep and rep["qb_depth"] and "qb" in c:
        c["qb"]["name"] = qbn_ or c["qb"]["name"]            # the QB who will actually start per ESPN's depth chart
        c["qb"]["flag"] = bool(rep["qb_out"])
        c["qb"]["listed_out"] = rep["qb_depth"][0] if rep["qb_out"] else None
    if f"{side}_inj_qb" in f.index:
        c["inj"] = {g_: f[f"{side}_inj_{g_}"] for g_ in INJ_GROUPS}
    return c


def ml_info(r):
    if pd.isna(r.mkt_ml_home) or pd.isna(r.mkt_ml_away):
        return {"ml_implied_home": None, "ml_value": None}
    ph, pa = american_prob(r.mkt_ml_home), american_prob(r.mkt_ml_away)
    fair = ph / (ph + pa)                      # remove the bookmaker's margin
    d = r.home_win_prob - fair
    return {"ml_implied_home": fair,
            "ml_value": (r.home if d >= 0.05 else r.away if d <= -0.05 else None)}



# --------------------------------------------------------------------------- #
# Opta player ratings (live only, not backtested): scale injury impact by who is actually hurt
# --------------------------------------------------------------------------- #
OPTA_URL = "https://theanalyst.com/wp-json/sdapi/v1/footballdata/playerelo?tmcl=7naonrhursiqrteihetgap6ac"
OPTA_FILE = "data/opta_ratings.json"
OPTA_ABBR = {"WSH": "WAS", "LAR": "LA", "JAX": "JAC", "LVR": "LV"}
OPTA_PTS = {"OL": 0.6, "WR": 0.5, "TE": 0.3, "RB": 0.3, "EDGE": 0.6, "DT": 0.4, "CB": 0.5, "S": 0.3, "MLB": 0.3}
OPTA_STATUS = {"out": 1.0, "doubtful": 0.75, "questionable": 0.25}
OPTA_CAP = 3.0          # max points one team can lose from offensive injuries, or give up from defensive ones


def load_opta():
    """Opta/The Analyst player ratings JSON: the live feed if reachable, else data/opta_ratings.json (pasted weekly)."""
    j = None
    try:
        r = requests.get(OPTA_URL, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=30)
        r.raise_for_status()
        j = r.json()
        print("  Opta ratings: live feed", file=sys.stderr)
    except Exception as e:
        try:
            with open(OPTA_FILE, encoding="utf-8") as fh:
                j = json.load(fh)
            print(f"  Opta ratings: {OPTA_FILE} (live feed unavailable: {e})", file=sys.stderr)
        except Exception:
            warn("Opta ratings unavailable (no live feed, no data/opta_ratings.json); injury impact is not scaled by player quality")
            return {}
    idx = {}
    for side in ("offense", "defense"):
        for p in j.get(side) or []:
            g = p.get("positionGroup")
            if g not in OPTA_PTS:
                continue
            pr = lambda k: float(p.get(k) or 50)
            if g == "OL":
                v = 0.6 * pr("passBlockPctRank") + 0.4 * pr("runBlockPctRank")
            elif g in ("WR", "TE", "RB"):
                v = 0.5 * pr("routesPctRank") + 0.5 * pr("catchingPctRank")
            elif g == "EDGE":
                v = 0.7 * pr("passRushPctRank") + 0.3 * pr("runDefensePctRank")
            elif g == "DT":
                v = 0.5 * pr("passRushPctRank") + 0.5 * pr("runDefensePctRank")
            elif g == "CB":
                v = pr("coveragePctRank")
            elif g == "S":
                v = 0.7 * pr("coveragePctRank") + 0.3 * pr("runDefensePctRank")
            else:
                v = 0.5 * pr("runDefensePctRank") + 0.5 * pr("coveragePctRank")
            team = OPTA_ABBR.get(p.get("teamAbbreviation"), p.get("teamAbbreviation"))
            snaps = float(p.get("snaps") or 0)
            val = max(0.0, (v - 50.0) / 50.0) * min(1.0, snaps / 150.0)    # above-average only; thin samples shrink
            idx[(team, norm_name(p.get("player")))] = {"side": "off" if side == "offense" else "def", "group": g,
                                                     "pct": round(v), "val": val, "snaps": snaps}
    return idx


def opta_team_adj(team, rep, idx):
    """-> {"off": pts lost, "def": pts conceded, "players": [...]} from this team's ESPN injured starters."""
    out = {"off": 0.0, "def": 0.0, "players": []}
    for p in (rep or {}).get("out", []):
        w = OPTA_STATUS.get(p["tier"], 0)
        o = idx.get((team, norm_name(p["name"])))
        if not o or not w or p.get("starter") is False:
            continue
        pts = w * o["val"] * OPTA_PTS[o["group"]]
        out[o["side"]] += pts
        out["players"].append({"name": p["name"], "pos": o["group"], "status": p["status"], "rating": o["pct"],
                               "pts": round(pts, 2)})
    out["off"], out["def"] = min(out["off"], OPTA_CAP), min(out["def"], OPTA_CAP)
    out["players"].sort(key=lambda x: -x["pts"])
    return out


def apply_opta(sub, espn, sigma):
    """Shift each upcoming game's scores for the quality of injured starters. Returns (sub, {team: adj})."""
    idx = load_opta()
    if not idx or not espn:
        return sub, {}
    adj = {t: opta_team_adj(t, espn.get(t), idx) for t in set(sub.home) | set(sub.away)}
    sub = sub.copy()
    for i, r in sub.iterrows():
        h, a = adj[r.home], adj[r.away]
        ph = r.pred_home - h["off"] + a["def"]
        pa = r.pred_away - a["off"] + h["def"]
        sub.loc[i, ["pred_home", "pred_away", "pred_margin", "pred_total"]] = [ph, pa, ph - pa, ph + pa]
        sub.loc[i, "home_win_prob"] = norm.cdf((ph - pa) / sigma)
        sub.loc[i, "opta_shift"] = (ph - pa) - r.pred_margin
    return sub, adj


def cmd_export(args):
    """Fit once, then write upcoming predictions + backtest + recent results as JSON for the website."""
    lg, games, ctx = prepare(args, forecast_days=args.days if args.league == "nfl" else None)
    feats, gc = featurize(lg, games, ctx, talent_elo="roster" not in args.drop_set)
    inc = chosen_groups(gc, args)
    cols = select_cols(gc, inc)
    wu = args.warmup if args.warmup is not None else lg.default_warmup
    played = games.home_pts.notna() & games.away_pts.notna()
    margin, total = games.home_pts - games.away_pts, games.home_pts + games.away_pts
    tr = played & (games.season >= games.loc[played, "season"].min() + wu)
    model = ScorePredictor(args.model, lg.default_sigma).fit(feats.loc[tr, cols], margin[tr], total[tr])

    today = pd.Timestamp.today().normalize()
    up = (~played) & (games.date >= today - pd.Timedelta(days=1)) & \
         (games.date <= today + pd.Timedelta(days=args.days))
    stats_season = int(games.loc[played, "season"].max())
    ts, pool_n = team_stats(games, ctx, stats_season)
    qbn = last_qbs(games)
    ctx.opta_adj = {}
    upcoming = []
    if up.any():
        sub = games.loc[up].join(model.predict(feats.loc[up, cols])).sort_values("date")
        sub["opta_shift"] = 0.0
        sub, opta_adj = apply_opta(sub, ctx.espn, model.sigma)
        ctx.opta_adj = opta_adj
        for idx, r in sub.iterrows():
            f = feats.loc[idx]
            wx = None
            if "weather" in gc:
                wx = {"dome": bool(f.is_dome), "temp": f.temp, "wind": f.wind, "known": bool(f.weather_known),
                      "gust": r.get("gust"), "pop": r.get("pop")}
            upcoming.append({"kickoff": _kickoff(r), "week": int(r.week), "game_id": _gid(r.game_id), "away": r.away, "home": r.home,
                             "neutral": bool(r.neutral), "pred_away": r.pred_away, "pred_home": r.pred_home,
                             "pred_margin": r.pred_margin, "pred_total": r.pred_total,
                             "home_win_prob": r.home_win_prob, "weather": wx,
                             "opta": {"home": ctx.opta_adj.get(r.home), "away": ctx.opta_adj.get(r.away),
                                      "shift": round(float(r.opta_shift), 2)} if ctx.opta_adj else None,
                             "home_info": team_card("home", r.home, f, ts, qbn, ctx.espn), "away_info": team_card("away", r.away, f, ts, qbn, ctx.espn),
                             "vegas_spread": r.mkt_spread, "vegas_total": r.mkt_total,
                             "ml_home": r.mkt_ml_home, "ml_away": r.mkt_ml_away, **ml_info(r),
                             **make_picks(r.home, r.away, r.pred_margin, r.pred_total, r.mkt_spread, r.mkt_total, r.mkt_ml_home, r.mkt_ml_away)})

    allr = run_backtest(lg, games, feats, cols, args.model, max(args.test_seasons, 11 if lg.name == "nfl" else 4), wu)
    recent_df = allr.join(games[["date", "home", "away", "home_pts", "away_pts"]]).sort_values("date").tail(30)
    recent = [{"date": r.date.strftime("%Y-%m-%d"), "home": r.home, "away": r.away,
               "pred_home": r.pred_home, "pred_away": r.pred_away,
               "home_pts": r.home_pts, "away_pts": r.away_pts,
               "vegas_spread": r.mkt_spread, "pred_margin": r.pred_margin,
               "correct": None if r.actual_margin == 0 else bool((r.pred_margin > 0) == (r.actual_margin > 0))}
              for r in recent_df.iloc[::-1].itertuples()]
    by_season = {str(int(s)): score_block(d) for s, d in allr.groupby("season")}
    by_season["ALL"] = score_block(allr)

    seasons2 = [current_season() - 3, current_season() - 2, current_season() - 1]   # three most recent completed seasons
    a2 = allr[allr.season.isin(seasons2)]
    bt2 = backtest_block(play_frame(a2, games))
    ytd_season = current_season()
    g26 = games[games.season == ytd_season]
    done_wk = [w for w, g in g26.groupby("week") if g.home_pts.notna().all() and g.away_pts.notna().all()]
    a26 = allr[(allr.season == ytd_season) & games.loc[allr.index, "week"].isin(done_wk)]   # only fully played weeks
    pf26 = play_frame(a26, games)
    bt_ytd = backtest_block(pf26) if len(pf26) else None
    if bt_ytd:
        bt_ytd["by_week"] = [{"week": int(w), **rec(g.res, g.units)} for w, g in pf26.groupby("week")]
        bt_ytd["season"] = ytd_season
        bt_ytd["weeks_done"] = len(done_wk)
    now = pd.Timestamp.now("UTC")
    upc = (~played) & games.mkt_spread.notna() & (games.date >= today - pd.Timedelta(days=1))
    cands = []
    if upc.any():
        for idx, r in games.loc[upc].join(model.predict(feats.loc[upc, cols])).iterrows():
            pk = make_picks(r.home, r.away, r.pred_margin, r.pred_total, r.mkt_spread, r.mkt_total, r.mkt_ml_home, r.mkt_ml_away)
            cands.append({"game_id": _gid(r.game_id), "season": int(r.season), "week": int(r.week),
                          "kickoff": _kickoff(r), "away": r.away, "home": r.home, "pred_away": r.pred_away,
                          "pred_home": r.pred_home, "spread_play": pk["spread_play"], "total_play": pk["total_play"],
                          "mkt_spread": r.mkt_spread, "mkt_total": r.mkt_total,
                          "mkt_ml_home": r.mkt_ml_home, "mkt_ml_away": r.mkt_ml_away})
    spread_only_backfill(args.log or f"data/picks_{lg.name}.json", allr, games, current_season(), now)
    live, official = update_pick_log(args.log or f"data/picks_{lg.name}.json", lg.name, cands, games, now)
    all_live = update_all_log(f"data/allpicks_{lg.name}.json", lg.name, upcoming, games, now)
    live["all"] = all_live
    logged = {(x["season"], x["week"]) for x in live.get("archive", [])} | {
        (x["season"], x["week"]) for x in live["picks"] if x.get("season") is not None}
    live["archive"] = sorted(live.get("archive", []) + replay_weeks(allr, games, current_season(), logged),
                             key=lambda a: (-a["season"], -a["week"]))
    ytd_all = allr[allr.season == current_season()]
    live["all_bt"] = {"ytd": all_pick_blocks(ytd_all) if len(ytd_all) else None, "ytd_season": current_season(),
                      "past": all_pick_blocks(allr[allr.season.isin(seasons2)]), "seasons": seasons2,
                      "by_season": {str(int(x)): all_pick_blocks(allr[allr.season == x]) for x in seasons2}}
    ok = allr.dropna(subset=["actual_margin", "pred_margin"])
    sig_m = float((ok.actual_margin - ok.pred_margin).std())      # out-of-sample error of the model's margin / total
    sig_t = float((ok.actual_total - ok.pred_total).std())
    for u in upcoming:   # only the locked top plays are "official"; every game keeps its sim side
        u["spread_play"] = official.get(f"{u['game_id']}|spread")
        u["total_play"] = official.get(f"{u['game_id']}|total")
        sp_, tp_ = u["spread_play"], u["total_play"]
        # the logged play keeps its side and line, but the gap shown is the model's CURRENT gap vs the market
        if sp_ and pd.notna(u.get("vegas_spread")) and pd.notna(u.get("pred_margin")):
            home_side = sp_.get("team") == u["home"]
            pm = u["pred_margin"] if home_side else -u["pred_margin"]
            mm = u["vegas_spread"] if home_side else -u["vegas_spread"]
            sp_["edge_logged"], sp_["edge"] = sp_["edge"], round(float(pm - mm), 1)
        if tp_ and pd.notna(u.get("vegas_total")) and pd.notna(u.get("pred_total")):
            d_ = u["pred_total"] - u["vegas_total"]
            tp_["edge_logged"], tp_["edge"] = tp_["edge"], round(float(d_ if tp_.get("side") == "Over" else -d_), 1)
        if sp_:
            if sp_.get("kind") == "ml":
                hp = u["home_win_prob"]
                sp_["prob"] = hp if sp_["team"] == u["home"] else 1 - hp
            else:
                sp_["prob"] = float(norm.cdf(sp_["edge"] / sig_m))
        if tp_:
            tp_["prob"] = float(norm.cdf(tp_["edge"] / sig_t))

    # keep the five plays but order them by the model's CURRENT gap (played games keep their logged gap)
    byk2 = defaultdict(list)
    for pid, o in official.items():
        byk2[tuple(o.get("wk") or ())].append(o)
    for os_ in byk2.values():
        for r_, o in enumerate(sorted(os_, key=lambda o: -(o["edge"] if o.get("edge") is not None else -99)), 1):
            o["rank"] = r_
    payload = _clean({
        "league": lg.name, "generated_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {"kind": args.model, "groups": ["base"] + inc, "sigma": model.sigma},
        "games": upcoming, "recent": recent, "backtest": by_season, "backtest_record": bt2, "backtest_ytd": bt_ytd, "backtest_seasons": seasons2, "live": live, "calibration": hist_calibration(allr, games), "stats_season": stats_season, "stats_pool": pool_n,
        "rules": {"edge_play": EDGE_PLAY, "edge_strong": EDGE_STRONG, "top_n": TOP_N}})
    out = args.out or f"data/{lg.name}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Wrote {out}: {len(upcoming)} upcoming games, {len(recent)} recent results")


def main():
    ap = argparse.ArgumentParser(description="NFL / CFB score predictor")
    ap.add_argument("command", choices=["backtest", "ablate", "predict", "export"])
    ap.add_argument("--league", choices=["nfl", "cfb"], required=True)
    ap.add_argument("--model", choices=["ridge", "gbm", "ensemble"], default="ensemble")
    ap.add_argument("--start", type=int, help="first season to load")
    ap.add_argument("--test-seasons", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=None, help="Elo burn-in seasons excluded from training")
    ap.add_argument("--days", type=int, default=7, help="predict games in the next N days")
    ap.add_argument("--drop", default="injuries", help="comma list: advanced,qb,injuries,weather,roster (default drops the old injury counts; Opta ratings handle injuries)")
    ap.add_argument("--log", help="(export) pick log path, default data/picks_<league>.json")
    ap.add_argument("--out", help="(export) output JSON path, default data/<league>.json")
    ap.add_argument("--with", dest="with_groups", default="", help="opt-in feature groups: adjusted,turnovers,injw,injq")
    ap.add_argument("--use-market", action="store_true", help="(NFL) add Vegas spread/total as features")
    args = ap.parse_args()
    args.drop_set = {x.strip() for x in args.drop.split(",") if x.strip()}
    args.with_set = {x.strip() for x in args.with_groups.split(",") if x.strip()}
    {"backtest": cmd_backtest, "ablate": cmd_ablate, "predict": cmd_predict, "export": cmd_export}[args.command](args)


if __name__ == "__main__":
    main()
