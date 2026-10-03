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
    for r in d[(d.grp == "qb") & (d.w >= 0.75)].itertuples(index=False):
        ctx.qb_out.setdefault((int(r.season), int(r.week), r.team), set()).add(r.gsis_id)


def apply_forecasts(games: pd.DataFrame, days: int) -> pd.DataFrame:
    """Fill temp/wind for upcoming outdoor NFL games from the free Open-Meteo forecast API."""
    today = pd.Timestamp.today().normalize()
    need = (games.home_pts.isna() & games.temp.isna()
            & ~games.roof.isin(["dome", "closed"])
            & (games.date >= today - pd.Timedelta(days=1))
            & (games.date <= today + pd.Timedelta(days=min(days, 15))))
    cache, n_ok = {}, 0
    for idx in games.index[need]:
        g = games.loc[idx]
        if g.home not in NFL_STADIUMS:
            continue
        try:
            if g.home not in cache:
                lat, lon = NFL_STADIUMS[g.home]
                r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=20, params=dict(
                    latitude=lat, longitude=lon, hourly="temperature_2m,wind_speed_10m",
                    temperature_unit="fahrenheit", wind_speed_unit="mph",
                    timezone="UTC", forecast_days=16))
                r.raise_for_status()
                h = r.json()["hourly"]
                cache[g.home] = (pd.to_datetime(h["time"]), np.array(h["temperature_2m"], float),
                                 np.array(h["wind_speed_10m"], float))
            times, temps, winds = cache[g.home]
            gt = g.gametime if isinstance(g.gametime, str) and ":" in g.gametime else "13:00"
            ko = (pd.Timestamp(f"{g.date:%Y-%m-%d} {gt}", tz="America/New_York")
                  .tz_convert("UTC").tz_localize(None))
            i = int(np.abs((times - ko).total_seconds()).argmin())
            games.loc[idx, "temp"], games.loc[idx, "wind"] = temps[i], winds[i]
            n_ok += 1
        except Exception as e:  # network down, API change, etc. -> fall back to climatology
            warn(f"forecast failed for {g.home}: {e}")
            break
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
    _cfb_roster(ctx, start, end, key)
    return df


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

    def __init__(self, lg: League, ctx: Context, groups: set, talent_elo: bool = True):
        self.lg, self.ctx, self.groups = lg, ctx, groups
        self.talent_elo = talent_elo and lg.talent_scale > 0 and bool(ctx.talent_z)
        self.teams = {}
        self.pts_sum, self.pts_n = lg.default_pts * 50, 50
        self.adv_sum = {s: 0.0 for s in ADV_STATS}
        self.adv_cnt = {s: 0 for s in ADV_STATS}
        self.qb_stats = {}  # qb_id -> [n, epa_sum]

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

    def _qb_value(self, qb):
        if qb is None:
            return self.BACKUP_VAL, 0.0
        n, s = self.qb_stats.get(qb, (0.0, 0.0))
        return (s + self.QB_K * self.QB_PRIOR) / (n + self.QB_K), n

    def _qb(self, g, ht, at):
        f = {}
        for sd, t, team, cur in (("home", ht, g.home, g.home_qb_id), ("away", at, g.away, g.away_qb_id)):
            if pd.isna(cur):  # starter not known yet -> assume last starter unless he's Out/Doubtful
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


def prepare(args, forecast_days=None):
    lg = LEAGUES[args.league]
    start = args.start or lg.default_start
    end = current_season()
    ctx = Context()
    if args.league == "nfl":
        games = load_nfl(start)
        games = load_nfl_pbp(games, ctx, start, end)
        load_nfl_injuries(ctx, end)
        if forecast_days is not None:
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
    return lg, games, ctx


def featurize(lg, games, ctx, talent_elo=True):
    groups = set()
    if games[["h_epa", "a_epa"]].notna().any().any():
        groups.add("advanced")
    if games["home_qb_id"].notna().any():
        groups.add("qb")
    if ctx.team_season:
        groups.add("roster")
    feats = Builder(lg, ctx, groups, talent_elo).run(games)
    gc = {"base": BASE_COLS}
    if "advanced" in groups:
        gc["advanced"] = ADV_COLS
    if "qb" in groups:
        gc["qb"] = QB_COLS
    if "roster" in groups:
        gc["roster"] = ROSTER_COLS
    if ctx.inj_counts is not None:
        feats = feats.join(injury_features(games, ctx.inj_counts))
        gc["injuries"] = INJ_COLS
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
    inc = [g for g in gc if g not in ("base", "market") and g not in args.drop_set]
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
    lg, games, ctx = prepare(args)
    feats, gc = featurize(lg, games, ctx)
    feats_nt = featurize(lg, games, ctx, talent_elo=False)[0] if "roster" in gc else feats
    avail = [g for g in gc if g not in ("base", "market")]
    wu = args.warmup if args.warmup is not None else lg.default_warmup
    configs = [("base only", [])] + [(f"base + {g}", [g]) for g in avail] + [("ALL groups", avail)] + \
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


def wilson(w, n, z=1.96):
    if n == 0:
        return None, None
    p, d = w / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def american_prob(ml):
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def make_picks(home, away, pred_margin, pred_total, v_spread, v_total):
    """v_spread = Vegas expected home margin (+ = home favored)."""
    out = {"sim_spread": None, "sim_total": None, "spread_play": None, "total_play": None}
    if pd.notna(v_spread):
        e = pred_margin - v_spread
        team, line = (home, -v_spread) if e > 0 else (away, v_spread)
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


def rec(res, units=None):
    res = np.asarray(res, float)
    u = np.ones(len(res)) if units is None else np.asarray(units, float)
    w, l = int((res == 1).sum()), int((res == -1).sum())
    lo, hi = wilson(w, w + l)
    net = float((u * np.where(res == 1, WIN_UNITS, np.where(res == -1, -1.0, 0.0))).sum())
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
    if e["type"] == "spread":
        d = ((hp - ap) if e["team"] == e["home"] else (ap - hp)) + e["line"]
    else:
        tot = hp + ap
        d = tot - e["line"] if e["side"] == "Over" else e["line"] - tot
    res = 1 if d > 0 else -1 if d < 0 else 0
    return res, e["units"] * (WIN_UNITS if res == 1 else -1.0 if res == -1 else 0.0)


TOP_N, LOCK_DAYS = 5, 3   # official plays per week; a week's plays lock this many days before its first kickoff


def _utc(k):
    t = pd.Timestamp(k)
    return t.tz_localize("UTC") if t.tzinfo is None else t


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
                for typ in ("spread", "total") for p in [u.get(f"{typ}_play")]
                if p and f"{u['game_id']}|{typ}" not in have]
        pool.sort(key=lambda x: -x[0])
        for edge, typ, u, p in pool[:TOP_N - cnt[k]]:
            log.append(_clean({"id": f"{u['game_id']}|{typ}", "type": typ, "game_id": u["game_id"], "league": league,
                               "season": u["season"], "week": u["week"], "kickoff": u["kickoff"],
                               "away": u["away"], "home": u["home"], "team": p.get("team"), "side": p.get("side"),
                               "line": p["line"], "edge": p["edge"], "units": p["units"],
                               "pred_away": u["pred_away"], "pred_home": u["pred_home"], "status": "pending",
                               "logged_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")}))

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
        return rec([e["result"] for e in xs], [e["units"] for e in xs])
    series, cum = [], 0.0
    for e in done:
        cum += e["net"]
        day = e["kickoff"][:10]
        if series and series[-1][0] == day:
            series[-1][1] = round(cum, 2)
        else:
            series.append([day, round(cum, 2)])
    live = {"record": {"all": sub(), "spread": sub("spread"), "total": sub("total")}, "series": series,
            "picks": pend + done[::-1][:50], "pending": len(pend),
            "since": min([e["logged_at"][:10] for e in log], default=None)}
    official, byk = {}, defaultdict(list)
    for e in log:
        byk[wk(e)].append(e)
    for es in byk.values():
        for r, e in enumerate(sorted(es, key=lambda e: -e["edge"]), 1):
            official[e["id"]] = {"team": e.get("team"), "side": e.get("side"), "line": e["line"],
                                 "edge": e["edge"], "units": e["units"], "rank": r, "of": TOP_N}
    return live, official


def play_frame(allr, games):
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
    t = d[d.mkt_total.notna()]
    et = t.pred_total - t.mkt_total
    tp = t[et.abs() >= EDGE_PLAY].copy()
    tp["edge"] = et[tp.index].abs()
    tp["type"] = "total"
    tp["res"] = np.where(et[tp.index] > 0, 1, -1) * np.sign(tp.actual_total - tp.mkt_total)
    tp["units"] = np.where(et[tp.index].abs() >= EDGE_STRONG, 1.5, 1.0)
    cols = ["date", "season", "week", "type", "res", "units", "edge"]
    out = pd.concat([sp[cols], tp[cols]])
    out = out.sort_values("edge", ascending=False).groupby(["season", "week"]).head(TOP_N).sort_values("date")
    out["net"] = out.units * np.where(out.res == 1, WIN_UNITS, np.where(out.res == -1, -1.0, 0.0))
    return out


def backtest_block(pf):
    def three(g):
        return {"all": rec(g.res, g.units), "spread": rec(g[g.type == "spread"].res, g[g.type == "spread"].units),
                "total": rec(g[g.type == "total"].res, g[g.type == "total"].units)}
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


def team_card(side, team, f, ts, qbn):
    c = dict(ts.get(team, {"games": 0, "record": "0-0", "s": {}}))
    c["elo"], c["rest"] = f.get(f"{side}_elo"), f.get(f"{side}_rest")
    if f"{side}_qb_val" in f.index:
        c["qb"] = {"name": qbn.get(team), "val": f[f"{side}_qb_val"], "flag": bool(f[f"{side}_qb_delta"] <= -0.05)}
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
    upcoming = []
    if up.any():
        sub = games.loc[up].join(model.predict(feats.loc[up, cols])).sort_values("date")
        for idx, r in sub.iterrows():
            f = feats.loc[idx]
            wx = None
            if "weather" in gc:
                wx = {"dome": bool(f.is_dome), "temp": f.temp, "wind": f.wind, "known": bool(f.weather_known)}
            upcoming.append({"kickoff": _kickoff(r), "week": int(r.week), "game_id": _gid(r.game_id), "away": r.away, "home": r.home,
                             "neutral": bool(r.neutral), "pred_away": r.pred_away, "pred_home": r.pred_home,
                             "pred_margin": r.pred_margin, "pred_total": r.pred_total,
                             "home_win_prob": r.home_win_prob, "weather": wx,
                             "home_info": team_card("home", r.home, f, ts, qbn), "away_info": team_card("away", r.away, f, ts, qbn),
                             "vegas_spread": r.mkt_spread, "vegas_total": r.mkt_total,
                             "ml_home": r.mkt_ml_home, "ml_away": r.mkt_ml_away, **ml_info(r),
                             **make_picks(r.home, r.away, r.pred_margin, r.pred_total, r.mkt_spread, r.mkt_total)})

    allr = run_backtest(lg, games, feats, cols, args.model, max(args.test_seasons, 4), wu)
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
            pk = make_picks(r.home, r.away, r.pred_margin, r.pred_total, r.mkt_spread, r.mkt_total)
            cands.append({"game_id": _gid(r.game_id), "season": int(r.season), "week": int(r.week),
                          "kickoff": _kickoff(r), "away": r.away, "home": r.home, "pred_away": r.pred_away,
                          "pred_home": r.pred_home, "spread_play": pk["spread_play"], "total_play": pk["total_play"]})
    live, official = update_pick_log(args.log or f"data/picks_{lg.name}.json", lg.name, cands, games, now)
    for u in upcoming:   # only the locked top plays are "official"; every game keeps its sim side
        u["spread_play"] = official.get(f"{u['game_id']}|spread")
        u["total_play"] = official.get(f"{u['game_id']}|total")

    payload = _clean({
        "league": lg.name, "generated_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {"kind": args.model, "groups": ["base"] + inc, "sigma": model.sigma},
        "games": upcoming, "recent": recent, "backtest": by_season, "backtest_record": bt2, "backtest_ytd": bt_ytd, "backtest_seasons": seasons2, "live": live, "stats_season": stats_season, "stats_pool": pool_n,
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
    ap.add_argument("--drop", default="", help="comma list: advanced,qb,injuries,weather,roster")
    ap.add_argument("--log", help="(export) pick log path, default data/picks_<league>.json")
    ap.add_argument("--out", help="(export) output JSON path, default data/<league>.json")
    ap.add_argument("--use-market", action="store_true", help="(NFL) add Vegas spread/total as features")
    args = ap.parse_args()
    args.drop_set = {x.strip() for x in args.drop.split(",") if x.strip()}
    {"backtest": cmd_backtest, "ablate": cmd_ablate, "predict": cmd_predict, "export": cmd_export}[args.command](args)


if __name__ == "__main__":
    main()
