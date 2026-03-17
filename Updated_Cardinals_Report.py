# Updated_Cardinals_Report.py
# To run: python Updated_Cardinals_Report.py --season 2025 --team ARI
#
# Generates an EPA leverage snapshot for any NFL team:
#   - Identifies situational buckets where the team underperforms league average
#   - Ranks gaps by (EPA/play gap) × (play volume) to prioritize highest-impact areas
#   - Lists relevant free agents at positions tied to each gap
#   - Outputs: CSV, bar charts (PNG), and an HTML report

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(0)


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    season: int
    team: str
    season_type: str = "REG"
    min_team_plays: int = 30
    min_league_plays: int = 200
    top_n: int = 6
    explosive_pass_yards: int = 20
    explosive_run_yards: int = 10
    out_dir: str = "./out"
    write_html: bool = True
    chart_exclude_buckets: Tuple[str, ...] = (
        "OFF: All plays",
        "DEF: All plays",
        "OFF: Explosive rate context",
        "DEF: Explosives allowed ctx",
    )


# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------

def setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("leverage_report")
    logger.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------
# Data Loading + Preprocessing
# ---------------------------------------------------------------

def load_pbp_data(season: int, logger: logging.Logger) -> pd.DataFrame:
    # Try loading from local cache first
    cache_dir = os.path.join(os.path.expanduser("~"), ".nfl_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"pbp_{season}.parquet")

    if os.path.exists(cache_path):
        logger.info(f"Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    try:
        from nflreadpy import load_pbp
    except Exception as e:
        raise RuntimeError(f"Failed to import nflreadpy: {e}")

    logger.info(f"Downloading {season} play-by-play (this only happens once)...")
    pbp = load_pbp(seasons=[season])
    if hasattr(pbp, "to_pandas"):
        pbp = pbp.to_pandas()

    # Save to cache
    pbp.to_parquet(cache_path)
    logger.info(f"Cached to {cache_path}")
    logger.info(f"Loaded pbp: {pbp.shape[0]:,} rows")
    return pbp


def validate_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")


def filter_real_plays(pbp: pd.DataFrame, cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    validate_cols(pbp, ["season_type", "play_type", "epa"])
    df = pbp[(pbp["season_type"] == cfg.season_type) &
             (pbp["play_type"].isin(["run", "pass"])) &
             (pbp["epa"].notna())].copy()
    for col in ["qb_kneel", "qb_spike", "no_play"]:
        if col in df.columns:
            df = df[df[col] != 1]
    logger.info(f"After filters: {df.shape[0]:,} plays")
    return df


def compute_flags(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    validate_cols(df, ["down", "ydstogo", "yardline_100", "yards_gained", "play_type"])
    out = df.copy()
    out["is_early_down"] = out["down"].isin([1, 2])
    out["is_third_down"] = out["down"] == 3
    out["is_red_zone"] = out["yardline_100"].between(1, 20, inclusive="both")
    out["is_goal_to_go"] = out["goal_to_go"].eq(1) if "goal_to_go" in out.columns else False
    out["is_explosive"] = np.where(
        out["play_type"].eq("pass"),
        out["yards_gained"] >= cfg.explosive_pass_yards,
        out["yards_gained"] >= cfg.explosive_run_yards,
    )
    out["third_short"] = out["is_third_down"] & (out["ydstogo"] <= 4)
    out["third_long"] = out["is_third_down"] & (out["ydstogo"] >= 7)
    out["success"] = (out["epa"] > 0).astype(int)
    return out


# ---------------------------------------------------------------
# Bucket Definitions + Summaries
# ---------------------------------------------------------------

BucketFn = Callable[[pd.DataFrame, str], pd.Series]

def build_bucket_definitions() -> Dict[str, BucketFn]:
    return {
        "OFF: All plays":               lambda d, t: d["posteam"].eq(t),
        "OFF: Early downs":             lambda d, t: d["posteam"].eq(t) & d["is_early_down"],
        "OFF: 3rd down":                lambda d, t: d["posteam"].eq(t) & d["is_third_down"],
        "OFF: 3rd & short (<=4)":       lambda d, t: d["posteam"].eq(t) & d["third_short"],
        "OFF: 3rd & long (>=7)":        lambda d, t: d["posteam"].eq(t) & d["third_long"],
        "OFF: Red zone (<=20)":         lambda d, t: d["posteam"].eq(t) & d["is_red_zone"],
        "OFF: Goal-to-go":              lambda d, t: d["posteam"].eq(t) & d["is_goal_to_go"],
        "DEF: All plays":               lambda d, t: d["defteam"].eq(t),
        "DEF: Early downs":             lambda d, t: d["defteam"].eq(t) & d["is_early_down"],
        "DEF: 3rd down":                lambda d, t: d["defteam"].eq(t) & d["is_third_down"],
        "DEF: 3rd & short (<=4)":       lambda d, t: d["defteam"].eq(t) & d["third_short"],
        "DEF: Red zone (<=20)":         lambda d, t: d["defteam"].eq(t) & d["is_red_zone"],
        "OFF: Explosive rate context":  lambda d, t: d["posteam"].eq(t),
        "DEF: Explosives allowed ctx":  lambda d, t: d["defteam"].eq(t),
    }


def league_mask(bucket_name: str, d: pd.DataFrame) -> pd.Series:
    if bucket_name.startswith("OFF:"):
        if "Early downs" in bucket_name: return d["is_early_down"]
        if "3rd & short" in bucket_name: return d["third_short"]
        if "3rd & long"  in bucket_name: return d["third_long"]
        if "3rd down"    in bucket_name: return d["is_third_down"]
        if "Red zone"    in bucket_name: return d["is_red_zone"]
        if "Goal-to-go"  in bucket_name: return d["is_goal_to_go"]
        return d["play_type"].isin(["pass", "run"])
    if "Early downs" in bucket_name: return d["is_early_down"]
    if "3rd & short" in bucket_name: return d["third_short"]
    if "3rd down"    in bucket_name: return d["is_third_down"]
    if "Red zone"    in bucket_name: return d["is_red_zone"]
    return d["play_type"].isin(["pass", "run"])


def summarize_bucket(d, bucket_name, mask_team, cfg):
    team_df = d[mask_team]
    lg_df   = d[league_mask(bucket_name, d)]
    if len(team_df) < cfg.min_team_plays or len(lg_df) < cfg.min_league_plays:
        return None
    t_epa, l_epa, n = float(team_df["epa"].mean()), float(lg_df["epa"].mean()), len(team_df)
    swing = (l_epa - t_epa) * n if bucket_name.startswith("OFF:") else (t_epa - l_epa) * n
    return {
        "bucket": bucket_name, "plays": n,
        "TEAM_EPA_per_play": t_epa, "LG_EPA_per_play": l_epa,
        "TEAM_success_rate": float(team_df["success"].mean()),
        "LG_success_rate":   float(lg_df["success"].mean()),
        "EPA_gap_to_LG":     float(swing),
    }


def build_leverage_table(d, cfg, logger):
    validate_cols(d, ["posteam", "defteam", "epa"])
    buckets = build_bucket_definitions()
    rows = []
    for name, fn in tqdm(buckets.items(), desc="Summarizing"):
        s = summarize_bucket(d, name, fn(d, cfg.team), cfg)
        if s:
            rows.append(s)
    if not rows:
        raise RuntimeError("No bucket summaries produced.")
    out = pd.DataFrame(rows).sort_values("EPA_gap_to_LG", ascending=False)
    logger.info(f"Built leverage table: {len(out)} buckets")
    return out


# ---------------------------------------------------------------
# League Ranks
# ---------------------------------------------------------------

NFL_TEAMS = [
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
    "HOU","IND","JAX","KC","LA","LAC","LV","MIA","MIN","NE","NO","NYG",
    "NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS",
]

def compute_league_ranks(d, cfg, logger):
    import time
    t0 = time.time()
    buckets = build_bucket_definitions()

    # Pre-compute team column for faster lookups
    posteam = d["posteam"].values
    defteam = d["defteam"].values
    epa_vals = d["epa"].values

    records = []
    for bname, fn in buckets.items():
        if bname in cfg.chart_exclude_buckets:
            continue
        is_off = bname.startswith("OFF:")

        # Build the situation mask once (not per-team)
        sit_mask = league_mask(bname, d).values
        team_col = posteam if is_off else defteam

        epas = {}
        for t in NFL_TEAMS:
            mask = sit_mask & (team_col == t)
            n = mask.sum()
            if n >= cfg.min_team_plays:
                epas[t] = float(epa_vals[mask].mean())

        if cfg.team not in epas or len(epas) < 5:
            continue
        ranked = sorted(epas.items(), key=lambda x: x[1], reverse=is_off)
        rmap = {t: i + 1 for i, (t, _) in enumerate(ranked)}
        records.append({"bucket": bname, "rank": rmap[cfg.team], "total_teams": len(epas)})

    logger.info(f"Computed ranks for {len(records)} buckets in {time.time()-t0:.1f}s")
    return pd.DataFrame(records)


def rank_suffix(n):
    if 11 <= (n % 100) <= 13: return f"{n}th"
    return {1: f"{n}st", 2: f"{n}nd", 3: f"{n}rd"}.get(n % 10, f"{n}th")


# ---------------------------------------------------------------
# Free Agent Data (simplified — factual only, no modeled values)
# ---------------------------------------------------------------

FA_CB = pd.DataFrame([
    {"player_name":"Alontae Taylor","position":"CB","age":27.1,"prev_team":"NO","prev_aav_usd":1801175},
    {"player_name":"Roger McCreary","position":"CB","age":25.9,"prev_team":"LAR","prev_aav_usd":2291402},
    {"player_name":"Greg Newsome","position":"CB","age":25.7,"prev_team":"JAX","prev_aav_usd":3187184},
    {"player_name":"Eric Stokes","position":"CB","age":26.8,"prev_team":"LV","prev_aav_usd":3500000},
    {"player_name":"Dee Alford","position":"CB","age":28.2,"prev_team":"ATL","prev_aav_usd":1500000},
    {"player_name":"Jeff Okudah","position":"CB","age":26.9,"prev_team":"MIN","prev_aav_usd":2350000},
    {"player_name":"Jamel Dean","position":"CB","age":29.2,"prev_team":"TB","prev_aav_usd":13000000},
    {"player_name":"Rasul Douglas","position":"CB","age":31.3,"prev_team":"MIA","prev_aav_usd":1572500},
])

FA_LB = pd.DataFrame([
    {"player_name":"Devin Lloyd","position":"LB","age":27.2,"prev_team":"JAX","prev_aav_usd":3234151},
    {"player_name":"Quay Walker","position":"LB","age":25.7,"prev_team":"GB","prev_aav_usd":3460410},
    {"player_name":"Devin Bush","position":"LB","age":27.5,"prev_team":"CLE","prev_aav_usd":3250000},
    {"player_name":"Quincy Williams","position":"LB","age":29.3,"prev_team":"NYJ","prev_aav_usd":6000000},
    {"player_name":"Kaden Elliss","position":"LB","age":30.5,"prev_team":"ATL","prev_aav_usd":7166667},
    {"player_name":"Alex Anzalone","position":"LB","age":31.3,"prev_team":"DET","prev_aav_usd":6250000},
    {"player_name":"Kenneth Murray","position":"LB","age":27.2,"prev_team":"DAL","prev_aav_usd":7750000},
    {"player_name":"Cole Holcomb","position":"LB","age":29.4,"prev_team":"PIT","prev_aav_usd":6000000},
])

# Bucket-specific FA pools so each section shows different, relevant players.
# Players are hand-assigned to the bucket where their skillset fits best.

FA_OFF_3RD_LONG = pd.DataFrame([
    {"player_name":"Keenan Allen","position":"WR","age":34,"prev_team":"LAC","prev_aav_usd":3020000},
    {"player_name":"Christian Kirk","position":"WR","age":30,"prev_team":"HOU","prev_aav_usd":18000000},
    {"player_name":"Wan'Dale Robinson","position":"WR","age":25,"prev_team":"NYG","prev_aav_usd":2046292},
])

FA_OFF_RED_ZONE = pd.DataFrame([
    {"player_name":"Jauan Jennings","position":"WR","age":29,"prev_team":"SF","prev_aav_usd":5945000},
    {"player_name":"Jahan Dotson","position":"WR","age":26,"prev_team":"PHI","prev_aav_usd":3762089},
    {"player_name":"Adam Thielen","position":"WR","age":36,"prev_team":"PIT","prev_aav_usd":5000000},
])

FA_OFF_EARLY_DOWN = pd.DataFrame([
    {"player_name":"Travis Etienne","position":"RB","age":27,"prev_team":"JAX","prev_aav_usd":3224528},
    {"player_name":"Breece Hall","position":"RB","age":25,"prev_team":"NYJ","prev_aav_usd":2253692},
    {"player_name":"Kenneth Walker III","position":"RB","age":26,"prev_team":"SEA","prev_aav_usd":2110395},
])

FA_OFF_GOAL = pd.DataFrame([
    {"player_name":"Najee Harris","position":"RB","age":28,"prev_team":"LAC","prev_aav_usd":5250000},
    {"player_name":"Jauan Jennings","position":"WR","age":29,"prev_team":"SF","prev_aav_usd":5945000},
    {"player_name":"Antonio Gibson","position":"RB","age":28,"prev_team":"NE","prev_aav_usd":3750000},
])

FA_OFF_EXPLOSIVE = pd.DataFrame([
    {"player_name":"Alec Pierce","position":"WR","age":26,"prev_team":"IND","prev_aav_usd":1650336},
    {"player_name":"Rashid Shaheed","position":"WR","age":28,"prev_team":"SEA","prev_aav_usd":3092500},
    {"player_name":"Travis Etienne","position":"RB","age":27,"prev_team":"JAX","prev_aav_usd":3224528},
])

FA_DEF_EARLY_DOWN = pd.DataFrame([
    {"player_name":"Devin Lloyd","position":"LB","age":27.2,"prev_team":"JAX","prev_aav_usd":3234151},
    {"player_name":"Quay Walker","position":"LB","age":25.7,"prev_team":"GB","prev_aav_usd":3460410},
    {"player_name":"Devin Bush","position":"LB","age":27.5,"prev_team":"CLE","prev_aav_usd":3250000},
])

FA_DEF_3RD_SHORT = pd.DataFrame([
    {"player_name":"Alontae Taylor","position":"CB","age":27.1,"prev_team":"NO","prev_aav_usd":1801175},
    {"player_name":"Roger McCreary","position":"CB","age":25.9,"prev_team":"LAR","prev_aav_usd":2291402},
    {"player_name":"Jamel Dean","position":"CB","age":29.2,"prev_team":"TB","prev_aav_usd":13000000},
])

FA_DEF_RED_ZONE = pd.DataFrame([
    {"player_name":"Devin Lloyd","position":"LB","age":27.2,"prev_team":"JAX","prev_aav_usd":3234151},
    {"player_name":"Alontae Taylor","position":"CB","age":27.1,"prev_team":"NO","prev_aav_usd":1801175},
    {"player_name":"Greg Newsome","position":"CB","age":25.7,"prev_team":"JAX","prev_aav_usd":3187184},
])

FA_DEF_3RD = pd.DataFrame([
    {"player_name":"Dee Alford","position":"CB","age":28.2,"prev_team":"ATL","prev_aav_usd":1500000},
    {"player_name":"Eric Stokes","position":"CB","age":26.8,"prev_team":"LV","prev_aav_usd":3500000},
    {"player_name":"Jeff Okudah","position":"CB","age":26.9,"prev_team":"MIN","prev_aav_usd":2350000},
])

FA_DEF_EXPLOSIVE = pd.DataFrame([
    {"player_name":"Roger McCreary","position":"CB","age":25.9,"prev_team":"LAR","prev_aav_usd":2291402},
    {"player_name":"Jamel Dean","position":"CB","age":29.2,"prev_team":"TB","prev_aav_usd":13000000},
    {"player_name":"Rasul Douglas","position":"CB","age":31.3,"prev_team":"MIA","prev_aav_usd":1572500},
])

# Map bucket keywords + side to the specific FA pool
def get_fa_targets(lbl: str, side: str, top_n: int = 3) -> pd.DataFrame:
    ll = lbl.lower()
    if side == "OFF":
        if "3rd" in ll and "long" in ll:   return FA_OFF_3RD_LONG.head(top_n)
        if "3rd" in ll and "short" in ll:  return FA_OFF_3RD_LONG.head(top_n)  # same concept — separation WRs
        if "red zone" in ll:               return FA_OFF_RED_ZONE.head(top_n)
        if "goal" in ll:                   return FA_OFF_GOAL.head(top_n)
        if "early down" in ll:             return FA_OFF_EARLY_DOWN.head(top_n)
        if "explosive" in ll:              return FA_OFF_EXPLOSIVE.head(top_n)
    else:
        if "3rd" in ll and "short" in ll:  return FA_DEF_3RD_SHORT.head(top_n)
        if "3rd" in ll:                    return FA_DEF_3RD.head(top_n)
        if "early down" in ll:             return FA_DEF_EARLY_DOWN.head(top_n)
        if "red zone" in ll:               return FA_DEF_RED_ZONE.head(top_n)
        if "explosive" in ll:              return FA_DEF_EXPLOSIVE.head(top_n)
    return pd.DataFrame()


def fa_html(lbl: str, side: str, top_n: int = 3) -> str:
    df = get_fa_targets(lbl, side, top_n)
    if df.empty:
        return ""
    rows_html = ""
    for _, r in df.iterrows():
        prev = f"${r['prev_aav_usd']/1e6:.1f}M"
        rows_html += (
            f"<tr>"
            f"<td><b>{r['player_name']}</b></td>"
            f"<td>{r['position']} | Age {r['age']}</td>"
            f"<td>{r['prev_team']}</td>"
            f"<td>{prev}</td>"
            f"</tr>"
        )
    return f"""
<div class="fa-block">
  <div class="fa-title">Relevant Free Agents</div>
  <table class="fa-table">
    <thead><tr>
      <th>Player</th><th>Pos / Age</th><th>Prev Team</th><th>Prior AAV</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""


# ---------------------------------------------------------------
# Explosives
# ---------------------------------------------------------------

def explosive_rates(d, team):
    off  = d[d["posteam"].eq(team)]
    deff = d[d["defteam"].eq(team)]
    return {
        "expl_off_rate":         float(off["is_explosive"].mean())  if len(off)  else float("nan"),
        "expl_def_allowed_rate": float(deff["is_explosive"].mean()) if len(deff) else float("nan"),
        "expl_lg_rate":          float(d["is_explosive"].mean())    if len(d)    else float("nan"),
    }


# ---------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------

def bucket_label(b):
    if not b: return "overall"
    label = b.split(":", 1)[-1].strip()
    label = label.replace("<=", "≤").replace(">=", "≥")
    words = label.split()
    cleaned = []
    for w in words:
        if w[0].isdigit() or w.lower() in ("and", "or", "the", "a", "an", "&"):
            cleaned.append(w.lower())
        else:
            cleaned.append(w.capitalize())
    return " ".join(cleaned)


def generate_narrative(cfg, leverage_table, ranks_df, expl):
    team    = cfg.team
    exclude = set(cfg.chart_exclude_buckets)

    def side_rows(prefix):
        return leverage_table[
            leverage_table["bucket"].str.startswith(prefix) &
            ~leverage_table["bucket"].isin(exclude) &
            (leverage_table["EPA_gap_to_LG"] > 0)
        ].sort_values("EPA_gap_to_LG", ascending=False)

    def_rows = side_rows("DEF:")
    off_rows = side_rows("OFF:")

    def get_rank_str(bucket):
        if ranks_df.empty: return ""
        row = ranks_df[ranks_df["bucket"] == bucket]
        if row.empty: return ""
        r, tot = int(row.iloc[0]["rank"]), int(row.iloc[0]["total_teams"])
        return f" ({rank_suffix(r)} of {tot})"

    # Bottom-line bullets (no wins conversion — just EPA and rank)
    bl = []
    if not def_rows.empty:
        r = def_rows.iloc[0]
        bl.append(
            f"<b>Largest gap is on defense:</b> {bucket_label(r['bucket'])}"
            f"{get_rank_str(r['bucket'])} — {r['EPA_gap_to_LG']:.0f} total EPA below league average."
        )
    if not off_rows.empty:
        r = off_rows.iloc[0]
        bl.append(
            f"<b>Top offensive gap:</b> {bucket_label(r['bucket'])}"
            f"{get_rank_str(r['bucket'])} — {r['EPA_gap_to_LG']:.0f} total EPA below league average."
        )
    lg_e  = expl.get("expl_lg_rate", float("nan"))
    off_e = expl.get("expl_off_rate", float("nan"))
    def_e = expl.get("expl_def_allowed_rate", float("nan"))
    if not any(np.isnan(v) for v in [lg_e, off_e, def_e]):
        bl.append(
            f"<b>Explosive plays are a two-sided issue:</b> {team} creates them at "
            f"{off_e:.1%} (league avg {lg_e:.1%}) and allows them at {def_e:.1%}."
        )

    bl_html = '<ol class="bl-list">\n'
    for b in bl:
        bl_html += f"  <li>{b}</li>\n"
    bl_html += "</ol>"

    # Archetype bullets
    def build_bullets(rows, side):
        bullets = []
        for _, row in rows.head(3).iterrows():
            lbl    = bucket_label(row["bucket"])
            rank_s = get_rank_str(row["bucket"])
            gap    = row["EPA_gap_to_LG"]
            t_epa  = row["TEAM_EPA_per_play"]
            lg_epa = row["LG_EPA_per_play"]
            stat   = f"{t_epa:.3f} vs {lg_epa:.3f} LG EPA/play — {gap:.0f} total EPA gap"
            ll     = lbl.lower()

            if side == "DEF":
                if   "early down" in ll: arch = "Early-down defense is often driven by front-seven depth — the ability to win at the point of attack on 1st and 2nd down limits opponent drive length."
                elif "red zone"   in ll: arch = "Red zone defense in condensed space typically correlates with interior run defense and press coverage ability."
                elif "short"      in ll: arch = "Short-yardage 3rd-down defense is often tied to pressure rate and the ability to disrupt timing routes at the line of scrimmage."
                elif "3rd"        in ll: arch = "Third-down defense correlates with pass rush and coverage discipline — getting the QB off his spot and limiting YAC."
                elif "explosive"  in ll: arch = "Explosive plays allowed correlate with deep safety range and cornerback length."
                else:                    arch = "Further film study is needed to determine whether this gap is personnel or scheme driven."
            else:
                if   "3rd" in ll and "long" in ll:
                    arch = "3rd-and-long conversion rate is heavily influenced by QB accuracy on intermediate throws and receiver separation ability."
                elif "3rd" in ll and "short" in ll:
                    arch = "Short-yardage offensive conversion often comes down to run-game push and quick-game passing concepts."
                elif "3rd"        in ll: arch = "Overall 3rd-down rate is driven by a combination of QB play, receiver separation, and play design."
                elif "red zone"   in ll: arch = "Red zone offense in condensed space often benefits from contested-catch receivers and goal-line run-game quality."
                elif "goal"       in ll: arch = "Goal-to-go efficiency is heavily influenced by offensive line push and short-yardage play design."
                elif "early down" in ll: arch = "Early-down offense sets up the rest of the drive — run-game efficiency and play-action success are common levers here."
                elif "explosive"  in ll: arch = "Explosive play creation correlates with receiver speed and the ability to attack downfield."
                else:                    arch = "Further study needed to determine whether this gap is personnel or scheme driven."

            bullets.append((lbl, f"{rank_s} &nbsp;|&nbsp; {stat}", arch, side))
        return bullets

    return {
        "bl_html":      bl_html,
        "def_bullets":  build_bullets(def_rows, "DEF"),
        "off_bullets":  build_bullets(off_rows, "OFF"),
    }


# ---------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------

def plot_top_bars(table, prefix, cfg, title, subtitle, out_path):
    df = (
        table[
            table["bucket"].str.startswith(prefix) &
            ~table["bucket"].isin(cfg.chart_exclude_buckets) &
            (table["EPA_gap_to_LG"] > 0)
        ]
        .sort_values("EPA_gap_to_LG", ascending=False)
        .head(cfg.top_n)
        .copy()
    )
    if df.empty:
        return

    labels = []
    for _, r in df.iterrows():
        base = r["bucket"].replace(prefix, "").strip()
        base = base.replace("(<=20)","").replace("(<=4)","").replace("(>=7)","").replace("&","and")
        labels.append(f"{base}  ({int(r['plays'])} plays)")

    vals = df["EPA_gap_to_LG"].to_numpy()
    n = len(vals)
    fig, ax = plt.subplots(figsize=(16, max(6, n * 1.6 + 2.5)))
    y = np.arange(n)
    ax.barh(y, vals, height=0.55, color='#d32f2f', alpha=0.88, edgecolor='white', linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13)
    ax.axvline(0, color='black', linewidth=1.2, alpha=0.3)
    ax.set_xlabel("EPA Gap to League Average (higher = larger opportunity)", fontsize=13, fontweight='bold', labelpad=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=12)
    fig.text(0.5, 0.98, subtitle, ha='center', va='top', fontsize=11, color='#555', style='italic')
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    mx = max(vals)
    ax.set_xlim(0, mx * 1.18)
    for i, v in enumerate(vals):
        ax.text(v + mx * 0.015, i, f"{v:.1f}", va="center", ha="left", fontsize=12, fontweight='bold', color='#8b0000')
    fig.text(0.5, 0.01, "Bars show where team underperformed league average, scaled by play volume",
             ha='center', fontsize=9.5, style='italic', color='#888')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()


# ---------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------

def write_html_report(cfg, leverage_table, expl, ranks_df, def_chart, off_chart, out_path):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
    team = cfg.team
    exclude = set(cfg.chart_exclude_buckets)
    narrative = generate_narrative(cfg, leverage_table, ranks_df, expl)

    def prep_table(prefix):
        df = leverage_table[
            leverage_table["bucket"].str.startswith(prefix) &
            ~leverage_table["bucket"].isin(exclude) &
            (leverage_table["EPA_gap_to_LG"] > 0)
        ].head(5).copy()
        if not ranks_df.empty:
            df = df.merge(ranks_df[["bucket","rank","total_teams"]], on="bucket", how="left")
            df["Rank"] = df.apply(
                lambda r: f"{rank_suffix(int(r['rank']))}/{int(r['total_teams'])}"
                if pd.notna(r.get("rank")) else "—", axis=1)
            df = df.drop(columns=["rank","total_teams"], errors="ignore")
        for col in ["TEAM_EPA_per_play","LG_EPA_per_play","TEAM_success_rate","LG_success_rate","EPA_gap_to_LG"]:
            if col in df.columns:
                df[col] = df[col].round(3)
        return df

    def bullets_html(bullets):
        items = ""
        for h, rank_s, arch, side in bullets:
            items += (
                f'  <li><b>{h}</b> <span class="rank-tag">{rank_s}</span>'
                f'<div class="archetype">{arch}</div>'
                f'{fa_html(h.lower(), side)}'
                f'</li>\n'
            )
        return f"<ul>\n{items}</ul>"

    top_def = prep_table("DEF:")
    top_off = prep_table("OFF:")

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{team} EPA Leverage Snapshot — {cfg.season} {cfg.season_type}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 28px 40px; background: #f7f7f7; color: #111; line-height: 1.5; }}

    .header {{ background: #1a1a1a; color: white; padding: 18px 28px; border-radius: 6px; margin-bottom: 20px; display: flex; align-items: baseline; gap: 18px; }}
    .header h1 {{ margin: 0; font-size: 22px; }}
    .header .sub {{ color: #aaa; font-size: 12px; margin: 0; }}

    .intro {{ background: white; border-radius: 6px; padding: 18px 24px; margin-bottom: 20px; border-left: 4px solid #1a1a1a; }}
    .intro h2 {{ margin: 0 0 10px 0; font-size: 15px; color: #222; }}
    .intro ul {{ margin: 0; padding-left: 20px; font-size: 13px; line-height: 1.8; color: #333; }}
    .intro li {{ margin-bottom: 4px; }}
    .intro .note {{ font-size: 11px; color: #888; margin-top: 10px; font-style: italic; }}

    .top-strip {{ display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-bottom: 20px; }}
    .exec-summary {{ background: white; border-left: 4px solid #d32f2f; padding: 14px 18px; font-size: 13.5px; line-height: 1.75; border-radius: 4px; }}
    .bl-title {{ font-size: 11px; font-weight: bold; text-transform: uppercase; color: #d32f2f; letter-spacing: 0.08em; margin-bottom: 8px; }}
    .bl-list {{ margin: 0; padding-left: 18px; }}
    .bl-list li {{ margin-bottom: 6px; font-size: 13.5px; line-height: 1.6; }}
    .right-panel {{ display: flex; flex-direction: column; gap: 12px; }}

    .callouts {{ background: white; border-radius: 4px; padding: 14px 16px; font-size: 13px; }}
    .callouts b {{ display: block; margin-bottom: 6px; font-size: 12px; text-transform: uppercase; color: #888; letter-spacing: 0.05em; }}
    .expl-row {{ display: flex; justify-content: space-between; margin-top: 4px; font-size: 13px; }}
    .expl-label {{ color: #555; }}
    .expl-val {{ font-weight: bold; font-family: ui-monospace, Menlo, Consolas, monospace; }}
    .expl-val.bad  {{ color: #d32f2f; }}
    .expl-val.good {{ color: #2e7d32; }}

    .section {{ background: white; border-radius: 6px; padding: 20px 24px; margin-bottom: 20px; }}
    .section h2 {{ margin: 0 0 16px 0; font-size: 16px; color: #222; border-bottom: 2px solid #d32f2f; padding-bottom: 6px; }}

    .section-body {{ display: grid; grid-template-columns: 3fr 2fr; gap: 24px; margin-bottom: 16px; align-items: start; }}
    .left-col {{ display: flex; flex-direction: column; gap: 16px; }}
    .left-col img {{ width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; padding: 4px; background: #fff; }}
    .table-label {{ margin: 0 0 8px 0; font-size: 11px; text-transform: uppercase; color: #888; letter-spacing: 0.05em; font-weight: bold; }}
    img {{ width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; padding: 4px; background: #fff; }}

    .bullets h3 {{ margin: 0 0 10px 0; font-size: 12px; text-transform: uppercase; color: #888; letter-spacing: 0.05em; }}
    .bullets ul {{ margin: 0; padding-left: 0; list-style: none; }}
    .bullets li {{ padding: 10px 12px; border-left: 3px solid #d32f2f; margin-bottom: 8px; background: #fafafa; border-radius: 0 4px 4px 0; font-size: 13px; }}
    .bullets li b {{ display: block; font-size: 13px; margin-bottom: 2px; }}
    .bullets .rank-tag {{ font-size: 11px; color: #d32f2f; font-weight: bold; }}
    .bullets .archetype {{ font-size: 12px; color: #444; margin-top: 3px; }}
    .fa-block {{ margin-top: 8px; }}
    .fa-title {{ font-size: 11px; text-transform: uppercase; color: #888; letter-spacing: 0.05em; margin-bottom: 4px; font-weight: bold; }}
    .fa-table {{ border-collapse: collapse; width: 100%; font-size: 11.5px; }}
    .fa-table th {{ background: #f0f0f0; padding: 4px 8px; font-weight: bold; color: #333; border: 1px solid #e0e0e0; }}
    .fa-table td {{ padding: 4px 8px; border: 1px solid #e0e0e0; }}
    .fa-table tr:nth-child(even) {{ background: #fafafa; }}

    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #e0e0e0; padding: 7px 10px; text-align: left; }}
    th {{ background: #f0f0f0; font-weight: bold; color: #333; }}
    tr:nth-child(even) {{ background: #fafafa; }}

    .disclaimer {{ color: #999; font-size: 11px; margin-top: 8px; padding-top: 12px; border-top: 1px solid #e0e0e0; }}
  </style>
</head>
<body>

  <div class="header">
    <h1>{team} EPA Leverage Snapshot — {cfg.season} {cfg.season_type}</h1>
    <p class="sub">Generated {ts} &nbsp;|&nbsp; Public play-by-play data (nflverse) &nbsp;|&nbsp; Skill demonstration project</p>
  </div>

  <div class="intro">
    <h2>About This Report</h2>
    <ul>
      <li><b>What it does:</b> Identifies the game situations where {team} underperformed league average the most, ranked by total EPA gap (per-play gap × play volume). Larger gaps = higher-leverage improvement areas.</li>
      <li><b>How to read it:</b> Each section shows the gap, a general positional archetype that typically addresses it, and relevant free agents at that position.</li>
      <li><b>Data source:</b> nflverse public play-by-play ({cfg.season} regular season). Free agent data from public contract databases.</li>
    </ul>
    <p class="note">Built as a portfolio project to demonstrate data pipeline and analytical thinking. Uses public data only — does not account for scheme, film, medical, or internal scouting context.</p>
  </div>

  <div class="top-strip">
    <div class="exec-summary">
      <div class="bl-title">Key Findings</div>
      {narrative["bl_html"]}
    </div>
    <div class="right-panel">
      <div class="callouts">
        <b>Explosive Play Rates</b>
        <div class="expl-row">
          <span class="expl-label">OFF created</span>
          <span class="expl-val {'bad' if expl['expl_off_rate'] < expl['expl_lg_rate'] else 'good'}">{expl['expl_off_rate']:.1%} vs {expl['expl_lg_rate']:.1%} LG</span>
        </div>
        <div class="expl-row">
          <span class="expl-label">DEF allowed</span>
          <span class="expl-val {'bad' if expl['expl_def_allowed_rate'] > expl['expl_lg_rate'] else 'good'}">{expl['expl_def_allowed_rate']:.1%} vs {expl['expl_lg_rate']:.1%} LG</span>
        </div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Defense: Highest Leverage Gaps</h2>
    <div class="section-body">
      <div class="left-col">
        <img src="{os.path.basename(def_chart)}" alt="DEF chart"/>
        <div>
          <h3 class="table-label">Detailed Breakdown</h3>
          {top_def.to_html(index=False)}
        </div>
      </div>
      <div class="bullets">
        <h3>Priority Areas</h3>
        {bullets_html(narrative["def_bullets"])}
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Offense: Highest Leverage Gaps</h2>
    <div class="section-body">
      <div class="left-col">
        <img src="{os.path.basename(off_chart)}" alt="OFF chart"/>
        <div>
          <h3 class="table-label">Detailed Breakdown</h3>
          {top_off.to_html(index=False)}
        </div>
      </div>
      <div class="bullets">
        <h3>Priority Areas</h3>
        {bullets_html(narrative["off_bullets"])}
      </div>
    </div>
  </div>

  <p class="disclaimer">
    EPA gap = difference between {team} and league-average efficiency, scaled by play volume.
    Rank among teams with ≥{cfg.min_team_plays} plays per bucket.
    Situational buckets overlap (e.g. red zone plays also appear in early-down totals) — gaps are not additive.
    Does not account for opponent quality, scheme, or injury context.
  </p>

</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------
# CLI / Main
# ---------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="EPA leverage snapshot for an NFL team.")
    p.add_argument("--season",           type=int, required=True)
    p.add_argument("--team",             type=str, required=True)
    p.add_argument("--out",              type=str, default="./out")
    p.add_argument("--top-n",            type=int, default=6)
    p.add_argument("--min-team-plays",   type=int, default=30)
    p.add_argument("--min-league-plays", type=int, default=200)
    p.add_argument("--no-html",          action="store_true")
    p.add_argument("--log-level",        type=str, default="INFO")
    return p.parse_args()


def main():
    args   = parse_args()
    logger = setup_logger(args.log_level)
    team   = args.team.strip().upper()

    cfg = Config(
        season=args.season, team=team, out_dir=args.out,
        top_n=args.top_n, min_team_plays=args.min_team_plays,
        min_league_plays=args.min_league_plays, write_html=not args.no_html,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)

    try:
        pbp = load_pbp_data(cfg.season, logger)
        pbp = filter_real_plays(pbp, cfg, logger)
        pbp = compute_flags(pbp, cfg)

        table    = build_leverage_table(pbp, cfg, logger)
        expl     = explosive_rates(pbp, cfg.team)
        ranks_df = compute_league_ranks(pbp, cfg, logger)

        csv_path = os.path.join(cfg.out_dir, f"{team}_{cfg.season}_{cfg.season_type}_leverage.csv")
        table.to_csv(csv_path, index=False)

        def_chart = os.path.join(cfg.out_dir, f"{team}_{cfg.season}_{cfg.season_type}_DEF.png")
        off_chart = os.path.join(cfg.out_dir, f"{team}_{cfg.season}_{cfg.season_type}_OFF.png")

        plot_top_bars(table, "DEF:", cfg,
            title=f"{team} Defense: Improvement Opportunities",
            subtitle=f"{cfg.season} Regular Season — areas below league average",
            out_path=def_chart)
        plot_top_bars(table, "OFF:", cfg,
            title=f"{team} Offense: Improvement Opportunities",
            subtitle=f"{cfg.season} Regular Season — areas below league average",
            out_path=off_chart)

        if cfg.write_html:
            html_path = os.path.join(cfg.out_dir, f"{team}_{cfg.season}_{cfg.season_type}_report.html")
            write_html_report(cfg, table, expl, ranks_df, def_chart, off_chart, html_path)
            logger.info(f"Report: {html_path}")

        logger.info("Done.")
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())