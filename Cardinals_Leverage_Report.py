# cardinals_leverage_report.py
# To run: python Cardinals_Leverage_Report.py --season 2025 --team ARI

"""
Cardinals / NFL Leverage Report (EPA Gap x Volume)
-------------------------------------------------
Builds a "highest leverage improvement opportunities" view by bucket:
    Potential_EPA_Swing = (LeagueAvgEPA - TeamEPA) * TeamPlays   [OFF]
    Potential_EPA_Swing = (TeamAllowedEPA - LeagueAllowedEPA) * TeamPlays [DEF]

Outputs:
- CSV table
- Separate OFF and DEF bar charts (PNG)
- Simple HTML report (optional, enabled by default)

Dependencies:
    pip install pandas numpy matplotlib pyarrow nflreadpy tqdm

Notes:
- nflreadpy often returns a Polars DataFrame; we convert to pandas safely.
- Schema varies across seasons/loader versions; we defensively handle missing columns.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
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


# -------------------------
# Configuration
# -------------------------

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


# -------------------------
# Logging
# -------------------------

def setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("leverage_report")
    logger.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


# -------------------------
# Data Loading
# -------------------------

def load_pbp_data(season: int, logger: logging.Logger) -> pd.DataFrame:
    try:
        from nflreadpy import load_pbp
    except Exception as e:
        raise RuntimeError(
            "Failed to import nflreadpy. Install via: pip install nflreadpy\n"
            f"Original error: {e}"
        )
    try:
        pbp = load_pbp(seasons=[season])
    except Exception as e:
        raise RuntimeError(
            f"Failed to load play-by-play for season={season}. "
            "Check network or dataset availability.\n"
            f"Original error: {e}"
        )
    if hasattr(pbp, "to_pandas"):
        pbp = pbp.to_pandas()
    if not isinstance(pbp, pd.DataFrame):
        raise RuntimeError("Unexpected pbp type after conversion.")
    logger.info(f"Loaded pbp: {pbp.shape[0]:,} rows, {pbp.shape[1]:,} cols")
    return pbp


def validate_required_columns(pbp: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in pbp.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            "\nTry updating nflreadpy or printing pbp.columns to inspect schema."
        )


# -------------------------
# Preprocessing
# -------------------------

def safe_filter_real_plays(pbp: pd.DataFrame, cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    validate_required_columns(pbp, ["season_type", "play_type", "epa"])
    df = pbp[pbp["season_type"] == cfg.season_type].copy()
    df = df[df["play_type"].isin(["run", "pass"])]
    df = df[df["epa"].notna()]
    for col in ["qb_kneel", "qb_spike", "no_play"]:
        if col in df.columns:
            df = df[df[col] != 1]
    logger.info(f"After play filters: {df.shape[0]:,} rows")
    return df


def compute_flags(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    validate_required_columns(df, ["down", "ydstogo", "yardline_100", "yards_gained", "play_type"])
    out = df.copy()
    out["is_third_down"] = out["down"] == 3
    out["is_fourth_down"] = out["down"] == 4
    out["is_early_down"] = out["down"].isin([1, 2])
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


# -------------------------
# Buckets
# -------------------------

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
        if "Early downs" in bucket_name:  return d["is_early_down"]
        if "3rd & short" in bucket_name:  return d["third_short"]
        if "3rd & long"  in bucket_name:  return d["third_long"]
        if "3rd down"    in bucket_name:  return d["is_third_down"]
        if "Red zone"    in bucket_name:  return d["is_red_zone"]
        if "Goal-to-go"  in bucket_name:  return d["is_goal_to_go"]
        return d["play_type"].isin(["pass", "run"])
    if "Early downs" in bucket_name:  return d["is_early_down"]
    if "3rd & short" in bucket_name:  return d["third_short"]
    if "3rd down"    in bucket_name:  return d["is_third_down"]
    if "Red zone"    in bucket_name:  return d["is_red_zone"]
    return d["play_type"].isin(["pass", "run"])


def summarize_bucket(
    d: pd.DataFrame, bucket_name: str, mask_team: pd.Series, cfg: Config
) -> Optional[dict]:
    team_df = d[mask_team]
    lg_df   = d[league_mask(bucket_name, d)]
    if len(team_df) < cfg.min_team_plays or len(lg_df) < cfg.min_league_plays:
        return None
    team_epa = float(team_df["epa"].mean())
    lg_epa   = float(lg_df["epa"].mean())
    n_plays  = int(len(team_df))
    potential_epa = (
        (lg_epa - team_epa) * n_plays if bucket_name.startswith("OFF:")
        else (team_epa - lg_epa) * n_plays
    )
    return {
        "bucket":                   bucket_name,
        "plays":                    n_plays,
        "TEAM_EPA_per_play":        team_epa,
        "LG_EPA_per_play":          lg_epa,
        "TEAM_success_rate":        float(team_df["success"].mean()),
        "LG_success_rate":          float(lg_df["success"].mean()),
        "Potential_EPA_swing_to_LG": float(potential_epa),
    }


def build_leverage_table(d: pd.DataFrame, cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    validate_required_columns(d, ["posteam", "defteam", "epa"])
    buckets = build_bucket_definitions()
    rows: List[dict] = []
    for name, fn in tqdm(buckets.items(), desc="Summarizing buckets"):
        summary = summarize_bucket(d, name, fn(d, cfg.team), cfg)
        if summary:
            rows.append(summary)
    if not rows:
        raise RuntimeError("No bucket summaries produced. Try lowering min sample thresholds.")
    out = pd.DataFrame(rows).sort_values("Potential_EPA_swing_to_LG", ascending=False)
    logger.info(f"Built leverage table with {len(out)} buckets")
    return out


# -------------------------
# League Ranks
# -------------------------

NFL_TEAMS = [
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE",
    "DAL","DEN","DET","GB","HOU","IND","JAX","KC",
    "LA","LAC","LV","MIA","MIN","NE","NO","NYG",
    "NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS",
]


def compute_league_ranks(d: pd.DataFrame, cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    buckets = build_bucket_definitions()
    records = []
    for bucket_name, fn in buckets.items():
        if bucket_name in cfg.chart_exclude_buckets:
            continue
        is_off = bucket_name.startswith("OFF:")
        team_epas = {}
        for t in NFL_TEAMS:
            sub = d[fn(d, t)]
            if len(sub) >= cfg.min_team_plays:
                team_epas[t] = float(sub["epa"].mean())
        if cfg.team not in team_epas or len(team_epas) < 5:
            continue
        sorted_teams = sorted(team_epas.items(), key=lambda x: x[1], reverse=is_off)
        rank_map = {t: i + 1 for i, (t, _) in enumerate(sorted_teams)}
        records.append({
            "bucket":      bucket_name,
            "rank":        rank_map[cfg.team],
            "total_teams": len(team_epas),
        })
    logger.info(f"Computed league ranks for {len(records)} buckets")
    return pd.DataFrame(records)


def rank_suffix(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return {1: f"{n}st", 2: f"{n}nd", 3: f"{n}rd"}.get(n % 10, f"{n}th")


# -------------------------
# Free Agent Data + Scouting Notes
# -------------------------

# FIX #2 & #3: Separate OFF vs DEF mappings, and add per-player scouting descriptions.

FA_CB = pd.DataFrame([
    {"player_name":"Trevon Diggs","position":"CB","age":28.3,"yoe":5,"prev_team":"GB","prev_aav_usd":19400000,"market_value_aav_usd":7536772,
     "scouting_note":"Elite ball-hawk corner with high INT upside; aggressive in press but gives up big plays in off coverage."},
    {"player_name":"Jamel Dean","position":"CB","age":29.2,"yoe":6,"prev_team":"TB","prev_aav_usd":13000000,"market_value_aav_usd":12468672,
     "scouting_note":"Long, physical outside corner who excels in press-man; limits deep shots with 4.3 speed and 6'1 frame."},
    {"player_name":"Jimmie Ward","position":"CB","age":34.5,"yoe":11,"prev_team":"HOU","prev_aav_usd":10508000,"market_value_aav_usd":None,
     "scouting_note":"Versatile safety/slot hybrid with leadership value; injury history is the primary concern at this stage."},
    {"player_name":"Jonathan Jones","position":"CB","age":32.3,"yoe":9,"prev_team":"WAS","prev_aav_usd":5500000,"market_value_aav_usd":5071032,
     "scouting_note":"Reliable slot corner with plus tackling; strong on 3rd-down blitz packages and short-area quickness."},
    {"player_name":"Amik Robertson","position":"CB","age":27.5,"yoe":5,"prev_team":"DET","prev_aav_usd":4625000,"market_value_aav_usd":3940390,
     "scouting_note":"Undersized but tenacious nickel corner; excels in zone coverage and run support from the slot."},
    {"player_name":"Eric Stokes","position":"CB","age":26.8,"yoe":4,"prev_team":"LV","prev_aav_usd":3500000,"market_value_aav_usd":7368096,
     "scouting_note":"Former 1st-rounder with elite speed (4.25); career derailed by injuries but upside play if healthy."},
    {"player_name":"Kader Kohou","position":"CB","age":27.1,"yoe":3,"prev_team":"MIA","prev_aav_usd":3263000,"market_value_aav_usd":2047307,
     "scouting_note":"UDFA success story; physical in run support and competitive in contested catches despite size limitations."},
    {"player_name":"Greg Newsome","position":"CB","age":25.7,"yoe":4,"prev_team":"JAX","prev_aav_usd":3187184,"market_value_aav_usd":9020706,
     "scouting_note":"Smooth outside corner with zone instincts; young enough to project improvement and fits multiple schemes."},
    {"player_name":"Ifeatu Melifonwu","position":"CB","age":26.7,"yoe":4,"prev_team":"MIA","prev_aav_usd":3010000,"market_value_aav_usd":3660364,
     "scouting_note":"Tall, long corner (6'3) who can match up with big receivers; still developing route recognition."},
    {"player_name":"TreDavious White","position":"CB","age":31.0,"yoe":8,"prev_team":"BUF","prev_aav_usd":3000000,"market_value_aav_usd":1203422,
     "scouting_note":"Former All-Pro whose game has declined post-ACL; could provide veteran depth at a low cost."},
    {"player_name":"Benjamin St-Juste","position":"CB","age":28.3,"yoe":4,"prev_team":"LAC","prev_aav_usd":2500000,"market_value_aav_usd":None,
     "scouting_note":"Big-bodied outside corner with length to disrupt at the catch point; inconsistent but high-ceiling snaps."},
    {"player_name":"Jeff Okudah","position":"CB","age":26.9,"yoe":5,"prev_team":"MIN","prev_aav_usd":2350000,"market_value_aav_usd":None,
     "scouting_note":"Former #3 overall pick rebuilding value; flashed in Detroit before injuries — buy-low candidate."},
    {"player_name":"Roger McCreary","position":"CB","age":25.9,"yoe":3,"prev_team":"LAR","prev_aav_usd":2291402,"market_value_aav_usd":12155481,
     "scouting_note":"Sticky man-coverage corner with plus ball skills; market value far exceeds prior deal — high-value target."},
    {"player_name":"Jason Pinnock","position":"CB","age":26.5,"yoe":4,"prev_team":"SF","prev_aav_usd":2200000,"market_value_aav_usd":None,
     "scouting_note":"Converted safety with range and physicality; still raw in coverage technique but has positional versatility."},
    {"player_name":"Josh Jobe","position":"CB","age":27.8,"yoe":3,"prev_team":"SEA","prev_aav_usd":2000000,"market_value_aav_usd":None,
     "scouting_note":"Physical press corner who struggles with speed in space; best fit as a CB3/4 in a press-heavy scheme."},
    {"player_name":"Nazeeh Johnson","position":"CB","age":27.5,"yoe":3,"prev_team":"KC","prev_aav_usd":1900000,"market_value_aav_usd":None,
     "scouting_note":"Special teams contributor who earned defensive reps in KC; developmental depth piece at corner/safety."},
    {"player_name":"Alontae Taylor","position":"CB","age":27.1,"yoe":3,"prev_team":"NO","prev_aav_usd":1801175,"market_value_aav_usd":11166006,
     "scouting_note":"Breakout corner with elite value ratio; thrived in man coverage in NO and projects as a CB1 on the open market."},
    {"player_name":"Adoree Jackson","position":"CB","age":30.3,"yoe":8,"prev_team":"PHI","prev_aav_usd":1755000,"market_value_aav_usd":None,
     "scouting_note":"Veteran with return ability and scheme versatility; speed has declined but route recognition compensates."},
    {"player_name":"Kris Boyd","position":"CB","age":29.3,"yoe":6,"prev_team":"NYJ","prev_aav_usd":1600000,"market_value_aav_usd":None,
     "scouting_note":"Core special teamer with limited defensive ceiling; roster depth and ST value only."},
    {"player_name":"Rasul Douglas","position":"CB","age":31.3,"yoe":9,"prev_team":"MIA","prev_aav_usd":1572500,"market_value_aav_usd":4018154,
     "scouting_note":"Big corner (6'2) who jumps routes and creates turnovers; gives up speed but wins at the catch point."},
    {"player_name":"Cam Lewis","position":"CB","age":28.8,"yoe":5,"prev_team":"BUF","prev_aav_usd":1550000,"market_value_aav_usd":1506691,
     "scouting_note":"Solid nickel corner in zone-heavy schemes; limited upside but dependable in a role."},
    {"player_name":"Marco Wilson","position":"CB","age":26.8,"yoe":4,"prev_team":"CIN","prev_aav_usd":1520000,"market_value_aav_usd":None,
     "scouting_note":"Former Cardinals draft pick with familiarity in ARI's scheme; inconsistent but young and cheap."},
    {"player_name":"Dee Alford","position":"CB","age":28.2,"yoe":3,"prev_team":"ATL","prev_aav_usd":1500000,"market_value_aav_usd":5896757,
     "scouting_note":"Slot specialist with strong instincts in zone coverage; limited outside but excellent value in the nickel."},
    {"player_name":"Noah Igbinoghene","position":"CB","age":26.1,"yoe":5,"prev_team":"WAS","prev_aav_usd":1500000,"market_value_aav_usd":None,
     "scouting_note":"Former 1st-rounder who hasn't developed; elite speed but poor technique — a project at this point."},
])

FA_LB = pd.DataFrame([
    {"player_name":"Bobby Wagner","position":"LB","age":35.5,"yoe":13,"prev_team":"WAS","prev_aav_usd":9000000,"market_value_aav_usd":7684822,
     "scouting_note":"Future Hall of Famer still producing at a high level; elite run-diagnosis and leadership presence."},
    {"player_name":"Lavonte David","position":"LB","age":36.0,"yoe":13,"prev_team":"TB","prev_aav_usd":9000000,"market_value_aav_usd":7430840,
     "scouting_note":"Instinctive three-down LB with pass coverage range; age is the only concern — production remains elite."},
    {"player_name":"Demario Davis","position":"LB","age":37.0,"yoe":13,"prev_team":"NO","prev_aav_usd":8625000,"market_value_aav_usd":9478723,
     "scouting_note":"Iron-man LB who hasn't missed games; dominant run defender and vocal defensive signal-caller."},
    {"player_name":"Kenneth Murray","position":"LB","age":27.2,"yoe":5,"prev_team":"DAL","prev_aav_usd":7750000,"market_value_aav_usd":4922042,
     "scouting_note":"Athletic LB with sideline-to-sideline speed; over-pursues at times but has three-down upside."},
    {"player_name":"Kaden Elliss","position":"LB","age":30.5,"yoe":6,"prev_team":"ATL","prev_aav_usd":7166667,"market_value_aav_usd":8989843,
     "scouting_note":"Versatile LB who can rush the passer and drop into coverage; ascending player with scheme flexibility."},
    {"player_name":"Matt Milano","position":"LB","age":30.8,"yoe":8,"prev_team":"BUF","prev_aav_usd":6306500,"market_value_aav_usd":4598821,
     "scouting_note":"Elite coverage LB when healthy; missed significant time with injuries but transforms a defense when available."},
    {"player_name":"Alex Anzalone","position":"LB","age":31.3,"yoe":8,"prev_team":"DET","prev_aav_usd":6250000,"market_value_aav_usd":7263314,
     "scouting_note":"Reliable starter who anchored Detroit's defense; solid in run fits and adequate in zone coverage."},
    {"player_name":"Von Miller","position":"LB","age":36.8,"yoe":14,"prev_team":"WAS","prev_aav_usd":6100000,"market_value_aav_usd":5842864,
     "scouting_note":"Future HOF pass rusher with diminished burst; still wins with technique and provides locker room gravity."},
    {"player_name":"Cole Holcomb","position":"LB","age":29.4,"yoe":5,"prev_team":"PIT","prev_aav_usd":6000000,"market_value_aav_usd":2674013,
     "scouting_note":"Run-stuffing MIKE coming off a major knee injury; if medicals check out, significant value at his price."},
    {"player_name":"Quincy Williams","position":"LB","age":29.3,"yoe":6,"prev_team":"NYJ","prev_aav_usd":6000000,"market_value_aav_usd":9196852,
     "scouting_note":"Explosive, high-motor LB with blitzing ability; can be liability in coverage but impacts early downs."},
    {"player_name":"Alex Singleton","position":"LB","age":32.1,"yoe":6,"prev_team":"DEN","prev_aav_usd":6000000,"market_value_aav_usd":4663788,
     "scouting_note":"High-tackle-volume LB who fills gaps quickly; limited athlete but consistently productive run defender."},
    {"player_name":"E.J. Speed","position":"LB","age":30.6,"yoe":6,"prev_team":"HOU","prev_aav_usd":3500000,"market_value_aav_usd":4761657,
     "scouting_note":"Rangy, athletic LB with special teams value; developing as a starter but excellent in space."},
    {"player_name":"Quay Walker","position":"LB","age":25.7,"yoe":3,"prev_team":"GB","prev_aav_usd":3460410,"market_value_aav_usd":9674871,
     "scouting_note":"Young, athletic former 1st-rounder with three-down potential; market value nearly 3x his prior deal — elite value target."},
    {"player_name":"Jadeveon Clowney","position":"LB","age":32.9,"yoe":11,"prev_team":"DAL","prev_aav_usd":3450000,"market_value_aav_usd":5711599,
     "scouting_note":"Disruptive edge presence who sets the edge vs. the run; inconsistent effort but game-wrecking snaps when engaged."},
    {"player_name":"Devin Bush","position":"LB","age":27.5,"yoe":6,"prev_team":"CLE","prev_aav_usd":3250000,"market_value_aav_usd":8899904,
     "scouting_note":"Former top-10 pick with speed and instincts; career revived in Cleveland — 2.7x value ratio makes him a buy-low LB."},
    {"player_name":"Devin Lloyd","position":"LB","age":27.2,"yoe":3,"prev_team":"JAX","prev_aav_usd":3234151,"market_value_aav_usd":20144133,
     "scouting_note":"Elite value ratio (6.2x) — versatile LB who can play all three downs, rush the passer, and cover TEs in man."},
    {"player_name":"Elandon Roberts","position":"LB","age":31.8,"yoe":9,"prev_team":"LV","prev_aav_usd":3010000,"market_value_aav_usd":4213132,
     "scouting_note":"Thumper run-stuffer who excels on early downs; limited in coverage but dominant in A-gap run defense."},
    {"player_name":"Justin Strnad","position":"LB","age":29.4,"yoe":5,"prev_team":"DEN","prev_aav_usd":2787500,"market_value_aav_usd":3819398,
     "scouting_note":"Rotational LB with special teams upside; adequate in zone coverage but lacks playmaking ability."},
    {"player_name":"Denzel Perryman","position":"LB","age":33.1,"yoe":10,"prev_team":"LAC","prev_aav_usd":2655000,"market_value_aav_usd":None,
     "scouting_note":"Hard-hitting inside LB who destroys run plays; durability and coverage are long-standing concerns."},
    {"player_name":"Eric Wilson","position":"LB","age":31.2,"yoe":8,"prev_team":"MIN","prev_aav_usd":2600000,"market_value_aav_usd":4326794,
     "scouting_note":"Coverage-capable LB who reads route concepts well; provides versatility but isn't a difference-maker vs. the run."},
    {"player_name":"Jake Martin","position":"LB","age":30.1,"yoe":7,"prev_team":"WAS","prev_aav_usd":2585000,"market_value_aav_usd":2776109,
     "scouting_note":"Pass-rush specialist off the edge; one-dimensional but generates pressure on passing downs."},
    {"player_name":"Christian Rozeboom","position":"LB","age":28.9,"yoe":5,"prev_team":"CAR","prev_aav_usd":2500000,"market_value_aav_usd":3046794,
     "scouting_note":"Special teams ace who earned defensive snaps; solid effort player who fits as a LB4/5."},
    {"player_name":"Jack Cochrane","position":"LB","age":26.9,"yoe":3,"prev_team":"KC","prev_aav_usd":2100000,"market_value_aav_usd":None,
     "scouting_note":"Young developmental LB who showed flashes in KC; zone instincts are ahead of his run-defense technique."},
    {"player_name":"Dennis Gardeck","position":"LB","age":31.4,"yoe":7,"prev_team":"JAX","prev_aav_usd":2000000,"market_value_aav_usd":None,
     "scouting_note":"Former Cardinal with edge-rush ability and ST dominance; fan-favorite reunion candidate at low cost."},
])

FA_WR = pd.DataFrame([
    {"player_name":"Tyreek Hill","position":"WR","age":32,"prev_team":"MIA","prev_aav_usd":30000000,"market_value_aav_usd":None,"contract_type":"SFA",
     "scouting_note":"Generational speed (4.29) who stretches defenses vertically; aging but still commands double teams and creates chunk plays."},
    {"player_name":"Deebo Samuel","position":"WR","age":30,"prev_team":"WAS","prev_aav_usd":23850000,"market_value_aav_usd":None,"contract_type":"UFA",
     "scouting_note":"YAC monster who can line up everywhere; run-after-catch ability directly addresses 3rd-down conversion needs."},
    {"player_name":"Mike Evans","position":"WR","age":33,"prev_team":"TB","prev_aav_usd":20500000,"market_value_aav_usd":None,"contract_type":"Void",
     "scouting_note":"Prototypical X receiver who wins contested catches in the red zone; 6'5 frame dominates in condensed space."},
    {"player_name":"Christian Kirk","position":"WR","age":30,"prev_team":"HOU","prev_aav_usd":18000000,"market_value_aav_usd":None,"contract_type":"UFA",
     "scouting_note":"Former Cardinal with reliable route-running; excels on intermediate crossers that convert 3rd-and-longs."},
    {"player_name":"Tutu Atwell","position":"WR","age":27,"prev_team":"LAR","prev_aav_usd":10000000,"market_value_aav_usd":None,"contract_type":"UFA",
     "scouting_note":"Blazing speed creates explosive play potential; undersized but can be schemed open on jet sweeps and go routes."},
    {"player_name":"Dyami Brown","position":"WR","age":27,"prev_team":"JAX","prev_aav_usd":10000000,"market_value_aav_usd":None,"contract_type":"Void",
     "scouting_note":"Deep-threat specialist who never found consistent role; high ceiling but hasn't put it together at the NFL level."},
    {"player_name":"Marquise Brown","position":"WR","age":29,"prev_team":"KC","prev_aav_usd":7000000,"market_value_aav_usd":None,"contract_type":"UFA",
     "scouting_note":"Speed receiver who can take the top off; injury concerns but creates explosive plays when healthy."},
    {"player_name":"Jauan Jennings","position":"WR","age":29,"prev_team":"SF","prev_aav_usd":5945000,"market_value_aav_usd":None,"contract_type":"UFA",
     "scouting_note":"Physical possession receiver who wins in contested situations; ideal red-zone and 3rd-down target with strong hands."},
    {"player_name":"DeAndre Hopkins","position":"WR","age":34,"prev_team":"BAL","prev_aav_usd":5000000,"market_value_aav_usd":None,"contract_type":"Void",
     "scouting_note":"Former Cardinal with legendary hands and route craft; speed has declined but still wins on timing and contested catches."},
    {"player_name":"Adam Thielen","position":"WR","age":36,"prev_team":"PIT","prev_aav_usd":5000000,"market_value_aav_usd":None,"contract_type":"UFA",
     "scouting_note":"Route-running technician who creates separation without elite speed; red-zone and 3rd-down conversion specialist."},
    {"player_name":"Kalif Raymond","position":"WR","age":32,"prev_team":"DET","prev_aav_usd":4000000,"market_value_aav_usd":None,"contract_type":"UFA",
     "scouting_note":"Burner with return ability; can create explosive plays from the slot but inconsistent hands limit target share."},
    {"player_name":"Jahan Dotson","position":"WR","age":26,"prev_team":"PHI","prev_aav_usd":3762089,"market_value_aav_usd":15048356,"contract_type":"Void",
     "scouting_note":"Young former 1st-rounder with route polish; best value ratio on the WR market (4.0x) — high-upside buy-low."},
    {"player_name":"Greg Dortch","position":"WR","age":28,"prev_team":"ARI","prev_aav_usd":3263000,"market_value_aav_usd":None,"contract_type":"UFA",
     "scouting_note":"Former Cardinal slot weapon with YAC ability; scheme-familiar and could return at a low price point."},
    {"player_name":"Noah Brown","position":"WR","age":30,"prev_team":"WAS","prev_aav_usd":3250000,"market_value_aav_usd":2470000,"contract_type":"UFA",
     "scouting_note":"Big-bodied role player who blocks well and catches contested balls; depth piece for red-zone packages."},
    {"player_name":"Rashid Shaheed","position":"WR","age":28,"prev_team":"SEA","prev_aav_usd":3092500,"market_value_aav_usd":None,"contract_type":"Void",
     "scouting_note":"Home-run speed on deep routes and jet sweeps; one-dimensional but directly addresses explosive play creation."},
    {"player_name":"Keenan Allen","position":"WR","age":34,"prev_team":"LAC","prev_aav_usd":3020000,"market_value_aav_usd":2255000,"contract_type":"UFA",
     "scouting_note":"Route-running savant who gets open on 3rd down; aging but still one of the best at finding soft spots in zone."},
    {"player_name":"Nick Westbrook-Ikhine","position":"WR","age":29,"prev_team":"MIA","prev_aav_usd":2995000,"market_value_aav_usd":3200000,"contract_type":"SFA",
     "scouting_note":"Reliable depth receiver with blocking ability; won't move the needle but provides roster stability."},
    {"player_name":"Josh Reynolds","position":"WR","age":31,"prev_team":"NYJ","prev_aav_usd":2750000,"market_value_aav_usd":2000000,"contract_type":"UFA",
     "scouting_note":"Solid WR3 with size and route diversity; won't create separation but wins at the catch point on back-shoulders."},
    {"player_name":"Tim Patrick","position":"WR","age":33,"prev_team":"JAX","prev_aav_usd":2500000,"market_value_aav_usd":2500000,"contract_type":"UFA",
     "scouting_note":"Contested-catch specialist who plays bigger than his frame; career interrupted by ACL but was a productive WR2."},
    {"player_name":"Zay Jones","position":"WR","age":31,"prev_team":"ARI","prev_aav_usd":2400000,"market_value_aav_usd":1300000,"contract_type":"UFA",
     "scouting_note":"Former Cardinal depth piece; steady hands but limited after the catch — replacement-level at this point."},
    {"player_name":"Tylan Wallace","position":"WR","age":27,"prev_team":"BAL","prev_aav_usd":2100000,"market_value_aav_usd":1350000,"contract_type":"Void",
     "scouting_note":"Physical receiver with punt-return ability; hasn't carved out an offensive role but provides ST value."},
    {"player_name":"Wan'Dale Robinson","position":"WR","age":25,"prev_team":"NYG","prev_aav_usd":2046292,"market_value_aav_usd":5791409,"contract_type":"UFA",
     "scouting_note":"Shifty slot receiver with elite YAC in short areas; 2.8x value ratio and youth make him a strong fit for 3rd-down packages."},
    {"player_name":"Rondale Moore","position":"WR","age":26,"prev_team":"MIN","prev_aav_usd":2000000,"market_value_aav_usd":250000,"contract_type":"UFA",
     "scouting_note":"Former Cardinal with electric speed; never stayed healthy enough to develop — minimal value at this stage."},
    {"player_name":"Hunter Renfrow","position":"WR","age":31,"prev_team":"CAR","prev_aav_usd":2000000,"market_value_aav_usd":830000,"contract_type":"SFA",
     "scouting_note":"Precise slot receiver who converts 3rd downs; speed limitations but excellent hands and route discipline."},
    {"player_name":"Braxton Berrios","position":"WR","age":31,"prev_team":"HOU","prev_aav_usd":1800000,"market_value_aav_usd":300000,"contract_type":"UFA",
     "scouting_note":"Return specialist with limited offensive role; depth piece only at this price."},
    {"player_name":"Kendrick Bourne","position":"WR","age":31,"prev_team":"SF","prev_aav_usd":1765000,"market_value_aav_usd":1255000,"contract_type":"UFA",
     "scouting_note":"YAC-friendly receiver who thrives in Shanahan-style schemes; brings energy and can stretch intermediate zones."},
    {"player_name":"Van Jefferson","position":"WR","age":30,"prev_team":"TEN","prev_aav_usd":1670000,"market_value_aav_usd":1170000,"contract_type":"UFA",
     "scouting_note":"Route-runner with deep speed; never became a consistent starter but can threaten vertically as a WR3/4."},
    {"player_name":"Alec Pierce","position":"WR","age":26,"prev_team":"IND","prev_aav_usd":1650336,"market_value_aav_usd":3691037,"contract_type":"UFA",
     "scouting_note":"Big, fast deep threat (6'3, 4.41) who creates explosive plays; drops and inconsistency limit his floor."},
    {"player_name":"Skyy Moore","position":"WR","age":26,"prev_team":"SF","prev_aav_usd":1612627,"market_value_aav_usd":3574481,"contract_type":"UFA",
     "scouting_note":"Young slot receiver with untapped potential; 2.2x value ratio and age make him a worthwhile developmental add."},
    {"player_name":"Olamide Zaccheaus","position":"WR","age":29,"prev_team":"CHI","prev_aav_usd":1500000,"market_value_aav_usd":750000,"contract_type":"UFA",
     "scouting_note":"Undersized slot with return ability; limited target share upside but fills a roster spot cheaply."},
    {"player_name":"Sterling Shepard","position":"WR","age":33,"prev_team":"TB","prev_aav_usd":1500000,"market_value_aav_usd":500000,"contract_type":"UFA",
     "scouting_note":"Veteran slot with reliable hands; body is breaking down but can mentor younger receivers."},
    {"player_name":"JuJu Smith-Schuster","position":"WR","age":30,"prev_team":"KC","prev_aav_usd":1422500,"market_value_aav_usd":1197500,"contract_type":"UFA",
     "scouting_note":"Physical receiver who plays well in traffic; won a Super Bowl in KC — leadership value exceeds current production."},
    {"player_name":"DeAndre Carter","position":"WR","age":33,"prev_team":"CLE","prev_aav_usd":1422500,"market_value_aav_usd":767500,"contract_type":"UFA",
     "scouting_note":"Return specialist and emergency receiver; minimal offensive impact but provides ST roster flexibility."},
])

FA_RB = pd.DataFrame([
    {"player_name":"Najee Harris","position":"RB","age":28,"prev_team":"LAC","prev_aav_usd":5250000,"market_value_aav_usd":5250000,"contract_type":"UFA",
     "scouting_note":"Workhorse back with excellent pass protection; between-the-tackles runner who sustains drives on early downs."},
    {"player_name":"Austin Ekeler","position":"RB","age":31,"prev_team":"WAS","prev_aav_usd":4215000,"market_value_aav_usd":4210000,"contract_type":"UFA",
     "scouting_note":"Elite receiving back who creates mismatches out of the backfield; directly addresses 3rd-down and red-zone needs."},
    {"player_name":"Antonio Gibson","position":"RB","age":28,"prev_team":"NE","prev_aav_usd":3750000,"market_value_aav_usd":5300000,"contract_type":"SFA",
     "scouting_note":"Versatile back who can line up at WR; explosive in space and contributes to explosive play creation."},
    {"player_name":"Travis Etienne","position":"RB","age":27,"prev_team":"JAX","prev_aav_usd":3224528,"market_value_aav_usd":12898112,"contract_type":"UFA",
     "scouting_note":"Home-run back with 4.4 speed — 4.0x value ratio; creates explosive runs and can catch out of the backfield."},
    {"player_name":"Rico Dowdle","position":"RB","age":28,"prev_team":"CAR","prev_aav_usd":2750000,"market_value_aav_usd":2750000,"contract_type":"UFA",
     "scouting_note":"Physical between-the-tackles runner who improved as a receiver; solid early-down contributor."},
    {"player_name":"Nick Chubb","position":"RB","age":31,"prev_team":"HOU","prev_aav_usd":2500000,"market_value_aav_usd":1500000,"contract_type":"UFA",
     "scouting_note":"Former elite runner returning from devastating knee injury; if healthy, still one of the most physical backs in football."},
    {"player_name":"Breece Hall","position":"RB","age":25,"prev_team":"NYJ","prev_aav_usd":2253692,"market_value_aav_usd":7080482,"contract_type":"UFA",
     "scouting_note":"Dynamic three-down back with elite receiving ability; 3.1x value ratio and youth make him the top RB value target."},
    {"player_name":"Kenneth Walker III","position":"RB","age":26,"prev_team":"SEA","prev_aav_usd":2110395,"market_value_aav_usd":6144040,"contract_type":"UFA",
     "scouting_note":"Explosive runner with home-run speed; creates chunk plays between the tackles and in space."},
    {"player_name":"J.K. Dobbins","position":"RB","age":28,"prev_team":"DEN","prev_aav_usd":2065000,"market_value_aav_usd":2065000,"contract_type":"UFA",
     "scouting_note":"Powerful runner with burst through the hole; injury history is concerning but flashes elite vision when healthy."},
    {"player_name":"Dare Ogunbowale","position":"RB","age":32,"prev_team":"HOU","prev_aav_usd":1800000,"market_value_aav_usd":750000,"contract_type":"UFA",
     "scouting_note":"Pass-catching specialist and solid pass protector; 3rd-down role player who won't move the needle as a runner."},
    {"player_name":"Kenneth Gainwell","position":"RB","age":27,"prev_team":"PIT","prev_aav_usd":1790000,"market_value_aav_usd":620000,"contract_type":"UFA",
     "scouting_note":"Change-of-pace back with receiving chops; limited as an early-down runner but provides 3rd-down versatility."},
    {"player_name":"Travis Homer","position":"RB","age":28,"prev_team":"CHI","prev_aav_usd":1750000,"market_value_aav_usd":1000000,"contract_type":"UFA",
     "scouting_note":"Special teams contributor who can catch passes; minimal rushing impact but roster flexibility."},
    {"player_name":"Jerome Ford","position":"RB","age":27,"prev_team":"CLE","prev_aav_usd":1750000,"market_value_aav_usd":1750000,"contract_type":"UFA",
     "scouting_note":"Explosive change-of-pace back with big-play ability; inconsistent workload but creates chunk runs."},
    {"player_name":"Raheem Mostert","position":"RB","age":34,"prev_team":"LV","prev_aav_usd":1600000,"market_value_aav_usd":175000,"contract_type":"UFA",
     "scouting_note":"Home-run speed back who is at the end of the line; health and age make him a low-priority depth add."},
    {"player_name":"Kene Nwangwu","position":"RB","age":28,"prev_team":"NYJ","prev_aav_usd":1500000,"market_value_aav_usd":550000,"contract_type":"UFA",
     "scouting_note":"Kick return specialist with breakaway speed; minimal offensive role but electric on special teams."},
    {"player_name":"Kareem Hunt","position":"RB","age":31,"prev_team":"KC","prev_aav_usd":1500000,"market_value_aav_usd":850000,"contract_type":"UFA",
     "scouting_note":"Veteran goal-line back with red-zone scoring history; still effective in short-yardage and as a receiver."},
    {"player_name":"A.J. Dillon","position":"RB","age":28,"prev_team":"PHI","prev_aav_usd":1337500,"market_value_aav_usd":167500,"contract_type":"UFA",
     "scouting_note":"Massive back (247 lbs) built for goal-line and short-yardage; limited in space but punishing between the tackles."},
    {"player_name":"Jeremy McNichols","position":"RB","age":31,"prev_team":"WAS","prev_aav_usd":1337500,"market_value_aav_usd":492500,"contract_type":"UFA",
     "scouting_note":"Journeyman receiving back; provides roster depth but minimal impact as a runner."},
    {"player_name":"Alexander Mattison","position":"RB","age":28,"prev_team":"MIA","prev_aav_usd":1337500,"market_value_aav_usd":1197500,"contract_type":"UFA",
     "scouting_note":"Solid backup who ran well behind Minnesota's line; dependable early-down option without explosive upside."},
    {"player_name":"Miles Sanders","position":"RB","age":29,"prev_team":"DAL","prev_aav_usd":1337500,"market_value_aav_usd":1197500,"contract_type":"UFA",
     "scouting_note":"Former 1,000-yard rusher whose production has fallen off; can still contribute as a rotational early-down back."},
    {"player_name":"Rachaad White","position":"RB","age":27,"prev_team":"TB","prev_aav_usd":1282500,"market_value_aav_usd":910908,"contract_type":"UFA",
     "scouting_note":"Receiving back with size (210 lbs); pass protection is his calling card, making him a natural 3rd-down option."},
    {"player_name":"Brian Robinson Jr.","position":"RB","age":27,"prev_team":"SF","prev_aav_usd":1261226,"market_value_aav_usd":849020,"contract_type":"UFA",
     "scouting_note":"Physical, downhill runner who moves the pile; limited receiving ability but effective in goal-to-go situations."},
    {"player_name":"Michael Carter","position":"RB","age":27,"prev_team":"ARI","prev_aav_usd":1170000,"market_value_aav_usd":None,"contract_type":"UFA",
     "scouting_note":"Former Cardinal with scheme familiarity; versatile but hasn't been a featured back — depth and ST value."},
    {"player_name":"Zamir White","position":"RB","age":27,"prev_team":"LV","prev_aav_usd":1100981,"market_value_aav_usd":743924,"contract_type":"UFA",
     "scouting_note":"Power runner with vision; struggled as a starter but could thrive in a reduced role with a better line."},
    {"player_name":"Dameon Pierce","position":"RB","age":26,"prev_team":"KC","prev_aav_usd":1100000,"market_value_aav_usd":None,"contract_type":"UFA",
     "scouting_note":"Young power back who had a strong rookie year then regressed; buy-low candidate with upside if given a role."},
])

FA_DATA: Dict[str, pd.DataFrame] = {"CB": FA_CB, "LB": FA_LB, "WR": FA_WR, "RB": FA_RB}

# FIX #2: Separate OFF and DEF bucket-to-position mappings.
# Keys are checked with substring matching, so order matters — more specific keys first.
BUCKET_FA_MAP_DEF: Dict[str, List[str]] = {
    "3rd and short": ["CB"],
    "3rd":           ["CB"],
    "early down":    ["LB"],
    "red zone":      ["LB", "CB"],
    "explosive":     ["CB"],
}

BUCKET_FA_MAP_OFF: Dict[str, List[str]] = {
    "3rd and long":  ["WR"],
    "3rd and short": ["WR"],
    "3rd":           ["WR"],
    "early down":    ["WR", "RB"],
    "red zone":      ["WR", "RB"],
    "goal":          ["WR", "RB"],
    "explosive":     ["WR", "RB"],
}


def get_fa_targets(lbl: str, side: str, top_n: int = 3) -> pd.DataFrame:
    """
    Given a bucket label and side (OFF/DEF), return top FA targets ranked by
    best value (market_value_aav_usd / prev_aav_usd — highest ratio = most underpriced).
    Falls back to players with no market value listed last.
    """
    fa_map = BUCKET_FA_MAP_OFF if side == "OFF" else BUCKET_FA_MAP_DEF
    positions = []
    for key, pos_list in fa_map.items():
        if key in lbl.lower():
            positions = pos_list
            break
    if not positions:
        return pd.DataFrame()

    frames = [FA_DATA[p] for p in positions if p in FA_DATA]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True).copy()
    df["value_ratio"] = df["market_value_aav_usd"] / df["prev_aav_usd"]
    known   = df[df["market_value_aav_usd"].notna()].sort_values("value_ratio", ascending=False)
    unknown = df[df["market_value_aav_usd"].isna()]
    return pd.concat([known, unknown], ignore_index=True).head(top_n)


def fa_html(lbl: str, side: str, top_n: int = 3) -> str:
    """Generate HTML table for FA targets, now including scouting notes."""
    df = get_fa_targets(lbl, side, top_n)
    if df.empty:
        return ""
    rows_html = ""
    for _, r in df.iterrows():
        prev  = f"${r['prev_aav_usd']/1e6:.1f}M"
        mkt   = f"${r['market_value_aav_usd']/1e6:.1f}M" if pd.notna(r.get("market_value_aav_usd")) else "N/A"
        ratio = f"{r['value_ratio']:.1f}x" if pd.notna(r.get("value_ratio")) else "—"
        note  = r.get("scouting_note", "")
        rows_html += (
            f"<tr>"
            f"<td><b>{r['player_name']}</b></td>"
            f"<td>{r['position']} | Age {r['age']}</td>"
            f"<td>{r['prev_team']}</td>"
            f"<td>{prev}</td>"
            f"<td>{mkt}</td>"
            f"<td style='color:#2e7d32;font-weight:bold'>{ratio}</td>"
            f"<td class='scout-note'>{note}</td>"
            f"</tr>"
        )
    return f"""
<div class="fa-block">
  <div class="fa-title">FA Targets</div>
  <table class="fa-table">
    <thead><tr>
      <th>Player</th><th>Pos / Age</th><th>Prev Team</th>
      <th>Prev AAV</th><th>Market Value</th><th>Value Ratio</th><th>Why This Player</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""


# -------------------------
# Explosives
# -------------------------

def explosive_rates(d: pd.DataFrame, team: str) -> dict:
    off  = d[d["posteam"].eq(team)]
    deff = d[d["defteam"].eq(team)]
    return {
        "expl_off_rate":         float(off["is_explosive"].mean())  if len(off)  else float("nan"),
        "expl_def_allowed_rate": float(deff["is_explosive"].mean()) if len(deff) else float("nan"),
        "expl_lg_rate":          float(d["is_explosive"].mean())    if len(d)    else float("nan"),
    }


# -------------------------
# Narrative Generation
# -------------------------

def bucket_label(b: str) -> str:
    if not b:
        return "overall"
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


def generate_narrative(
    cfg: Config,
    leverage_table: pd.DataFrame,
    ranks_df: pd.DataFrame,
    expl: dict,
) -> dict:
    team    = cfg.team
    exclude = set(cfg.chart_exclude_buckets)

    def_rows = leverage_table[
        leverage_table["bucket"].str.startswith("DEF:") &
        ~leverage_table["bucket"].isin(exclude) &
        (leverage_table["Potential_EPA_swing_to_LG"] > 0)
    ].sort_values("Potential_EPA_swing_to_LG", ascending=False)

    off_rows = leverage_table[
        leverage_table["bucket"].str.startswith("OFF:") &
        ~leverage_table["bucket"].isin(exclude) &
        (leverage_table["Potential_EPA_swing_to_LG"] > 0)
    ].sort_values("Potential_EPA_swing_to_LG", ascending=False)

    def get_rank_str(bucket: str) -> str:
        if ranks_df.empty:
            return ""
        row = ranks_df[ranks_df["bucket"] == bucket]
        if row.empty:
            return ""
        r, tot = int(row.iloc[0]["rank"]), int(row.iloc[0]["total_teams"])
        return f" ({rank_suffix(r)} of {tot} teams)"

    top_def_bucket = def_rows.iloc[0]["bucket"] if not def_rows.empty else None
    top_off_bucket = off_rows.iloc[0]["bucket"] if not off_rows.empty else None

    total_def_swing = def_rows["Potential_EPA_swing_to_LG"].sum() if not def_rows.empty else 0
    total_off_swing = off_rows["Potential_EPA_swing_to_LG"].sum() if not off_rows.empty else 0
    wins_def = total_def_swing / 140
    wins_off = total_off_swing / 140

    # Build bottom-line bullets (scannable, GM-friendly)
    bl_bullets = []

    if top_def_bucket:
        r = def_rows.iloc[0]
        bl_bullets.append(
            f"<b>Biggest gap is on defense</b> — {bucket_label(top_def_bucket)} "
            f"is the #1 problem area{get_rank_str(top_def_bucket)}. "
            f"Fixing defense to league average is worth ~{wins_def:.1f} wins."
        )

    if top_off_bucket:
        r = off_rows.iloc[0]
        bl_bullets.append(
            f"<b>Offense needs work on {bucket_label(top_off_bucket)}</b>"
            f"{get_rank_str(top_off_bucket)}. "
            f"Total offensive upside: ~{wins_off:.1f} wins to league average."
        )

    lg_expl  = expl.get("expl_lg_rate", float("nan"))
    off_expl = expl.get("expl_off_rate", float("nan"))
    def_expl = expl.get("expl_def_allowed_rate", float("nan"))

    if not any(np.isnan(v) for v in [lg_expl, off_expl, def_expl]):
        bl_bullets.append(
            f"<b>Explosive plays are a two-sided problem</b> — {team} creates them at "
            f"{off_expl:.1%} (league avg {lg_expl:.1%}) and allows them at {def_expl:.1%}."
        )

    exec_summary_bullets = '<ol class="bl-list">\n'
    for b in bl_bullets:
        exec_summary_bullets += f"  <li>{b}</li>\n"
    exec_summary_bullets += "</ol>"

    def build_bullets(rows: pd.DataFrame, side: str) -> List[tuple]:
        bullets = []
        for _, row in rows.head(3).iterrows():
            lbl    = bucket_label(row["bucket"])
            rank_s = get_rank_str(row["bucket"])
            swing  = row["Potential_EPA_swing_to_LG"]
            t_epa  = row["TEAM_EPA_per_play"]
            lg_epa = row["LG_EPA_per_play"]
            stat   = f"{t_epa:.3f} vs {lg_epa:.3f} LG EPA/play — {swing:.0f} EPA gap (~{swing/140:.1f} wins)"
            lbl_lower = lbl.lower()

            if side == "DEF":
                if   "early down" in lbl_lower: arch = "Front-seven depth is the lever here. Winning on first and second down reduces opponent third-down opportunities and limits drive length, which is where the largest EPA gap originates."
                elif "red zone"   in lbl_lower: arch = "Interior run stoppers and press-man corners hold up best in condensed space. Stopping the run inside the 20 is the most direct path to reducing red zone EPA allowed."
                elif "short"      in lbl_lower: arch = "Pass rush and zone coverage are the primary levers on third down. Getting the quarterback off his spot limits YAC on intermediate routes and forces punts."
                elif "3rd"        in lbl_lower: arch = "Pass rush and zone coverage are the primary levers on third down. Getting the quarterback off his spot limits YAC on intermediate routes and forces punts."
                elif "explosive"  in lbl_lower: arch = "Speed at safety and length at corner cap deep shot plays. Allowing explosives creates disproportionate EPA swings that drag down overall defensive efficiency."
                else:                           arch = "Identify whether the root cause is personnel or scheme before targeting a position group."
            else:  # OFF
                if   "3rd" in lbl_lower and "long" in lbl_lower:
                    arch = "A reliable slot receiver or tight end on crossing routes converts third-and-long at a higher rate. Reducing the frequency of third-and-long situations via better early-down efficiency is equally high-leverage."
                elif "3rd" in lbl_lower and "short" in lbl_lower:
                    arch = "A physical receiver who wins in traffic on slants and quick outs, or a back who can pick up 3-4 yards consistently, closes this gap. Better early-down play calling also reduces pressure on short-yardage conversions."
                elif "3rd"        in lbl_lower:
                    arch = "Third-down conversion rate is driven by receiver separation and QB accuracy under pressure. A reliable chain-mover at WR or TE directly improves this bucket."
                elif "red zone"   in lbl_lower:
                    arch = "A jump-ball receiver or contested-catch tight end wins in condensed space where separation is limited. Red zone scoring rate is one of the strongest predictors of offensive EPA."
                elif "goal"       in lbl_lower:
                    arch = "A short-yardage back who can push the pile or a physical receiver who wins fade routes are the primary levers. Goal-to-go efficiency is heavily influenced by offensive line push and play-action design."
                elif "early down" in lbl_lower:
                    arch = "A run-game upgrade — either offensive line depth or a between-the-tackles back — sets up manageable third downs and keeps the offense on schedule. Early-down success rate is the foundation of offensive EPA."
                elif "explosive"  in lbl_lower:
                    arch = "A receiver with separation speed or a back with home-run ability directly addresses the chunk-play creation gap. Explosives create outsized EPA swings that move the needle quickly."
                else:
                    arch = "Identify whether the root cause is personnel or scheme before targeting a position group."

            bullets.append((lbl, f"{rank_s} &nbsp;|&nbsp; {stat}", arch, side))
        return bullets

    return {
        "exec_summary_bullets": exec_summary_bullets,
        "def_bullets":  build_bullets(def_rows, "DEF"),
        "off_bullets":  build_bullets(off_rows, "OFF"),
        "wins_def":     wins_def,
        "wins_off":     wins_off,
    }


# -------------------------
# Plotting
# -------------------------

def plot_top_bars(
    table: pd.DataFrame,
    prefix: str,
    cfg: Config,
    title: str,
    subtitle: str,
    out_path: str,
) -> None:
    df = (
        table[
            table["bucket"].str.startswith(prefix) &
            ~table["bucket"].isin(cfg.chart_exclude_buckets) &
            (table["Potential_EPA_swing_to_LG"] > 0)
        ]
        .sort_values("Potential_EPA_swing_to_LG", ascending=False)
        .head(cfg.top_n)
        .copy()
    )
    if df.empty:
        print(f"Warning: No improvement opportunities found for {prefix}")
        return

    labels = []
    for _, r in df.iterrows():
        base = r["bucket"].replace(prefix, "").strip()
        base = base.replace("(<=20)", "").replace("(<=4)", "").replace("(>=7)", "").replace("&", "and")
        labels.append(f"{base}  ({int(r['plays'])} plays)")

    values = df["Potential_EPA_swing_to_LG"].to_numpy()
    n      = len(values)
    fig, ax = plt.subplots(figsize=(16, max(6, n * 1.6 + 2.5)))
    y = np.arange(n)

    ax.barh(y, values, height=0.55, color='#d32f2f', alpha=0.88, edgecolor='white', linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13)
    ax.axvline(0, color='black', linewidth=1.2, alpha=0.3)
    ax.set_xlabel("Potential EPA Gain if Improved to League Average", fontsize=13, fontweight='bold', labelpad=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=12)
    fig.text(0.5, 0.98, subtitle, ha='center', va='top', fontsize=11, color='#555', style='italic')
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    max_val = max(values)
    ax.set_xlim(0, max_val * 1.18)
    for i, v in enumerate(values):
        ax.text(v + max_val * 0.015, i, f"{v:.1f}",
                va="center", ha="left", fontsize=12, fontweight='bold', color='#8b0000')

    fig.text(0.5, 0.01,
             "All bars show improvement opportunities (areas where team underperformed league average)",
             ha='center', fontsize=9.5, style='italic', color='#888')

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()


# -------------------------
# HTML Report
# -------------------------

def write_html_report(
    cfg: Config,
    leverage_table: pd.DataFrame,
    expl: dict,
    ranks_df: pd.DataFrame,
    def_chart: str,
    off_chart: str,
    out_path: str,
) -> None:
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M")
    team    = cfg.team
    exclude = set(cfg.chart_exclude_buckets)

    narrative = generate_narrative(cfg, leverage_table, ranks_df, expl)

    def prep_table(prefix: str) -> pd.DataFrame:
        df = leverage_table[
            leverage_table["bucket"].str.startswith(prefix) &
            ~leverage_table["bucket"].isin(exclude) &
            (leverage_table["Potential_EPA_swing_to_LG"] > 0)
        ].head(5).copy()

        if not ranks_df.empty:
            df = df.merge(ranks_df[["bucket", "rank", "total_teams"]], on="bucket", how="left")
            df["Rank"] = df.apply(
                lambda r: f"{rank_suffix(int(r['rank']))}/{int(r['total_teams'])}"
                if pd.notna(r.get("rank")) else "—", axis=1
            )
            df = df.drop(columns=["rank", "total_teams"], errors="ignore")

        df = df.drop(columns=["abs_potential"], errors="ignore")
        for col in ["TEAM_EPA_per_play", "LG_EPA_per_play", "TEAM_success_rate",
                    "LG_success_rate", "Potential_EPA_swing_to_LG"]:
            if col in df.columns:
                df[col] = df[col].round(3)
        return df

    def bullets_html(bullets: list) -> str:
        items = ""
        for h, rank_s, archetype, side in bullets:
            items += (
                f'  <li>'
                f'<b>{h}</b> <span class="rank-tag">{rank_s}</span>'
                f'<div class="archetype">{archetype}</div>'
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
  <title>{team} Leverage Snapshot — {cfg.season} {cfg.season_type}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 28px 40px; background: #f7f7f7; color: #111; line-height: 1.5; }}

    .header {{ background: #1a1a1a; color: white; padding: 18px 28px; border-radius: 6px; margin-bottom: 20px; display: flex; align-items: baseline; gap: 18px; }}
    .header h1 {{ margin: 0; font-size: 22px; }}
    .header .sub {{ color: #aaa; font-size: 12px; margin: 0; }}

    /* FIX #1: Introduction summary block */
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

    .badges {{ background: white; border-radius: 4px; padding: 14px 16px; }}
    .badges h4 {{ margin: 0 0 8px 0; font-size: 12px; text-transform: uppercase; color: #888; letter-spacing: 0.05em; }}
    .wins-badge {{
      display: block; background: #d32f2f; color: white;
      border-radius: 4px; padding: 6px 12px; font-size: 14px;
      font-weight: bold; margin-bottom: 6px; text-align: center;
    }}
    .badge-note {{ font-size: 10.5px; color: #999; text-align: center; }}

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
    .col-table {{ }}
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
    .fa-table {{ border-collapse: collapse; width: 100%; font-size: 11.5px; margin-top: 0; }}
    .fa-table th {{ background: #f0f0f0; padding: 4px 8px; font-weight: bold; color: #333; border: 1px solid #e0e0e0; }}
    .fa-table td {{ padding: 4px 8px; border: 1px solid #e0e0e0; }}
    .fa-table tr:nth-child(even) {{ background: #fafafa; }}
    .fa-table .scout-note {{ font-size: 10.5px; color: #555; font-style: italic; max-width: 280px; }}

    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #e0e0e0; padding: 7px 10px; text-align: left; }}
    th {{ background: #f0f0f0; font-weight: bold; color: #333; }}
    tr:nth-child(even) {{ background: #fafafa; }}

    .disclaimer {{ color: #999; font-size: 11px; margin-top: 8px; padding-top: 12px; border-top: 1px solid #e0e0e0; }}
  </style>
</head>
<body>

  <div class="header">
    <h1>{team} Leverage Snapshot — {cfg.season} {cfg.season_type}</h1>
    <p class="sub">Generated {ts} &nbsp;|&nbsp; Public EPA (nflverse) &nbsp;|&nbsp; Prioritization context only</p>
  </div>

  <!-- Introduction + Bottom Line (replaces dense exec summary paragraph) -->
  <div class="intro">
    <h2>What This Report Is</h2>
    <ul>
      <li><b>Purpose:</b> Identify {team}'s biggest improvement opportunities by comparing efficiency against league averages in key game situations, weighted by how often those situations occur.</li>
      <li><b>How to Read:</b> Bigger bars in the charts = bigger total gains if {team} improves to league average in that area. Each section includes the gap, a recommended player archetype, and specific free agent targets.</li>
      <li><b>FA Targets:</b> Players are ranked by <em>Value Ratio</em> (projected market value ÷ prior salary). Higher ratio = player is being paid well below what he's worth. Each includes a scouting note explaining the fit.</li>
    </ul>
    <p class="note">Built from public EPA data (nflverse). Intended as prioritization context — not a substitute for internal scouting, scheme fit, or medical evaluations.</p>
  </div>

  <!-- Bottom Line + Key Numbers -->
  <div class="top-strip">
    <div class="exec-summary">
      <div class="bl-title">BOTTOM LINE</div>
      {narrative["exec_summary_bullets"]}
    </div>
    <div class="right-panel">
      <div class="badges">
        <h4>Estimated Upside</h4>
        <span class="wins-badge">DEF: ~{narrative['wins_def']:.1f} wins</span>
        <span class="wins-badge" style="background:#b71c1c;">OFF: ~{narrative['wins_off']:.1f} wins</span>
        <p class="badge-note">~140 EPA ≈ 1 win (directional estimate)</p>
      </div>
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

  <!-- Defense -->
  <div class="section">
    <h2>Defense: Highest Leverage Gaps</h2>
    <div class="section-body">
      <div class="left-col">
        <img src="{os.path.basename(def_chart)}" alt="DEF chart"/>
        <div class="col-table">
          <h3 class="table-label">Detailed Breakdown</h3>
          {top_def.to_html(index=False)}
        </div>
      </div>
      <div class="bullets">
        <h3>Priority Targets</h3>
        {bullets_html(narrative["def_bullets"])}
      </div>
    </div>
  </div>

  <!-- Offense -->
  <div class="section">
    <h2>Offense: Highest Leverage Gaps</h2>
    <div class="section-body">
      <div class="left-col">
        <img src="{os.path.basename(off_chart)}" alt="OFF chart"/>
        <div class="col-table">
          <h3 class="table-label">Detailed Breakdown</h3>
          {top_off.to_html(index=False)}
        </div>
      </div>
      <div class="bullets">
        <h3>Priority Targets</h3>
        {bullets_html(narrative["off_bullets"])}
      </div>
    </div>
  </div>

  <p class="disclaimer">
    Potential EPA swing = gap between {team} and league-average efficiency scaled by play volume.
    Rank among qualifying teams (min {cfg.min_team_plays} plays/bucket). Wins estimate uses ~140 EPA/win — directional only.
    Does not account for opponent quality, scheme, or injury context. Not a substitute for internal evaluation.
  </p>

</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# -------------------------
# CLI / Main
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EPA leverage snapshot (gap x volume) for a team.")
    p.add_argument("--season",           type=int, required=True,        help="Season year, e.g. 2025")
    p.add_argument("--team",             type=str, required=True,        help="Team abbreviation, e.g. ARI")
    p.add_argument("--out",              type=str, default="./out",      help="Output directory")
    p.add_argument("--top-n",            type=int, default=6,            help="Top N bars per chart")
    p.add_argument("--min-team-plays",   type=int, default=30,           help="Min team plays per bucket")
    p.add_argument("--min-league-plays", type=int, default=200,          help="Min league plays per bucket")
    p.add_argument("--no-html",          action="store_true",            help="Disable HTML report")
    p.add_argument("--log-level",        type=str, default="INFO",       help="INFO / WARNING / ERROR / DEBUG")
    return p.parse_args()


def main() -> int:
    args   = parse_args()
    logger = setup_logger(args.log_level)

    team = args.team.strip().upper()
    if len(team) != 3:
        logger.error("Team should be a 3-letter abbreviation (e.g., ARI).")
        return 2
    if not (1999 <= args.season <= 2100):
        logger.error("Season looks invalid. Use a realistic year like 2025.")
        return 2

    cfg = Config(
        season=args.season, team=team, out_dir=args.out,
        top_n=args.top_n, min_team_plays=args.min_team_plays,
        min_league_plays=args.min_league_plays, write_html=not args.no_html,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)

    try:
        pbp = load_pbp_data(cfg.season, logger)
        pbp = safe_filter_real_plays(pbp, cfg, logger)
        pbp = compute_flags(pbp, cfg)
        validate_required_columns(pbp, ["posteam", "defteam"])

        table    = build_leverage_table(pbp, cfg, logger)
        expl     = explosive_rates(pbp, cfg.team)
        ranks_df = compute_league_ranks(pbp, cfg, logger)

        csv_path = os.path.join(cfg.out_dir, f"{team}_{cfg.season}_{cfg.season_type}_leverage_table.csv")
        table.to_csv(csv_path, index=False)
        logger.info(f"Saved table: {csv_path}")

        def_chart = os.path.join(cfg.out_dir, f"{team}_{cfg.season}_{cfg.season_type}_DEF_top.png")
        off_chart = os.path.join(cfg.out_dir, f"{team}_{cfg.season}_{cfg.season_type}_OFF_top.png")

        plot_top_bars(table, "DEF:", cfg,
            title=f"{team} Defense: Top Improvement Opportunities",
            subtitle=f"{cfg.season} Season • Areas where defense underperformed league average",
            out_path=def_chart)

        plot_top_bars(table, "OFF:", cfg,
            title=f"{team} Offense: Top Improvement Opportunities",
            subtitle=f"{cfg.season} Season • Areas where offense underperformed league average",
            out_path=off_chart)

        logger.info(f"Saved charts: {def_chart} | {off_chart}")

        if cfg.write_html:
            html_path = os.path.join(cfg.out_dir, f"{team}_{cfg.season}_{cfg.season_type}_report.html")
            write_html_report(cfg, table, expl, ranks_df, def_chart, off_chart, html_path)
            logger.info(f"Saved HTML report: {html_path}")

        logger.info(
            f"Explosives OFF {expl['expl_off_rate']:.2%} vs LG {expl['expl_lg_rate']:.2%} | "
            f"DEF allowed {expl['expl_def_allowed_rate']:.2%} vs LG {expl['expl_lg_rate']:.2%}"
        )
        logger.info("Done.")
        return 0

    except Exception as e:
        logger.error(f"Run failed: {e}")
        logger.debug("Full exception:", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())