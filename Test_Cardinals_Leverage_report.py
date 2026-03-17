# cardinals_leverage_report.py
# To run: python .\Test_Cardinals_Leverage_Report.py --season 2025 --team ARI

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
    """Clean, properly-cased label from a bucket string like 'OFF: 3rd & long (>=7)'."""
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

    exec_lines = []

    if top_def_bucket:
        r = def_rows.iloc[0]
        exec_lines.append(
            f"{team}'s largest efficiency gap is on defense — specifically {bucket_label(top_def_bucket)}, "
            f"where they allowed {r['TEAM_EPA_per_play']:.3f} EPA/play vs. a league average of "
            f"{r['LG_EPA_per_play']:.3f}{get_rank_str(top_def_bucket)}. "
            f"Closing that gap to league average is worth an estimated {wins_def:.1f} wins of upside "
            f"(~{total_def_swing:.0f} total EPA, using a ~140 EPA/win approximation)."
        )

    if top_off_bucket:
        r = off_rows.iloc[0]
        exec_lines.append(
            f"On offense, the primary gap is {bucket_label(top_off_bucket)}: "
            f"{team} generated {r['TEAM_EPA_per_play']:.3f} EPA/play vs. "
            f"{r['LG_EPA_per_play']:.3f} league average{get_rank_str(top_off_bucket)}. "
            f"Offensive upside totals approximately {wins_off:.1f} wins if brought to league average."
        )

    lg_expl  = expl.get("expl_lg_rate", float("nan"))
    off_expl = expl.get("expl_off_rate", float("nan"))
    def_expl = expl.get("expl_def_allowed_rate", float("nan"))

    if not any(np.isnan(v) for v in [lg_expl, off_expl, def_expl]):
        exec_lines.append(
            f"Explosive play rates compound both gaps: {team} creates explosives at {off_expl:.1%} "
            f"(league avg {lg_expl:.1%}, {'below' if off_expl < lg_expl else 'above'} average) "
            f"and allows them at {def_expl:.1%} "
            f"({'above' if def_expl > lg_expl else 'below'} average) — a double-sided leverage point."
        )

    # Archetype bullets
    def build_bullets(rows: pd.DataFrame, side: str) -> List[tuple]:
        bullets = []
        for _, row in rows.head(3).iterrows():
            lbl   = bucket_label(row["bucket"])
            rank_s = get_rank_str(row["bucket"])
            swing  = row["Potential_EPA_swing_to_LG"]
            t_epa  = row["TEAM_EPA_per_play"]
            lg_epa = row["LG_EPA_per_play"]

            if side == "DEF":
                body = (
                    f"{team} allowed {t_epa:.3f} EPA/play vs. {lg_epa:.3f} league average{rank_s}. "
                    f"Potential EPA swing to league average: {swing:.1f} (~{swing/140:.1f} wins). "
                )
                if   "early down" in lbl: body += "Archetype leverage: front-seven depth that wins 1st/2nd down — reduces opponent 3rd-down conversion rate and limits drive length."
                elif "red zone"   in lbl: body += "Archetype leverage: interior run-stoppers and man-coverage corners that hold up in compressed space."
                elif "short"      in lbl: body += "Archetype leverage: physical defensive linemen and safeties who can crowd the box without giving up the run/pass option."
                elif "3rd"        in lbl: body += "Archetype leverage: pass rushers and zone-coverage profiles that limit YAC on intermediate routes."
                elif "explosive"  in lbl: body += "Archetype leverage: coverage profiles that cap deep gains — speed at safety and length at corner."
                else:                     body += "Review film for schematic vs. personnel root cause before targeting a specific archetype."
            else:
                body = (
                    f"{team} generated {t_epa:.3f} EPA/play vs. {lg_epa:.3f} league average{rank_s}. "
                    f"Potential EPA swing to league average: {swing:.1f} (~{swing/140:.1f} wins). "
                )
                if   "explosive"  in lbl: body += "Archetype leverage: a receiver with separation speed or a back with home-run ability to generate chunk plays."
                elif "red zone"   in lbl: body += "Archetype leverage: a tight end or jump-ball receiver who wins in condensed areas; red zone scoring rate is a strong TD predictor."
                elif "long"       in lbl: body += "Archetype leverage: a reliable slot receiver or tight end on crossing routes — reducing 3rd-and-long frequency is equally important."
                elif "early down" in lbl: body += "Archetype leverage: a run-game upgrade (either line depth or a between-the-tackles back) to set up manageable 3rd downs."
                else:                     body += "Review scheme vs. personnel before targeting a specific archetype."

            bullets.append((lbl, body))
        return bullets

    return {
        "exec_summary": " ".join(exec_lines),
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

    ax.text(0.5, -0.08,
            "All bars show improvement opportunities (areas where team underperformed league average)",
            transform=ax.transAxes, ha='center', fontsize=9.5, style='italic', color='#666',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', edgecolor='#ddd', alpha=0.9))

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
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

        # Merge rank
        if not ranks_df.empty:
            df = df.merge(ranks_df[["bucket", "rank", "total_teams"]], on="bucket", how="left")
            df["Rank"] = df.apply(
                lambda r: f"{rank_suffix(int(r['rank']))}/{int(r['total_teams'])}"
                if pd.notna(r.get("rank")) else "—", axis=1
            )
            df = df.drop(columns=["rank", "total_teams"], errors="ignore")

        # Drop redundant column, round floats
        df = df.drop(columns=["abs_potential"], errors="ignore")
        for col in ["TEAM_EPA_per_play", "LG_EPA_per_play", "TEAM_success_rate",
                    "LG_success_rate", "Potential_EPA_swing_to_LG"]:
            if col in df.columns:
                df[col] = df[col].round(3)
        return df

    def bullets_html(bullets: list) -> str:
        return "<ul>\n" + "".join(
            f"  <li><b>{h}:</b> {b}</li>\n" for h, b in bullets
        ) + "</ul>"

    top_def = prep_table("DEF:")
    top_off = prep_table("OFF:")

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{team} Leverage Snapshot — {cfg.season} {cfg.season_type}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; max-width: 1100px; line-height: 1.5; }}
    h1   {{ margin-bottom: 4px; font-size: 22px; }}
    h2   {{ font-size: 16px; color: #222; margin-top: 28px; margin-bottom: 6px;
            border-bottom: 2px solid #d32f2f; padding-bottom: 4px; }}
    h3   {{ font-size: 14px; margin-bottom: 4px; }}
    .sub {{ color: #666; margin-top: 0; font-size: 12px; }}
    .exec-summary {{
      background: #fff8f8; border-left: 4px solid #d32f2f;
      padding: 14px 18px; margin: 16px 0; font-size: 14px; line-height: 1.7;
    }}
    .wins-badge {{
      display: inline-block; background: #d32f2f; color: white;
      border-radius: 4px; padding: 3px 10px; font-size: 13px;
      font-weight: bold; margin: 4px 4px 4px 0;
    }}
    .callouts {{ margin: 14px 0; padding: 12px 16px; background: #fafafa; border: 1px solid #eee; font-size: 13px; }}
    .grid     {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 16px; }}
    img       {{ max-width: 100%; border: 1px solid #ddd; padding: 6px; background: #fff; }}
    table     {{ border-collapse: collapse; width: 100%; font-size: 11.5px; margin-top: 8px; }}
    th, td    {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th        {{ background: #f5f5f5; font-weight: bold; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    ul  {{ margin-top: 6px; padding-left: 18px; }}
    li  {{ margin-bottom: 8px; font-size: 13px; }}
    .mono {{ font-family: ui-monospace, Menlo, Consolas, monospace; }}
    .disclaimer {{ color: #888; font-size: 11px; margin-top: 24px; border-top: 1px solid #eee; padding-top: 10px; }}
  </style>
</head>
<body>

  <h1>{team} Leverage Snapshot — {cfg.season} {cfg.season_type}</h1>
  <p class="sub">Generated {ts} &nbsp;|&nbsp; Public play-by-play EPA (nflverse) &nbsp;|&nbsp; For prioritization context only</p>

  <h2>Executive Summary</h2>
  <div class="exec-summary">{narrative["exec_summary"]}</div>

  <div style="margin: 10px 0 18px 0;">
    <span class="wins-badge">DEF upside: ~{narrative['wins_def']:.1f} wins</span>
    <span class="wins-badge">OFF upside: ~{narrative['wins_off']:.1f} wins</span>
    <span class="sub" style="font-size:11px;">&nbsp;Rough estimate: ~140 EPA ≈ 1 win above replacement</span>
  </div>

  <div class="callouts">
    <b>Explosive Play Rates</b> &nbsp;(pass ≥20 yds &nbsp;|&nbsp; run ≥10 yds)<br/>
    <span class="mono">OFF: {expl['expl_off_rate']:.2%}</span> vs League <span class="mono">{expl['expl_lg_rate']:.2%}</span>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <span class="mono">DEF Allowed: {expl['expl_def_allowed_rate']:.2%}</span> vs League <span class="mono">{expl['expl_lg_rate']:.2%}</span>
  </div>

  <h2>Defense: Highest Leverage Gaps</h2>
  <div class="grid">
    <div><img src="{os.path.basename(def_chart)}" alt="DEF chart"/></div>
    <div>
      <h3>Archetype Targets</h3>
      {bullets_html(narrative["def_bullets"])}
    </div>
  </div>
  <h3>Top Defensive Buckets</h3>
  {top_def.to_html(index=False)}

  <h2>Offense: Highest Leverage Gaps</h2>
  <div class="grid">
    <div><img src="{os.path.basename(off_chart)}" alt="OFF chart"/></div>
    <div>
      <h3>Archetype Targets</h3>
      {bullets_html(narrative["off_bullets"])}
    </div>
  </div>
  <h3>Top Offensive Buckets</h3>
  {top_off.to_html(index=False)}

  <p class="disclaimer">
    Potential EPA swing = gap between {team} and league-average efficiency scaled by play volume.
    Rank is among qualifying teams (min {cfg.min_team_plays} plays per bucket).
    Wins-equivalent uses a ~140 EPA/win approximation — treat as directional, not precise.
    This is a public-data snapshot and does not account for opponent quality, scheme context,
    or injury-adjusted rosters. Not a substitute for internal evaluation.
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

        # CSV
        csv_path = os.path.join(cfg.out_dir, f"{team}_{cfg.season}_{cfg.season_type}_leverage_table.csv")
        table.to_csv(csv_path, index=False)
        logger.info(f"Saved table: {csv_path}")

        # Charts
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

        # HTML
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