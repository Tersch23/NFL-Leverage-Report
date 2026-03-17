# Steps

# Arizona Cardinals: Highest_leverage EPA Gains (Offense + Defense) - 2025 Regular Season

# Split plays into High-leverage buckets (3rd down, red zone, early downs, explosive plays, etc.)

# Compute Cardinals EPA vs League Average in each bucket

# Translate the gap into potential season EPA Swing if ARI moved to league average:
    # Potential_EPA_Swing = (LeagueAvgEPA - ARI_EPA) * AIR_PlayCount

# Rank the buckets so you can say:
    # If you could fic 1-2 things, these are the biggest needle movers



# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# 1) Load play-by-play data
# 2025 regular season
SEASON = 2025
TEAM = "ARI"

# nflreadpy is a python port of nflreadr for nflverse data
from nflreadpy import load_pbp

pbp = load_pbp(seasons=[SEASON])  # pulls nflverse pbp with EPA columns
pbp = pbp.to_pandas()



# 2) Clean / filter to "real" plays
# Safe filters that work across nflverse schema versions
pbp = pbp[pbp["season_type"] == "REG"].copy()

pbp = pbp[pbp["play_type"].isin(["pass", "run"])]

pbp = pbp[pbp["epa"].notna()]

# Handle optional columns gracefully
if "qb_kneel" in pbp.columns:
    pbp = pbp[pbp["qb_kneel"] != 1]

if "qb_spike" in pbp.columns:
    pbp = pbp[pbp["qb_spike"] != 1]

# no_play is not always present in nflreadpy output
if "no_play" in pbp.columns:
    pbp = pbp[pbp["no_play"] != 1]

print(pbp.shape)
print(pbp.columns)




# 3) Helper columns (situational buckets)
pbp["is_third_down"] = pbp["down"] == 3
pbp["is_fourth_down"] = pbp["down"] == 4
pbp["is_early_down"] = pbp["down"].isin([1, 2])

pbp["is_red_zone"] = pbp["yardline_100"].between(1, 20, inclusive="both")
pbp["is_goal_to_go"] = pbp["goal_to_go"] == 1

pbp["is_explosive"] = pbp["yards_gained"] >= np.where(pbp["play_type"]=="pass", 20, 10)

# Define "high-leverage 3rd down" as 3rd & 4 or less vs 3rd & long
pbp["third_short"] = (pbp["down"] == 3) & (pbp["ydstogo"] <= 4)
pbp["third_long"]  = (pbp["down"] == 3) & (pbp["ydstogo"] >= 7)

# Success rate proxy using EPA (success if EPA > 0)
pbp["success"] = (pbp["epa"] > 0).astype(int)




# 4) Build bucket definitions
# We'll measure offense using posteam == TEAM
# Defense using defteam == TEAM (EPA is from offense perspective, so "lower allowed" is better)
bucket_defs = {
    "OFF: All plays":              lambda d: (d["posteam"] == TEAM),
    "OFF: Early downs":            lambda d: (d["posteam"] == TEAM) & (d["is_early_down"]),
    "OFF: 3rd down":               lambda d: (d["posteam"] == TEAM) & (d["is_third_down"]),
    "OFF: 3rd & short (<=4)":      lambda d: (d["posteam"] == TEAM) & (d["third_short"]),
    "OFF: 3rd & long (>=7)":       lambda d: (d["posteam"] == TEAM) & (d["third_long"]),
    "OFF: Red zone (<=20)":        lambda d: (d["posteam"] == TEAM) & (d["is_red_zone"]),
    "OFF: Goal-to-go":             lambda d: (d["posteam"] == TEAM) & (d["is_goal_to_go"]),
    "OFF: Explosive rate context": lambda d: (d["posteam"] == TEAM),  # used for explosive rate
    "DEF: All plays":              lambda d: (d["defteam"] == TEAM),
    "DEF: Early downs":            lambda d: (d["defteam"] == TEAM) & (d["is_early_down"]),
    "DEF: 3rd down":               lambda d: (d["defteam"] == TEAM) & (d["is_third_down"]),
    "DEF: 3rd & short (<=4)":      lambda d: (d["defteam"] == TEAM) & (d["third_short"]),
    "DEF: Red zone (<=20)":        lambda d: (d["defteam"] == TEAM) & (d["is_red_zone"]),
    "DEF: Explosives allowed ctx": lambda d: (d["defteam"] == TEAM),  # used for explosive allowed rate
}

# League baselines for each bucket:
# Offense baseline uses posteam for "all teams"
# Defense baseline uses defteam for "all teams"
def league_mask(bucket_name, d):
    if bucket_name.startswith("OFF:"):
        # Use same condition but replace TEAM with "any team" by removing posteam constraint
        # We'll reconstruct logic for each bucket:
        # (We keep the situational filters, drop posteam == TEAM)
        if "Early downs" in bucket_name:   return d["is_early_down"]
        if "3rd & short" in bucket_name:   return d["third_short"]
        if "3rd & long" in bucket_name:    return d["third_long"]
        if "3rd down" in bucket_name:      return d["is_third_down"]
        if "Red zone" in bucket_name:      return d["is_red_zone"]
        if "Goal-to-go" in bucket_name:    return d["is_goal_to_go"]
        return d["play_type"].isin(["pass","run"])
    else:
        if "Early downs" in bucket_name:   return d["is_early_down"]
        if "3rd & short" in bucket_name:   return d["third_short"]
        if "3rd down" in bucket_name:      return d["is_third_down"]
        if "Red zone" in bucket_name:      return d["is_red_zone"]
        return d["play_type"].isin(["pass","run"])

def summarize_bucket(name):
    team_df = pbp[bucket_defs[name](pbp)]
    lg_df   = pbp[league_mask(name, pbp)]

    # Handle small sample buckets safely
    if len(team_df) < 30 or len(lg_df) < 200:
        return None

    team_epa = team_df["epa"].mean()
    lg_epa   = lg_df["epa"].mean()
    n_plays  = len(team_df)

    # Potential swing:
    # OFF: improvement = (lg - team) * plays
    # DEF: improvement = (team_allowed - lg_allowed) * plays  (since lower allowed is better)
    if name.startswith("OFF:"):
        potential_epa = (lg_epa - team_epa) * n_plays
    else:
        potential_epa = (team_epa - lg_epa) * n_plays

    team_sr = team_df["success"].mean()
    lg_sr   = lg_df["success"].mean()

    return {
        "bucket": name,
        "plays": n_plays,
        "ARI_EPA_per_play": team_epa,
        "LG_EPA_per_play": lg_epa,
        "ARI_success_rate": team_sr,
        "LG_success_rate": lg_sr,
        "Potential_EPA_swing_to_LG": potential_epa
    }

rows = [summarize_bucket(k) for k in bucket_defs.keys()]
rows = [r for r in rows if r is not None]
out = pd.DataFrame(rows)

# Rank the biggest opportunities (positive potential = biggest gain if fixed)
out["abs_potential"] = out["Potential_EPA_swing_to_LG"].abs()
out_sorted = out.sort_values("Potential_EPA_swing_to_LG", ascending=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

print(out_sorted.head(12))




# 5) Add explosive rate comparisons (simple but very GM-readable)
def rate_explosive_offense(df, team):
    t = df[(df["posteam"]==team)]
    return t["is_explosive"].mean()

def rate_explosive_defense(df, team):
    t = df[(df["defteam"]==team)]
    return t["is_explosive"].mean()

ari_expl_off = rate_explosive_offense(pbp, TEAM)
lg_expl_off  = pbp["is_explosive"].mean()

ari_expl_def = rate_explosive_defense(pbp, TEAM)
lg_expl_def  = pbp["is_explosive"].mean()

print("Explosive rate (OFF): ARI vs LG", ari_expl_off, lg_expl_off)
print("Explosive allowed rate (DEF): ARI vs LG", ari_expl_def, lg_expl_def)




# 6) Simple charts for the PDF
top = out_sorted[out_sorted["Potential_EPA_swing_to_LG"]>0].head(8).copy()

plt.figure(figsize=(10,5))
plt.barh(top["bucket"][::-1], top["Potential_EPA_swing_to_LG"][::-1])
plt.title(f"{TEAM} Highest-Leverage Buckets (Potential EPA swing to League Avg) — {SEASON} REG")
plt.xlabel("Potential Total EPA Swing (season)")
plt.tight_layout()
plt.savefig("ari_top_epa_opportunities.png", dpi=220)
plt.show()

# Table export
cols = ["bucket","plays","ARI_EPA_per_play","LG_EPA_per_play","ARI_success_rate","LG_success_rate","Potential_EPA_swing_to_LG"]
out_sorted[cols].to_csv("ari_bucket_epa_table.csv", index=False)

print("Saved: ari_top_epa_opportunities.png and ari_bucket_epa_table.csv")
