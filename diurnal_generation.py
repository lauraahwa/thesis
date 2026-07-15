#!/usr/bin/env python3
"""Weighted average-day (diurnal) generation profiles from GenX TDR results.

Computes the average MW by hour-of-day across the modeled year, weighting each
model hour by its TDR time weight (results_pN/time_weights.csv). Mathematically identical to expanding the 52 representative weeks back to the
full multi-year record via Period_map and averaging there — a plain unweighted
average would over-weight the extreme (peak) weeks by ~7x.

Usage:
    python3 diurnal_generation.py \
        --orig scenarios/z10/PJM_DC_Island_750MW_z10 \
        --dr   scenarios/z10/PJM_DC_Island_750MW_z10_dr \
        --period 1 --zones pjm --out sample_plots/diurnal_750MW_z10_p1.png
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Zones 3 (ISONE), 9 (MISC), 11/12 (NY), 21 (SERC) are border zones, consistent
# with non_pjm_zones in utils.py / consumer_cost_utils.py. Everything else,
# including the DC Island (z28) when present, counts as PJM.
NON_PJM_ZONES = {3, 9, 11, 12, 21}
DC_ISLAND_ZONE = 28

# Stack order, bottom -> top. DR is shed load (the demand_response module's
# injection), not generation, so it sits on top as a hatched band.
STACK_ORDER = ["Nuclear", "Coal", "NGCC", "NGCT", "Hydro", "Wind",
               "SolarPV", "Battery", "Other", "DR"]

COLORS = {
    "Nuclear": "#FFA400",
    "Coal":    "#131313",
    "NGCC":    "#4581B4",
    "NGCT":    "#758698",
    "Hydro":   "#19196F",
    "Wind":    "#84CDF9",
    "SolarPV": "#FEFE00",
    "Battery": "#7F007F",
    "Other":   "#b0afa8",  
    "DR":      "#b0afa8",   
}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"


def classify(name):
    """Map a GenX resource name to a technology group."""
    n = name.lower()
    if n.endswith("_dr") or "_dr_" in n:
        return "DR"
    if "nuclear" in n:
        return "Nuclear"
    if "coal" in n:
        return "Coal"
    if "combined_cycle" in n or n.endswith("_cc_new"):
        return "NGCC"
    if "combustion_turbine" in n or n.endswith("_ct_new"):
        return "NGCT"
    if "hydroelectric" in n:      # includes hydroelectric_pumped_storage
        return "Hydro"
    if "wind" in n:
        return "Wind"
    if "pv" in n or "solar" in n or "distributed_generation" in n:
        return "SolarPV"
    if "batter" in n:
        return "Battery"
    return "Other"


def zone_set(spec):
    """Resolve a --zones spec to a set of zone numbers (or None for all)."""
    if spec == "all":
        return None
    if spec == "pjm":
        return {"exclude": NON_PJM_ZONES}
    if spec == "island":
        return {"include": {DC_ISLAND_ZONE}}
    return {"include": {int(z) for z in spec.split(",")}}


def diurnal_by_tech(case_dir, period, zones="pjm", verbose=False):
    """Return a 24 x tech-group DataFrame of weight-averaged MW by hour of day."""
    res_dir = os.path.join(case_dir, "results", f"results_p{period}")
    power = pd.read_csv(os.path.join(res_dir, "power.csv"), index_col=0)
    weights = pd.read_csv(os.path.join(res_dir, "time_weights.csv"))["Weight"].to_numpy()

    power = power.drop(columns=["Total"], errors="ignore")
    res_zone = power.loc["Zone"].astype(float).astype(int)
    data = power.drop(index=["Zone", "AnnualSum"]).astype(float)
    T = len(data)
    if T != len(weights):
        raise ValueError(f"{case_dir}: {T} timesteps but {len(weights)} weights")
    if T % 24:
        raise ValueError(f"{case_dir}: {T} timesteps is not a whole number of days")

    zf = zone_set(zones)
    if zf is not None:
        if "include" in zf:
            keep = res_zone[res_zone.isin(zf["include"])].index
        else:
            keep = res_zone[~res_zone.isin(zf["exclude"])].index
        data = data[keep]

    groups = {}
    for col in data.columns:
        groups.setdefault(classify(col), []).append(col)
    if verbose:
        for g in STACK_ORDER:
            if g in groups:
                print(f"  {g:8s} <- {len(groups[g])} resources")
        if "Other" in groups:
            techs = sorted({c.split("_", 1)[-1] for c in groups["Other"]})
            print(f"  Other contains: {techs}")

    hod = np.arange(T) % 24                      # t1 -> hour 0; 168 % 24 == 0 so
    wsum = np.bincount(hod, weights, minlength=24)  # every rep week aligns
    out = {}
    for g, cols in groups.items():
        x = data[cols].sum(axis=1).to_numpy()
        out[g] = np.bincount(hod, weights * x, minlength=24) / wsum
    df = pd.DataFrame(out, index=range(24))
    return df.reindex(columns=[g for g in STACK_ORDER if g in df.columns], fill_value=0.0)


def _style_axis(ax):
    ax.set_facecolor("white")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.margins(y=0)


def _stack(ax, df, unit_div):
    hours = df.index.to_numpy()
    base = np.zeros(len(df))
    for g in df.columns:
        top = base + df[g].to_numpy() / unit_div
        ax.fill_between(hours, base, top,
                        facecolor=COLORS[g], edgecolor="white", linewidth=0.7,
                        hatch="///" if g == "DR" else None)
        base = top
    return base  # total


def plot_comparison(orig_df, dr_df, labels, title, out_png):
    vmax = max(orig_df.sum(axis=1).max(), dr_df.sum(axis=1).max())
    unit_div, unit = (1000.0, "GW") if vmax > 10000 else (1.0, "MW")

    cols = [g for g in STACK_ORDER if g in set(orig_df.columns) | set(dr_df.columns)]
    orig_df = orig_df.reindex(columns=cols, fill_value=0.0)
    dr_df = dr_df.reindex(columns=cols, fill_value=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor="white")
    for ax, df, label in zip(axes, (orig_df, dr_df), labels):
        _stack(ax, df, unit_div)
        _style_axis(ax)
        ax.set_title(label, fontsize=11, color=INK, pad=8)
        ax.set_xlabel("Hour of day", fontsize=9.5, color=INK2)
    axes[0].set_ylabel(f"Average generation ({unit})", fontsize=9.5, color=INK2)

    handles = [Patch(facecolor=COLORS[g], edgecolor="white",
                     hatch="///" if g == "DR" else None, label=g)
               for g in reversed(cols)]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.905, 0.5),
               frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle(title, fontsize=12, color=INK, x=0.08, ha="left")
    fig.subplots_adjust(left=0.08, right=0.89, top=0.84, bottom=0.13, wspace=0.08)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_difference(orig_df, dr_df, labels, title, out_png):
    """Line plot of (dr - orig) by technology, per hour of day.

    Reveals the formulation difference the near-identical stacks hide. Each tech
    is one line; a bold neutral line marks the net change in total generation.
    """
    cols = [g for g in STACK_ORDER if g in set(orig_df.columns) | set(dr_df.columns)]
    orig_df = orig_df.reindex(columns=cols, fill_value=0.0)
    dr_df = dr_df.reindex(columns=cols, fill_value=0.0)
    delta = dr_df - orig_df

    vmax = delta.abs().to_numpy().max()
    unit_div, unit = (1000.0, "GW") if vmax > 10000 else (1.0, "MW")
    hours = delta.index.to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="white")
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    for g in cols:
        ax.plot(hours, delta[g].to_numpy() / unit_div,
                color=COLORS[g], linewidth=2, label=g)
    ax.plot(hours, delta.sum(axis=1).to_numpy() / unit_div,
            color=INK, linewidth=2.5, linestyle=(0, (4, 2)), label="Net total")
    _style_axis(ax)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("#c3c2b7")
    ax.set_xlabel("Hour of day", fontsize=9.5, color=INK2)
    ax.set_ylabel(f"{labels[1]} − {labels[0]}  ({unit})", fontsize=9.5, color=INK2)
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=8)

    handles = [plt.Line2D([], [], color=COLORS[g], linewidth=2, label=g) for g in cols]
    handles.append(plt.Line2D([], [], color=INK, linewidth=2.5,
                              linestyle=(0, (4, 2)), label="Net total"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=9, labelcolor=INK2)
    fig.subplots_adjust(left=0.1, right=0.82, top=0.9, bottom=0.12)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--orig", required=True, help="original case folder")
    p.add_argument("--dr", required=True, help="DR / peak-shaving case folder")
    p.add_argument("--period", type=int, default=1)
    p.add_argument("--zones", default="pjm",
                   help="'pjm' (default), 'island', 'all', or comma-separated zone numbers")
    p.add_argument("--out", required=True, help="output PNG path")
    p.add_argument("--labels", default="Original,Demand response")
    p.add_argument("--diff", action="store_true",
                   help="plot (dr - orig) difference by tech instead of side-by-side stacks")
    args = p.parse_args()

    print(f"[orig] {args.orig}")
    orig = diurnal_by_tech(args.orig, args.period, args.zones, verbose=True)
    print(f"[dr]   {args.dr}")
    dr = diurnal_by_tech(args.dr, args.period, args.zones, verbose=True)

    zone_lbl = {"pjm": "PJM zones", "island": "DC Island (z28)", "all": "all zones"}.get(
        args.zones, f"zones {args.zones}")
    kind = "difference" if args.diff else "generation by technology"
    title = f"Average day, {kind} — {zone_lbl}, p{args.period}"
    labels = args.labels.split(",")
    if args.diff:
        plot_difference(orig, dr, labels, title, args.out)
    else:
        plot_comparison(orig, dr, labels, title, args.out)


if __name__ == "__main__":
    main()
