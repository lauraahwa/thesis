"""
Analytics tools for completed GenX scenario results.
Provides NPV, capacity cost ($/MW-day), and capacity mix calculations.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import GENX_DIR
from tools.slurm import find_case

sys.path.insert(0, GENX_DIR)
from utils import PJM_ZONES, CAPRES_DICT, get_capacity_mix as _get_capacity_mix

# ── Constants ──────────────────────────────────────────────────────────────────

RESOURCE_FILES = ["Thermal.csv", "Vre.csv", "Storage.csv", "Hydro.csv", "Must_run.csv"]
INV_COST_COLS  = ["Inv_Cost_per_MWyr", "Inv_Cost_per_MWhyr"]
NON_PJM_ZONES  = [3, 8, 9, 11, 12, 21]
ALL_ZONES      = PJM_ZONES + NON_PJM_ZONES
PJM_CAPRES     = [3, 4, 5]   # RTO Main, DOM, E-MAAC


# ── Private helpers ────────────────────────────────────────────────────────────

def _read_ms_settings(scenario_path):
    yml = os.path.join(scenario_path, "settings", "multi_stage_settings.yml")
    with open(yml) as f:
        return yaml.safe_load(f)


def _load_all_resources(resource_dir):
    dfs = []
    for fname in RESOURCE_FILES:
        fpath = os.path.join(resource_dir, fname)
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        keep = ["Resource"] + [c for c in INV_COST_COLS if c in df.columns]
        dfs.append(df[keep])
    combined = pd.concat(dfs, ignore_index=True)
    for col in INV_COST_COLS:
        if col not in combined.columns:
            combined[col] = 0.0
    return combined.fillna(0)


def _get_ctotal(costs_csv, zones):
    df = pd.read_csv(costs_csv, index_col=0)
    zone_cols = [f"Zone{z}" for z in zones]
    return (
        pd.to_numeric(df.loc["cTotal", zone_cols]).sum()
        + float(df.loc["cNetworkExp", "Total"])
        + float(df.loc["cUnmetPolicyPenalty", "Total"])
    )


def _compute_trailing_capex(stage_results, stage_resources, stage_lens, zones):
    """
    For each stage k > 0, compute the annual capital recovery payments owed
    from all prior stages j < k whose CRP extends into stage k.

    Returns list of (trailing_total, detail_df), indexed from stage 1:
      result[0] = trailing owed in P2, result[1] = trailing owed in P3, etc.
    """
    ms = pd.read_csv(os.path.join(stage_resources[0], "Resource_multistage_data.csv"))
    num_stages = len(stage_results)
    results = []

    for k in range(1, num_stages):
        trailing_dfs = []

        for j in range(k):
            res = _load_all_resources(stage_resources[j])
            cap = pd.read_csv(os.path.join(stage_results[j], "capacity.csv"))
            cap = cap[cap["Zone"].isin(zones)]

            # MW builds
            new_mw = cap[cap["NewCap"] > 0][["Resource", "NewCap"]].copy()
            new_mw = new_mw.merge(res[["Resource", "Inv_Cost_per_MWyr"]], on="Resource", how="left").fillna(0)
            new_mw["annual_payment"] = new_mw["NewCap"] * new_mw["Inv_Cost_per_MWyr"]

            # MWh builds (storage energy capacity)
            new_mwh = cap[cap["NewEnergyCap"] > 0][["Resource", "NewEnergyCap"]].copy()
            new_mwh = new_mwh.merge(res[["Resource", "Inv_Cost_per_MWhyr"]], on="Resource", how="left").fillna(0)
            new_mwh["annual_payment"] = new_mwh["NewEnergyCap"] * new_mwh["Inv_Cost_per_MWhyr"]

            df = pd.concat([new_mw[["Resource", "annual_payment"]],
                            new_mwh[["Resource", "annual_payment"]]
                           ]).groupby("Resource", as_index=False)["annual_payment"].sum()
            df = df[df["annual_payment"] > 0]
            df = df.merge(ms[["Resource", "Capital_Recovery_Period"]], on="Resource", how="left")
            df["Capital_Recovery_Period"] = df["Capital_Recovery_Period"].fillna(0).astype(int)

            # Transmission expansion trailing capex
            tx_input = pd.read_csv(os.path.join(os.path.dirname(stage_resources[j]), "system", "Network.csv"))
            tx_exp   = pd.read_csv(os.path.join(stage_results[j], "network_expansion.csv"))
            tx = tx_exp[tx_exp["New_Trans_Capacity"] > 0][["Line", "New_Trans_Capacity"]].copy()
            tx = tx.merge(
                tx_input[["Network_Lines", "Line_Reinforcement_Cost_per_MWyr", "Capital_Recovery_Period"]],
                left_on="Line", right_on="Network_Lines", how="left"
            ).fillna(0)
            tx["Resource"]       = "Transmission_Line_" + tx["Line"].astype(str)
            tx["annual_payment"] = tx["New_Trans_Capacity"] * tx["Line_Reinforcement_Cost_per_MWyr"]

            years_elapsed = sum(stage_lens[j:k])

            df["still_paying"] = df["Capital_Recovery_Period"] > years_elapsed
            df["trailing"]     = df["annual_payment"] * df["still_paying"]
            df["from_stage"]   = j + 1

            tx["still_paying"] = tx["Capital_Recovery_Period"] > years_elapsed
            tx["trailing"]     = tx["annual_payment"] * tx["still_paying"]
            tx["from_stage"]   = j + 1
            tx = tx[["Resource", "annual_payment", "Capital_Recovery_Period", "still_paying", "trailing", "from_stage"]]

            trailing_dfs.append(df)
            trailing_dfs.append(tx)

        combined = pd.concat(trailing_dfs, ignore_index=True)
        results.append((combined["trailing"].sum(), combined))

    return results


# ── Public analytics functions ─────────────────────────────────────────────────

def compute_npv(scenario_path: str) -> dict:
    """
    Compute total system cost NPV for a completed multi-stage GenX scenario.
    Discounts each stage back to the P1 base year using WACC from
    multi_stage_settings.yml. Adds trailing CAPEX corrections for generation
    and transmission investments whose capital recovery period extends beyond
    the stage in which they were built.

    Returns a dict with total_npv_B, per-stage NPVs, and metadata.
    """
    s          = _read_ms_settings(scenario_path)
    wacc       = s["WACC"]
    stage_lens = s["StageLengths"]
    num_stages = s["NumStages"]

    results_dir = os.path.join(scenario_path, "results")
    inputs_dir  = os.path.join(scenario_path, "inputs")

    # Include zone 28 (DC Island) unless this is a baseline or UnifInflation scenario
    scenario_name = os.path.basename(scenario_path)
    is_reference  = any(kw in scenario_name for kw in ("Baseline", "Unif"))
    zones = ALL_ZONES if is_reference else ALL_ZONES + [28]

    stage_results   = [os.path.join(results_dir, f"results_p{i+1}") for i in range(num_stages)]
    stage_resources = [os.path.join(inputs_dir,  f"inputs_p{i+1}", "resources") for i in range(num_stages)]

    trailing_results = _compute_trailing_capex(stage_results, stage_resources, stage_lens, zones)
    ctotals = [_get_ctotal(os.path.join(stage_results[k], "costs.csv"), zones)
               for k in range(num_stages)]

    stage_npvs = []
    total = 0.0
    for k in range(num_stages):
        trailing_k = trailing_results[k-1][0] if k > 0 else 0.0
        cum_years  = sum(stage_lens[0:k])
        discount   = 1 / (1 + wacc) ** cum_years
        npv_k      = (ctotals[k] + trailing_k) * discount
        stage_npvs.append(npv_k)
        total += npv_k

    return {
        "scenario":        os.path.basename(scenario_path),
        "total_npv_B":     round(total / 1e9, 3),
        "wacc":            wacc,
        "num_stages":      num_stages,
        "stage_lens":      stage_lens,
        "stage_npvs_B":    [round(v / 1e9, 3) for v in stage_npvs],
        "stage_ctotals_B": [round(v / 1e9, 3) for v in ctotals],
        "includes_z28":    28 in zones,
    }


def compute_capacity_cost(scenario_path: str, period: int) -> dict:
    """
    Compute PJM-wide capacity cost ($/MW-day) for a given scenario and period.

    Follows the protocol:
      TotalAnnualCapacityCost_PJM = sum_r sum_t ( lambda[t,r] * regional_demand[t] * omega_t )
      Price_PJM_annual = TotalAnnualCapacityCost_PJM / max_t( sum_{z in PJM} D[t,z] )
      Price_PJM_day    = Price_PJM_annual / 365

    Only PJM CapRes regions (3=RTO Main, 4=DOM, 5=E-MAAC) are included.
    """
    dem_path    = os.path.join(scenario_path, "inputs", f"inputs_p{period}", "TDR_results", "Demand_data.csv")
    resmar_path = os.path.join(scenario_path, "results", f"results_p{period}", "ReserveMargin_w.csv")

    dem_in = pd.read_csv(dem_path)
    resmar = pd.read_csv(resmar_path)

    weights        = dem_in["Sub_Weights"].dropna().values
    hourly_weights = np.array([w / 168 for w in weights for _ in range(168)])

    pjm_total_cost = 0.0
    for capres_num in PJM_CAPRES:
        capres_col      = f"CapRes_{capres_num}"
        zone_list       = [z for z in CAPRES_DICT[capres_num] if z in PJM_ZONES]
        regional_demand = dem_in[[f"Demand_MW_z{z}" for z in zone_list]].sum(axis=1).values
        lambda_t        = resmar[capres_col].values
        pjm_total_cost += (lambda_t * regional_demand * hourly_weights).sum()

    pjm_demand_t     = dem_in[[f"Demand_MW_z{z}" for z in PJM_ZONES]].sum(axis=1).values
    pjm_peak         = pjm_demand_t.max()
    pjm_price_annual = pjm_total_cost / pjm_peak
    pjm_price_day    = pjm_price_annual / 365

    return {
        "scenario":           os.path.basename(scenario_path),
        "period":             period,
        "price_per_mw_day":   round(pjm_price_day, 2),
        "price_per_mw_yr":    round(pjm_price_annual, 2),
        "pjm_peak_demand_mw": round(float(pjm_peak), 1),
    }


def get_capacity_mix(scenario_path: str, period: int) -> dict:
    """
    Get new build, retirement, and end capacity by resource category (PJM zones only).
    Categories are defined by classify_resource() in utils.py.
    """
    mix = _get_capacity_mix(scenario_path, period, pjm_only=True)
    return {
        "scenario":        os.path.basename(scenario_path),
        "period":          period,
        "end_cap_mw":      {k: round(v, 1) for k, v in mix["by_category"].items()},
        "new_cap_mw":      {k: round(v, 1) for k, v in mix["new_by_category"].items()},
        "retired_cap_mw":  {k: round(v, 1) for k, v in mix["retired_by_category"].items()},
        "total_endcap_mw": round(float(mix["total_endcap"]), 1),
    }


# ── Capacity mix color scheme ──────────────────────────────────────────────────
# Colors match the reference plot style
CATEGORY_COLORS = {
    "Coal":          "#4a4a4a",   # dark charcoal
    "Natural Gas":   "#6b8eae",   # steel blue
    "Solar":         "#f5d060",   # warm yellow
    "Wind":          "#b8a0d0",   # soft lavender
    "Batteries":     "#c0c0c0",   # light gray
    "Nuclear":       "#e8943a",   # orange
    "Hydro":         "#4a90d9",   # blue
    "Hydro Storage": "#2060a0",   # dark blue
    "Petroleum":     "#8b4513",   # brown
    "Biomass":       "#4a8a4a",   # green
    "Distributed":   "#a0c060",   # light green
    "Other":         "#999999",   # gray
}

# Display order (matches reference: firm dispatchable → variable → storage)
CATEGORY_ORDER = [
    "Coal", "Nuclear", "Petroleum", "Biomass",
    "Natural Gas",
    "Solar", "Wind", "Distributed",
    "Hydro", "Hydro Storage",
    "Batteries",
    "Other",
]


def plot_capacity_mix_vs_baseline(scenario_path: str) -> dict:
    """
    Plot net new capacity buildout for a scenario vs baseline, summed across
    both periods. Saves a bar chart and returns the file path.

    Bars show: scenario new builds - baseline new builds per resource category.
    Positive = more built in scenario than baseline, negative = less.
    """
    def net_capacity_change(path):
        m1 = _get_capacity_mix(path, 1, pjm_only=True)
        m2 = _get_capacity_mix(path, 2, pjm_only=True)
        new = m1["new_by_category"].add(m2["new_by_category"], fill_value=0)
        ret = m1["retired_by_category"].add(m2["retired_by_category"], fill_value=0)
        return new.subtract(ret, fill_value=0)

    scenario_new = net_capacity_change(scenario_path)
    baseline_new = net_capacity_change(BASELINE_PATH)

    diff = (scenario_new - baseline_new).fillna(0)

    # Filter to nonzero, preserve display order
    ordered = [c for c in CATEGORY_ORDER if c in diff.index and diff[c] != 0]
    diff = diff[ordered]

    if diff.empty:
        return {"message": "No difference in capacity mix vs baseline.", "plot_path": None}

    colors = [CATEGORY_COLORS.get(c, "#999999") for c in diff.index]

    fig, ax = plt.subplots(figsize=(max(6, len(diff) * 1.2), 5))
    bars = ax.bar(diff.index, diff.values, color=colors, width=0.6, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ Capacity (MW)")
    ax.set_title(
        f"{os.path.basename(scenario_path)}\nRelative to {os.path.basename(BASELINE_PATH)}",
        fontweight="bold"
    )

    # Value labels above/below each bar
    for bar, val in zip(bars, diff.values):
        offset = 8 if val >= 0 else -16
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + offset,
            f"{val:+.0f}",
            ha="center", va="bottom" if val >= 0 else "top",
            fontsize=9
        )

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    plots_dir = os.path.join(GENX_DIR, "mcp", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"{os.path.basename(scenario_path)}_capacity_mix.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "scenario":  os.path.basename(scenario_path),
        "baseline":  os.path.basename(BASELINE_PATH),
        "plot_path": out_path,
        "diff_mw":   {k: round(float(v), 1) for k, v in diff.items()},
    }


# ── Baseline reference ─────────────────────────────────────────────────────────

BASELINE_PATH = os.path.join(GENX_DIR, "scenarios", "PJM_Baseline_Example_copy")


# ── Case-name wrappers (used by MCP server) ────────────────────────────────────

def npv_for_case(case_name: str) -> dict:
    result = compute_npv(find_case(case_name))

    baseline = compute_npv(BASELINE_PATH)
    result["baseline_scenario"]  = baseline["scenario"]
    result["baseline_npv_B"]     = baseline["total_npv_B"]
    result["diff_vs_baseline_B"] = round(result["total_npv_B"] - baseline["total_npv_B"], 3)

    return result


def capacity_cost_for_case(case_name: str, period: int) -> dict:
    return compute_capacity_cost(find_case(case_name), period)


def capacity_mix_for_case(case_name: str, period: int) -> dict:
    return get_capacity_mix(find_case(case_name), period)


def capacity_mix_plot_for_case(case_name: str) -> dict:
    return plot_capacity_mix_vs_baseline(find_case(case_name))


METRIC_ALIASES = {
    "total cost":     "npv",
    "total system cost": "npv",
    "system cost":    "npv",
    "npv":            "npv",
    "capacity cost":  "capacity_cost",
    "capacity price": "capacity_cost",
}


def average_metric_across_scenarios(metric: str, transmission_mw: int, zones: str = "all") -> dict:
    """
    Compute average total system cost NPV across DC Island scenarios for a given
    transmission level and set of zones.

    Args:
        metric: Natural language metric name. Supported: "total cost", "total system cost",
                "npv", "capacity cost", "capacity price".
        transmission_mw: Transmission capacity in MW (e.g., 0, 250, 500, 750, 1000).
        zones: Comma-separated zone numbers (e.g., "10,13,14") or "all" for all PJM zones.

    Returns dict with per-scenario values, average, and any cases that were skipped.
    """
    metric_key = METRIC_ALIASES.get(metric.strip().lower())
    if metric_key is None:
        supported = list(METRIC_ALIASES.keys())
        raise ValueError(f"Unknown metric '{metric}'. Supported: {supported}")

    if zones.strip().lower() == "all":
        zone_list = PJM_ZONES
    else:
        zone_list = [int(z.strip()) for z in zones.split(",")]

    values = []
    skipped = []
    per_scenario = []

    baseline = compute_npv(BASELINE_PATH)

    for z in zone_list:
        case_name = f"PJM_DC_Island_{transmission_mw}MW_z{z}"
        candidate = os.path.join(GENX_DIR, "scenarios", f"z{z}", case_name)
        if not os.path.isdir(candidate):
            skipped.append(case_name)
            continue
        try:
            if metric_key == "npv":
                result = compute_npv(candidate)
                val = result["total_npv_B"]
                entry = {"case": case_name, "value_B": val, "includes_z28": result["includes_z28"]}
            elif metric_key == "capacity_cost":
                # Average capacity cost across both periods
                r1 = compute_capacity_cost(candidate, 1)
                r2 = compute_capacity_cost(candidate, 2)
                val = round((r1["price_per_mw_day"] + r2["price_per_mw_day"]) / 2, 2)
                entry = {"case": case_name, "value_per_mw_day": val,
                         "p1": r1["price_per_mw_day"], "p2": r2["price_per_mw_day"]}
            values.append(val)
            per_scenario.append(entry)
        except Exception as e:
            skipped.append(f"{case_name} (error: {e})")

    average = round(sum(values) / len(values), 3) if values else None

    if metric_key == "npv":
        baseline_val = baseline["total_npv_B"]
        diff = round(average - baseline_val, 3) if average is not None else None
        return {
            "metric":             "total_system_cost_npv",
            "transmission_mw":    transmission_mw,
            "zones_requested":    zone_list,
            "n_scenarios":        len(values),
            "average_B":          average,
            "baseline_npv_B":     baseline_val,
            "diff_vs_baseline_B": diff,
            "per_scenario":       per_scenario,
            "skipped":            skipped,
        }
    elif metric_key == "capacity_cost":
        return {
            "metric":           "capacity_cost_per_mw_day",
            "transmission_mw":  transmission_mw,
            "zones_requested":  zone_list,
            "n_scenarios":      len(values),
            "average_per_mw_day": average,
            "per_scenario":     per_scenario,
            "skipped":          skipped,
        }
