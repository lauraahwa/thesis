"""
PJM-wide capacity cost ($/MW-day) for a completed GenX scenario period.

Reconstructs the total annual value of the binding capacity reserve margin
constraints, matching the GenX formulation (cap_reserve_margin.jl):

    cCapacityResMargin[res,t]:
        eCapResMarBalance >= sum_z pD[t,z] * (1 + dfCapRes[z,res])

so the per-zone reserve-margin uplift (1 + dfCapRes) is applied. The dual is
read from ReserveMargin_w.csv (= raw dual / omega) and multiplied back by omega:

    TotalAnnualCapacityCost = sum_res sum_t lambda[t,res] * omega_t
                              * sum_z pD[t,z] * (1 + dfCapRes[z,res])
    Price_annual = TotalAnnualCapacityCost / max_t( sum_{z in PJM} D[t,z] )
    Price_day    = Price_annual / 365

Self-contained: depends only on os, numpy, and pandas.
"""

import os

import numpy as np
import pandas as pd

# Included CapRes regions: PJM only (3=RTO Main, 4=DOM, 5=E-MAAC). The DC island
# (CapRes_8 / z28) is excluded from both the numerator and the peak-demand
# denominator.
PJM_CAPRES = [3, 4, 5]

# 21 PJM zones (excludes border zones and the DC Island zone 28).
PJM_ZONES = [1, 2, 4, 5, 6, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20,
             22, 23, 24, 25, 26, 27]


def resolve_scenario(scenario_path: str, period: int) -> str:
    """
    Resolve `scenario_path` to an absolute scenario directory containing
    results/results_p{period}/ReserveMargin_w.csv.

    Accepts an absolute path, a path relative to GENX_DIR, or a path relative
    to the current working directory. Returns the first candidate containing
    the marker file, falling back to the absolute path (which will fail
    downstream if it's wrong).
    """
    marker = os.path.join("results", f"results_p{period}", "ReserveMargin_w.csv")
    expanded = os.path.expanduser(scenario_path)
    genx_dir = os.environ.get("GENX_DIR")
    candidates = [expanded] if os.path.isabs(expanded) else [os.path.abspath(expanded)]
    if genx_dir and not os.path.isabs(expanded):
        candidates.insert(0, os.path.join(genx_dir, expanded))

    for cand in candidates:
        if os.path.isfile(os.path.join(cand, marker)):
            return cand
    return candidates[0]


def compute_capacity_cost(scenario_path: str, period: int = 1) -> dict:
    """
    Compute PJM-wide capacity cost ($/MW-day) for a given scenario and period.

    Zone membership and per-zone RM are read from Capacity_reserve_margin.csv
    (the constraint's own data), so heterogeneous reserve margins are handled
    exactly.
    """
    scenario = resolve_scenario(scenario_path, period)

    dem_path    = os.path.join(scenario, "inputs", f"inputs_p{period}", "TDR_results", "Demand_data.csv")
    resmar_path = os.path.join(scenario, "results", f"results_p{period}", "ReserveMargin_w.csv")
    capres_path = os.path.join(scenario, "inputs", f"inputs_p{period}", "policies", "Capacity_reserve_margin.csv")

    for path in (dem_path, resmar_path, capres_path):
        if not os.path.isfile(path):
            return {"success": False,
                    "message": f"Missing required file: {path}"}

    dem_in = pd.read_csv(dem_path)
    resmar = pd.read_csv(resmar_path)
    capres = pd.read_csv(capres_path).set_index("Network_zones")  # index 'z1'..; cols CapRes_*

    # omega weights: read hours-per-rep-period from the file rather than assuming 168
    hours_per_period = int(dem_in["Timesteps_per_Rep_Period"].dropna().iloc[0])
    weights          = dem_in["Sub_Weights"].dropna().values
    hourly_weights   = np.array([w / hours_per_period for w in weights for _ in range(hours_per_period)])

    total_cost = 0.0
    for capres_num in PJM_CAPRES:
        capres_col = f"CapRes_{capres_num}"
        if capres_col not in resmar.columns or capres_col not in capres.columns:
            continue
        # Member zones = those with a nonzero requirement (exactly the constraint's zone set)
        members = capres[capres[capres_col] != 0][capres_col]
        if members.empty:
            continue
        # sum_z D[t,z] * (1 + RM_z)  -- per-zone reserve-margin uplift
        regional_demand = sum(
            dem_in[f"Demand_MW_z{int(zlabel[1:])}"].values * (1.0 + rm_z)
            for zlabel, rm_z in members.items()
        )
        lambda_t    = resmar[capres_col].values
        total_cost += (lambda_t * regional_demand * hourly_weights).sum()

    denom_zones = [z for z in PJM_ZONES if f"Demand_MW_z{z}" in dem_in.columns]
    peak_demand = dem_in[[f"Demand_MW_z{z}" for z in denom_zones]].sum(axis=1).values.max()
    price_annual = total_cost / peak_demand
    price_day    = price_annual / 365

    return {
        "success":            True,
        "scenario":           os.path.basename(scenario),
        "scenario_path":      scenario,
        "period":             period,
        "price_per_mw_day":   round(float(price_day), 2),
        "price_per_mw_yr":    round(float(price_annual), 2),
        "pjm_peak_demand_mw": round(float(peak_demand), 1),
    }
