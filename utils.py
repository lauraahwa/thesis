"""
Shared utility functions for GenX analysis.
Used by both the Streamlit dashboard (app.py) and Jupyter notebooks.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# Configuration / Constants
# =============================================================================

# Base path for scenario folders. Set GENX_DIR in your environment (or .env);
# falls back to the current directory so imports don't fail when it's unset.
DEFAULT_BASE_PATH = Path(os.environ.get("GENX_DIR", ".")) / "scenarios"
DEFAULT_BASELINE_PATH = DEFAULT_BASE_PATH / 'PJM_Baseline_Example_copy'

# PJM zones (excluding neighboring regions)
PJM_ZONES = [1, 2, 4, 5, 6, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27]
EXCLUSIONS = [3, 8, 9, 11, 12, 21]

ZONE_NAMES = {
    1: 'DE1', 2: 'Il2', 3: 'ISONE', 4: 'KY3', 5: 'KY4', 6: 'MD1', 7: 'MD2',
    8: 'MI', 9: 'MISC', 10: 'NJ1', 11: 'NY1', 12: 'NY2', 13: 'OH1', 14: 'OH2',
    15: 'OH3', 16: 'OH4', 17: 'PA1', 18: 'PA2', 19: 'PA3', 20: 'PA4', 21: 'SERC',
    22: 'VA1', 23: 'VA2', 24: 'VA3', 25: 'VA4', 26: 'WV1', 27: 'WV2'
}

# CapRes region mapping
CAPRES_DICT = {
    1: [8, 9],
    2: [21],
    3: [2, 4, 5, 6, 13, 14, 15, 16, 17, 18, 22, 26, 27],
    4: [7, 23, 24, 25],
    5: [1, 10, 19, 20],
    6: [11, 12],
    7: [3]
}

CAPRES_NAMES = {
    1: 'MI/MISC', 2: 'SERC', 3: 'RTO (Main)', 4: 'DOM',
    5: 'E-MAAC', 6: 'NY', 7: 'ISONE'
}

# E-MAAC = Eastern mid-atlantic region (NJ, delaware, PA)

PJM_CAPRES = [3, 4, 5]  # Exclude MI/MISC(1), SERC(2), NY(6), and ISONE(7)


# =============================================================================
# System Cost Functions
# =============================================================================

def load_system_costs(folder_path, period, exclusions=None):
    """
    Load system-wide costs from costs.csv.

    Parameters:
    - folder_path: Path to the case folder (str or Path)
    - period: Period number (1 or 2)
    - exclusions: List of zone numbers to exclude from PJM total (default: EXCLUSIONS)

    Returns:
    - Dictionary with 'total', 'pjm_total', 'by_zone', and 'full_df'
    """
    if exclusions is None:
        exclusions = EXCLUSIONS

    folder_path = Path(folder_path)
    costs_df = pd.read_csv(
        folder_path / f'results/results_p{period}/costs.csv'
    ).set_index('Costs')

    ctotal = costs_df.loc['cTotal', 'Total']

    # PJM total = sum of PJM zone columns + network expansion costs
    pjm_zones_sum = np.array([
        costs_df.loc['cTotal', f'Zone{z}'] for z in PJM_ZONES
    ]).astype(np.float64).sum()

    network_exp = 0.0
    if 'cNetworkExp' in costs_df.index:
        network_exp = float(costs_df.loc['cNetworkExp', 'Total'])

    ctotal_pjm = pjm_zones_sum + network_exp

    return {
        'total': ctotal,
        'pjm_total': ctotal_pjm,
        'by_zone': {f'Zone{i}': float(costs_df.loc['cTotal', f'Zone{i}'])
                    for i in range(1, 28)},
        'full_df': costs_df
    }


# =============================================================================
# Capacity Cost Functions
# =============================================================================

def get_reserve_margins(baseline_path=None):
    """
    Get reserve margins for each CapRes region.

    Parameters:
    - baseline_path: Path to baseline folder (default: DEFAULT_BASELINE_PATH)

    Returns:
    - Dictionary mapping CapRes region names to margin values
    """
    if baseline_path is None:
        baseline_path = DEFAULT_BASELINE_PATH

    baseline_path = Path(baseline_path)
    capres_margin_df = pd.read_csv(
        baseline_path / 'inputs/inputs_p1/policies/Capacity_reserve_margin.csv'
    )
    capres_margins = {}
    for capres_num, zone_list in CAPRES_DICT.items():
        zone_row = capres_margin_df[capres_margin_df['Network_zones'] == f'z{zone_list[0]}']
        margin_val = zone_row[f'CapRes_{capres_num}'].values[0]
        capres_margins[f'CapRes_{capres_num}'] = margin_val
    return capres_margins


def aggregate_demand_by_capres(demand_df, capres_dict=None):
    """
    Aggregate demand data by CapRes regions.

    Parameters:
    - demand_df: DataFrame with columns Demand_MW_z1, Demand_MW_z2, ..., Demand_MW_z27
    - capres_dict: Dictionary mapping CapRes region number to list of zone numbers

    Returns:
    - DataFrame with columns CapRes_1, CapRes_2, ..., CapRes_7 containing summed demand
    """
    if capres_dict is None:
        capres_dict = CAPRES_DICT

    capres_demand = {}
    for capres_num, zone_list in capres_dict.items():
        zone_cols = [f'Demand_MW_z{z}' for z in zone_list]
        capres_demand[f'CapRes_{capres_num}'] = demand_df[zone_cols].sum(axis=1)
    return pd.DataFrame(capres_demand)


def get_peak_demand_per_capres(folder_path, period):
    """
    Get peak demand per CapRes region from full (non-TDR) demand data.

    For PJM regions (CapRes 3, 4, 5), uses **coincident peak** - the combined
    demand of all three regions at the hour when their sum is highest. This
    follows PJM's capacity market methodology where capacity obligations are
    allocated based on load during system-wide peak hours.

    For non-PJM regions (CapRes 1, 2, 6, 7), uses individual peak (non-coincident)
    since they are separate capacity markets.

    Parameters:
    - folder_path: Path to the case folder (str or Path)
    - period: Period number (1 or 2)

    Returns:
    - Series with peak demand (MW) for each CapRes region
    """
    folder_path = Path(folder_path)
    demand_full_df = pd.read_csv(
        folder_path / f'inputs/inputs_p{period}/system/Demand_data.csv'
    )
    capres_demand_full_df = aggregate_demand_by_capres(demand_full_df)

    # For non-PJM regions, use individual (non-coincident) peak
    peak_demand = capres_demand_full_df.max()

    # For PJM regions (CapRes 3, 4, 5), use coincident peak × (1 + reserve margin)
    # Find the hour when combined PJM demand is highest
    pjm_capres_cols = ['CapRes_3', 'CapRes_4', 'CapRes_5']
    pjm_combined = capres_demand_full_df[pjm_capres_cols].sum(axis=1)
    coincident_peak_hour = pjm_combined.idxmax()

    # PJM reserve margin is 17.7%
    PJM_RESERVE_MARGIN = 0.177

    # Get the total PJM coincident peak with reserve margin
    pjm_coincident_peak_with_rm = pjm_combined.loc[coincident_peak_hour] * (1 + PJM_RESERVE_MARGIN)

    # Store as 'PJM' entry for combined PJM price calculation
    peak_demand['PJM'] = pjm_coincident_peak_with_rm

    # For individual CapRes regions, use their demand at coincident peak hour × (1 + margin)
    for col in pjm_capres_cols:
        peak_demand[col] = capres_demand_full_df.loc[coincident_peak_hour, col] * (1 + PJM_RESERVE_MARGIN)

    return peak_demand


def capacity_breakdown(folder_path, period, baseline_path=None, exclude_non_pjm=True, stage_length=1):
    """
    Calculate capacity costs for a given folder and period.

    Parameters:
    - folder_path: Path to the case folder (str or Path)
    - period: Period number (1 or 2)
    - baseline_path: Path to baseline folder for reserve margins (default: DEFAULT_BASELINE_PATH)
    - exclude_non_pjm: If True, exclude SERC (CapRes_2) and ISONE (CapRes_7) from totals
    - stage_length: Number of years in the model stage (default: 5 for multi-stage GenX)

    Returns:
    - Dictionary with:
        - 'cost_by_capres': Total weighted cost per CapRes region ($/yr, annualized)
        - 'peak_demand': Peak demand per CapRes region (MW)
        - 'price_per_mw_yr': Capacity price per CapRes region ($/MW-yr)
        - 'price_per_mw_day': Capacity price per CapRes region ($/MW-day)
        - 'total_cost': System-wide total cost ($/yr)
        - 'total_cost_pjm': PJM-only total cost ($/yr), if exclude_non_pjm=True
    """
    folder_path = Path(folder_path)

    # Load ReserveMargin_w.csv (shadow prices)
    rm_df = pd.read_csv(folder_path / f'results/results_p{period}/ReserveMargin_w.csv')

    # Load TDR demand data
    demand_df = pd.read_csv(
        folder_path / f'inputs/inputs_p{period}/TDR_results/Demand_data.csv'
    )

    # Load time_weights.csv for proper per-timestep weights
    # This accounts for TDR representative period weighting
    time_weights_df = pd.read_csv(folder_path / f'results/results_p{period}/time_weights.csv')
    time_weights = time_weights_df['Weight'].values

    capres_margins = get_reserve_margins(baseline_path)
    capres_margins_s = pd.Series(capres_margins)
    capres_demand_df = aggregate_demand_by_capres(demand_df)

    rmw_capres = rm_df[[f'CapRes_{i}' for i in range(1, 8)]]

    # Calculate weighted capacity costs
    # Formula: cost = shadow_price × demand × (1+margin) × time_weight
    # The result is total cost over the stage (e.g., 5 years)
    capacity_cost_weighted = (
        rmw_capres.values * capres_demand_df.values * (1 + capres_margins_s.values) * time_weights.reshape(-1, 1)
    )
    total_stage_cost = capacity_cost_weighted.sum(axis=0)

    # Convert to annual cost by dividing by stage length
    total_annual_cost = pd.Series(total_stage_cost / stage_length, index=[f'CapRes_{i}' for i in range(1, 8)])

    peak_demand = get_peak_demand_per_capres(folder_path, period)

    # Calculate $/MW-yr for each CapRes region
    capres_cols = [f'CapRes_{i}' for i in range(1, 8)]
    price_per_mw_yr = total_annual_cost[capres_cols] / peak_demand[capres_cols]

    total_cost = total_annual_cost.sum()

    # PJM-only total (excluding MI/MISC(1), SERC(2), NY(6), ISONE(7))
    non_pjm_capres = ['CapRes_1', 'CapRes_2', 'CapRes_6', 'CapRes_7']
    pjm_capres = ['CapRes_3', 'CapRes_4', 'CapRes_5']
    total_cost_pjm = total_annual_cost[pjm_capres].sum() if exclude_non_pjm else None

    # Calculate combined PJM $/MW-yr using total PJM cost / (coincident peak × 1.177)
    pjm_price_per_mw_yr = total_cost_pjm / peak_demand['PJM'] if exclude_non_pjm else None

    return {
        'cost_by_capres': total_annual_cost,
        'peak_demand': peak_demand,
        'price_per_mw_yr': price_per_mw_yr,
        'price_per_mw_day': price_per_mw_yr / 365,
        'total_cost': total_cost,
        'total_cost_pjm': total_cost_pjm,
        'pjm_price_per_mw_yr': pjm_price_per_mw_yr,
        'pjm_price_per_mw_day': pjm_price_per_mw_yr / 365 if pjm_price_per_mw_yr else None
    }


# =============================================================================
# Energy Price Functions
# =============================================================================

def compute_avg_energy_price(folder_path, period, pjm_only=True):
    """
    Compute load-weighted average energy price ($/MWh) for ratepayers.

    This calculates the average price ratepayers pay per MWh of energy consumed,
    weighted by actual demand at each timestep and zone.

    Formula:
        $/MWh = Σ(Price_t,z * Demand_t,z * Weight_t) / Σ(Demand_t,z * Weight_t)

    Parameters:
    - folder_path: Path to the case folder (str or Path)
    - period: Period number (1 or 2)
    - pjm_only: If True, exclude non-PJM zones (3, 9, 11, 12, 21)

    Returns:
    - Dictionary with:
        - 'avg_price_system': System-wide load-weighted average $/MWh
        - 'avg_price_by_zone': Per-zone load-weighted average $/MWh (Series)
        - 'total_energy_cost': Total energy cost ($)
        - 'total_demand_mwh': Total energy consumed (MWh)
        - 'zone_costs': Total cost per zone ($) (Series)
        - 'zone_demand_mwh': Total demand per zone (MWh) (Series)
    """
    folder_path = Path(folder_path)

    # Load TDR demand data (time-reduced with Sub_Weights for scaling to full year)
    demand_df = pd.read_csv(
        folder_path / f'inputs/inputs_p{period}/TDR_results/Demand_data.csv'
    )

    # Load prices from results
    prices_df = pd.read_csv(
        folder_path / f'results/results_p{period}/prices.csv'
    )

    # Get zone columns (1-27)
    all_zones = list(range(1, 28))
    non_pjm_zones = [3, 9, 11, 12, 21]
    zones_to_use = [z for z in all_zones if z not in non_pjm_zones] if pjm_only else all_zones

    # Extract demand columns (Demand_MW_z1, ..., Demand_MW_z27)
    demand_cols = [f'Demand_MW_z{z}' for z in zones_to_use]
    demand_matrix = demand_df[demand_cols].values  # Shape: (T, Z)

    # Extract price columns (columns '1', '2', ..., '27' in prices.csv)
    price_cols = [str(z) for z in zones_to_use]
    price_matrix = prices_df[price_cols].values  # Shape: (T, Z)

    # Get Sub_Weights: N rep periods × 168 timesteps each
    # Only first row of each rep period has the weight, rest are NaN
    # Detect number of rep periods dynamically from data
    timesteps_per_period = 168
    total_timesteps = len(demand_df)
    n_rep_periods = total_timesteps // timesteps_per_period
    sub_weights_raw = demand_df['Sub_Weights'].values[:n_rep_periods]  # Shape: (n_rep_periods,)
    sub_weights = np.repeat(sub_weights_raw, timesteps_per_period)  # Shape: (total_timesteps,)

    # Hadamard product: element-wise Price * Demand * Weight
    # This gives cost in $ for each (time, zone) pair, scaled to annual
    cost_matrix = price_matrix * demand_matrix * sub_weights.reshape(-1, 1)

    # Weighted demand (for denominator)
    weighted_demand_matrix = demand_matrix * sub_weights.reshape(-1, 1)

    # System-wide totals
    total_energy_cost = cost_matrix.sum()  # Total $ spent on energy
    total_demand_mwh = weighted_demand_matrix.sum()  # Total MWh consumed
    avg_price_system = total_energy_cost / total_demand_mwh  # $/MWh

    # Per-zone calculation
    zone_costs = cost_matrix.sum(axis=0)  # Sum across time for each zone
    zone_demand = weighted_demand_matrix.sum(axis=0)  # Sum across time for each zone
    zone_avg_prices = zone_costs / zone_demand  # $/MWh per zone

    avg_price_by_zone = pd.Series(zone_avg_prices, index=[f'z{z}' for z in zones_to_use])

    return {
        'avg_price_system': avg_price_system,
        'avg_price_by_zone': avg_price_by_zone,
        'total_energy_cost': total_energy_cost,
        'total_demand_mwh': total_demand_mwh,
        'zone_costs': pd.Series(zone_costs, index=[f'z{z}' for z in zones_to_use]),
        'zone_demand_mwh': pd.Series(zone_demand, index=[f'z{z}' for z in zones_to_use])
    }


# =============================================================================
# Capacity Resource Mix Functions
# =============================================================================

# Resource category mappings based on resource name keywords
RESOURCE_CATEGORIES = {
    'Hydro Storage': ['hydroelectric_pumped_storage', 'hydro_storage'],
    'Hydro': ['hydroelectric', 'hydro'],
    'Natural Gas': ['natural_gas', 'naturalgas', '_cc', '_ct'],
    'Petroleum': ['petroleum'],
    'Nuclear': ['nuclear'],
    'Coal': ['coal'],
    'Solar': ['photovoltaic', 'utilitypv', 'solar_pv', 'solar'],
    'Wind': ['wind'],  # Covers landbasedwind, offshorewind, onshore_wind, offshore_wind
    'Distributed': ['distributed'],
    'Biomass': ['biomass'],
    'Batteries': ['batt']  # covers battery, batteries, utilityscale_battery_storage
}


def classify_resource(resource_name):
    """
    Classify a resource into a category based on its name.

    Parameters:
    - resource_name: Name of the resource (str)

    Returns:
    - Category name (str)
    """
    resource_lower = resource_name.lower()

    # Check Hydro Storage first (before general Hydro and Batteries)
    if 'hydro' in resource_lower and 'storage' in resource_lower:
        return 'Hydro Storage'

    # Check each category's keywords
    for category, keywords in RESOURCE_CATEGORIES.items():
        if category == 'Hydro Storage':
            continue  # Already handled above
        for keyword in keywords:
            if keyword in resource_lower:
                return category


def get_capacity_mix(folder_path, period, pjm_only=True, exclusions=None):
    """
    Get capacity mix by resource type for a given scenario.

    Parameters:
    - folder_path: Path to the case folder (str or Path)
    - period: Period number (1 or 2)
    - pjm_only: If True, exclude non-PJM zones
    - exclusions: List of zone numbers to exclude (default: EXCLUSIONS)

    Returns:
    - Dictionary with:
        - 'by_category': Series with EndCap by resource category (MW)
        - 'start_by_category': Series with StartCap by resource category (MW)
        - 'new_by_category': Series with NewCap by resource category (MW)
        - 'retired_by_category': Series with RetCap by resource category (MW)
        - 'total_endcap': Total end capacity (MW)
        - 'detail_df': Full DataFrame with resource-level details
    """
    if exclusions is None:
        exclusions = EXCLUSIONS

    folder_path = Path(folder_path)
    capacity_df = pd.read_csv(folder_path / f'results/results_p{period}/capacity.csv')

    # Remove the summary "Total" row
    capacity_df = capacity_df[capacity_df['Resource'] != 'Total']

    # Filter to PJM zones if requested
    if pjm_only:
        capacity_df = capacity_df[~capacity_df['Zone'].isin(exclusions)]

    # Classify each resource
    capacity_df['Category'] = capacity_df['Resource'].apply(classify_resource)

    # Aggregate by category
    by_category = capacity_df.groupby('Category')['EndCap'].sum()
    start_by_category = capacity_df.groupby('Category')['StartCap'].sum()
    new_by_category = capacity_df.groupby('Category')['NewCap'].sum()
    retired_by_category = capacity_df.groupby('Category')['RetCap'].sum()

    # Ensure all categories are present
    all_categories = list(RESOURCE_CATEGORIES.keys()) + ['Other']
    by_category = by_category.reindex(all_categories, fill_value=0)
    start_by_category = start_by_category.reindex(all_categories, fill_value=0)
    new_by_category = new_by_category.reindex(all_categories, fill_value=0)
    retired_by_category = retired_by_category.reindex(all_categories, fill_value=0)

    return {
        'by_category': by_category,
        'start_by_category': start_by_category,
        'new_by_category': new_by_category,
        'retired_by_category': retired_by_category,
        'total_endcap': by_category.sum(),
        'detail_df': capacity_df
    }


# =============================================================================
# Emissions Functions
# =============================================================================

def load_emissions(folder_path, period, pjm_only=True, exclusions=None):
    """
    Load CO2 emissions data from emissions.csv.

    The emissions.csv file contains:
    - Row "CO2_Price_*": Shadow price of CO2 in $/tonne (if CO2 caps exist)
    - Row "AnnualSum": Total annual emissions in tonnes CO2 per zone
    - Rows "t1", "t2", ...: Hourly emissions in tonnes CO2

    Parameters:
    - folder_path: Path to the case folder (str or Path)
    - period: Period number (1 or 2)
    - pjm_only: If True, exclude non-PJM zones from totals
    - exclusions: List of zone numbers to exclude (default: EXCLUSIONS)

    Returns:
    - Dictionary with:
        - 'total': System-wide total annual emissions (tonnes CO2)
        - 'pjm_total': PJM-only total annual emissions (tonnes CO2)
        - 'by_zone': Dict mapping zone to annual emissions (tonnes CO2)
        - 'co2_price': CO2 shadow price if available ($/tonne), else None
        - 'full_df': Full DataFrame
    """
    if exclusions is None:
        exclusions = EXCLUSIONS

    folder_path = Path(folder_path)
    emissions_df = pd.read_csv(
        folder_path / f'results/results_p{period}/emissions.csv'
    )

    # The first column is "Zone" which contains row labels
    emissions_df = emissions_df.set_index('Zone')

    # Get total (system-wide) from the "Total" column
    total = emissions_df.loc['AnnualSum', 'Total']

    # Get per-zone emissions
    zone_cols = [str(i) for i in range(1, 28)]
    by_zone = {}
    for z in range(1, 28):
        if str(z) in emissions_df.columns:
            by_zone[f'Zone{z}'] = emissions_df.loc['AnnualSum', str(z)]

    # Calculate PJM-only total (excluding non-PJM zones)
    pjm_total = total
    if pjm_only:
        for z in exclusions:
            if f'Zone{z}' in by_zone:
                pjm_total -= by_zone[f'Zone{z}']

    # Check for CO2 price rows
    co2_price = None
    for idx in emissions_df.index:
        if 'CO2_Price' in str(idx):
            # Get average non-zero price across zones
            price_row = emissions_df.loc[idx, zone_cols].astype(float)
            non_zero_prices = price_row[price_row > 0]
            if len(non_zero_prices) > 0:
                co2_price = non_zero_prices.mean()
            break

    return {
        'total': total,
        'pjm_total': pjm_total,
        'by_zone': by_zone,
        'co2_price': co2_price,
        'full_df': emissions_df
    }


# =============================================================================
# Energy Mix Functions
# =============================================================================

def get_energy_mix(folder_path, period, pjm_only=True, exclusions=None):
    """
    Get energy generation mix by resource type and new/existing status.

    Calculates the percentage of total generation from each resource category,
    separated by whether the capacity was existing or newly built.

    Parameters:
    - folder_path: Path to the case folder (str or Path)
    - period: Period number (1 or 2)
    - pjm_only: If True, exclude non-PJM zones
    - exclusions: List of zone numbers to exclude (default: EXCLUSIONS)

    Returns:
    - Dictionary with:
        - 'by_category': Series with MWh by resource category
        - 'by_category_pct': Same as above but as percentage of generation
        - 'by_new_existing': DataFrame with MWh split by category + Existing/New Build
        - 'by_new_existing_pct': Same as above but as percentages
        - 'total_generation_mwh': Total generation (excluding storage)
        - 'storage_summary': Dict with storage discharge, charge, and round-trip losses
        - 'transmission_losses_mwh': Total transmission losses
        - 'loss_percentages': Dict with storage and transmission losses as % of load
        - 'detail_df': Resource-level breakdown DataFrame
    """
    if exclusions is None:
        exclusions = EXCLUSIONS

    folder_path = Path(folder_path)
    results_path = folder_path / f'results/results_p{period}'
    inputs_path = folder_path / f'inputs/inputs_p{period}'

    # -------------------------------------------------------------------------
    # 1. Load power.csv (transposed format: resources as columns)
    # -------------------------------------------------------------------------
    power_df = pd.read_csv(results_path / 'power.csv')

    # Extract AnnualSum row (MWh since 1h timesteps)
    annual_row = power_df[power_df['Resource'] == 'AnnualSum'].iloc[0]
    zone_row = power_df[power_df['Resource'] == 'Zone'].iloc[0]

    # Build resource DataFrame (skip 'Resource' column and 'Total' if present)
    resource_cols = [c for c in power_df.columns if c not in ['Resource', 'Total']]

    resources_df = pd.DataFrame({
        'Resource': resource_cols,
        'Generation_MWh': [annual_row[c] for c in resource_cols],
        'Zone': [int(zone_row[c]) for c in resource_cols]
    })

    # -------------------------------------------------------------------------
    # 2. Load capacity.csv to determine what was actually built (NewCap > 0)
    # -------------------------------------------------------------------------
    capacity_df = pd.read_csv(results_path / 'capacity.csv')
    capacity_df = capacity_df[capacity_df['Resource'] != 'Total']

    # Create mapping: resource -> NewCap value
    newcap_map = dict(zip(capacity_df['Resource'], capacity_df['NewCap']))

    # -------------------------------------------------------------------------
    # 3. Load resource input files to get New_Build flag (candidate status)
    # -------------------------------------------------------------------------
    new_build_map = {}
    resource_files = ['Thermal.csv', 'Vre.csv', 'Storage.csv', 'Hydro.csv', 'Must_run.csv']

    for resource_file in resource_files:
        file_path = inputs_path / 'resources' / resource_file
        if file_path.exists():
            df = pd.read_csv(file_path)
            if 'New_Build' in df.columns:
                for _, row in df.iterrows():
                    new_build_map[row['Resource']] = int(row['New_Build'])

    # -------------------------------------------------------------------------
    # 4. Filter to PJM zones if requested
    # -------------------------------------------------------------------------
    if pjm_only:
        resources_df = resources_df[~resources_df['Zone'].isin(exclusions)].copy()

    # -------------------------------------------------------------------------
    # 5. Add classification columns
    # -------------------------------------------------------------------------
    resources_df['Category'] = resources_df['Resource'].apply(classify_resource)
    resources_df['Is_New_Build_Candidate'] = resources_df['Resource'].map(new_build_map).fillna(0).astype(int)
    resources_df['NewCap'] = resources_df['Resource'].map(newcap_map).fillna(0)

    # A resource is "New Build" if it was a candidate AND actually built capacity
    resources_df['Was_Built'] = (
        (resources_df['Is_New_Build_Candidate'] == 1) &
        (resources_df['NewCap'] > 0)
    ).astype(int)
    resources_df['Build_Type'] = resources_df['Was_Built'].map({0: 'Existing', 1: 'New Build'})

    # -------------------------------------------------------------------------
    # 6. Separate storage from generation
    # -------------------------------------------------------------------------
    storage_categories = ['Batteries', 'Hydro Storage']
    gen_df = resources_df[~resources_df['Category'].isin(storage_categories)].copy()
    storage_df = resources_df[resources_df['Category'].isin(storage_categories)].copy()

    # Only count positive generation (some generators might have negative if curtailed)
    gen_df['Positive_Gen_MWh'] = gen_df['Generation_MWh'].clip(lower=0)

    # -------------------------------------------------------------------------
    # 7. Aggregate generation by category
    # -------------------------------------------------------------------------
    by_category = gen_df.groupby('Category')['Positive_Gen_MWh'].sum().sort_values(ascending=False)
    total_generation = by_category.sum()
    by_category_pct = (by_category / total_generation * 100).round(2)

    # -------------------------------------------------------------------------
    # 8. Aggregate by category + build type
    # -------------------------------------------------------------------------
    by_new_existing = gen_df.pivot_table(
        values='Positive_Gen_MWh',
        index='Category',
        columns='Build_Type',
        aggfunc='sum',
        fill_value=0
    )

    # Ensure both columns exist
    for col in ['Existing', 'New Build']:
        if col not in by_new_existing.columns:
            by_new_existing[col] = 0
    by_new_existing = by_new_existing[['Existing', 'New Build']]

    # Add total column and sort
    by_new_existing['Total'] = by_new_existing['Existing'] + by_new_existing['New Build']
    by_new_existing = by_new_existing.sort_values('Total', ascending=False)

    # Calculate percentages
    by_new_existing_pct = by_new_existing.copy()
    for col in ['Existing', 'New Build', 'Total']:
        by_new_existing_pct[col] = (by_new_existing[col] / total_generation * 100).round(2)

    # -------------------------------------------------------------------------
    # 9. Calculate storage losses (charge - discharge = round-trip loss)
    # -------------------------------------------------------------------------
    # Load charge data
    charge_df = pd.read_csv(results_path / 'charge.csv')
    charge_annual = charge_df[charge_df['Resource'] == 'AnnualSum'].iloc[0]

    # Get storage resources that are in PJM zones
    storage_resources = storage_df['Resource'].tolist()

    # Sum discharge (from power.csv) and charge
    total_discharge = storage_df['Generation_MWh'].sum()

    total_charge = 0
    for res in storage_resources:
        if res in charge_annual.index:
            total_charge += charge_annual[res]

    # Round-trip loss = charge - discharge (energy lost in storage cycle)
    storage_loss = total_charge - total_discharge

    storage_summary = {
        'discharge_mwh': total_discharge,
        'charge_mwh': total_charge,
        'roundtrip_loss_mwh': storage_loss
    }

    # -------------------------------------------------------------------------
    # 10. Calculate transmission losses
    # -------------------------------------------------------------------------
    tlosses_df = pd.read_csv(results_path / 'tlosses.csv')
    tlosses_annual = tlosses_df[tlosses_df['Line'] == 'AnnualSum'].iloc[0]
    transmission_losses = tlosses_annual['Total']

    # -------------------------------------------------------------------------
    # 11. Calculate loss percentages relative to total load
    # -------------------------------------------------------------------------
    # Total load = generation - transmission losses - storage losses
    # Or alternatively: generation = load + transmission_losses + storage_losses
    # So load = generation - losses
    total_load = total_generation - transmission_losses - storage_loss

    loss_percentages = {
        'storage_loss_pct': round(storage_loss / total_load * 100, 2) if total_load > 0 else 0,
        'transmission_loss_pct': round(transmission_losses / total_load * 100, 2) if total_load > 0 else 0,
        'total_loss_pct': round((storage_loss + transmission_losses) / total_load * 100, 2) if total_load > 0 else 0
    }

    return {
        'by_category': by_category,
        'by_category_pct': by_category_pct,
        'by_new_existing': by_new_existing,
        'by_new_existing_pct': by_new_existing_pct,
        'total_generation_mwh': total_generation,
        'total_load_mwh': total_load,
        'storage_summary': storage_summary,
        'transmission_losses_mwh': transmission_losses,
        'loss_percentages': loss_percentages,
        'detail_df': resources_df
    }


def print_energy_mix_summary(energy_mix):
    """
    Print a formatted summary of the energy mix results.

    Parameters:
    - energy_mix: Dictionary returned by get_energy_mix()
    """
    print("=" * 70)
    print("ENERGY GENERATION MIX SUMMARY")
    print("=" * 70)

    print(f"\nTotal Generation: {energy_mix['total_generation_mwh']:,.0f} MWh")
    print(f"Total Load: {energy_mix['total_load_mwh']:,.0f} MWh")

    print("\n" + "-" * 70)
    print("GENERATION BY RESOURCE CATEGORY")
    print("-" * 70)
    print(f"{'Category':<20} {'Generation (MWh)':>20} {'Percentage':>15}")
    print("-" * 70)
    for cat in energy_mix['by_category'].index:
        gen = energy_mix['by_category'][cat]
        pct = energy_mix['by_category_pct'][cat]
        print(f"{cat:<20} {gen:>20,.0f} {pct:>14.2f}%")

    print("\n" + "-" * 70)
    print("GENERATION BY CATEGORY AND BUILD TYPE")
    print("-" * 70)
    df = energy_mix['by_new_existing']
    pct_df = energy_mix['by_new_existing_pct']
    print(f"{'Category':<20} {'Existing (MWh)':>18} {'New Build (MWh)':>18} {'Total (MWh)':>15}")
    print("-" * 70)
    for cat in df.index:
        existing = df.loc[cat, 'Existing']
        new = df.loc[cat, 'New Build']
        total = df.loc[cat, 'Total']
        print(f"{cat:<20} {existing:>18,.0f} {new:>18,.0f} {total:>15,.0f}")

    # Totals row
    print("-" * 70)
    total_existing = df['Existing'].sum()
    total_new = df['New Build'].sum()
    total_all = df['Total'].sum()
    print(f"{'TOTAL':<20} {total_existing:>18,.0f} {total_new:>18,.0f} {total_all:>15,.0f}")
    print(f"{'(Percentage)':<20} {total_existing/total_all*100:>17.2f}% {total_new/total_all*100:>17.2f}%")

    print("\n" + "-" * 70)
    print("STORAGE SUMMARY")
    print("-" * 70)
    ss = energy_mix['storage_summary']
    print(f"Storage Discharge: {ss['discharge_mwh']:>20,.0f} MWh")
    print(f"Storage Charge:    {ss['charge_mwh']:>20,.0f} MWh")
    print(f"Round-trip Loss:   {ss['roundtrip_loss_mwh']:>20,.0f} MWh")

    print("\n" + "-" * 70)
    print("SYSTEM LOSSES (as % of load)")
    print("-" * 70)
    lp = energy_mix['loss_percentages']
    print(f"Transmission Losses: {energy_mix['transmission_losses_mwh']:>15,.0f} MWh ({lp['transmission_loss_pct']:.2f}%)")
    print(f"Storage Losses:      {ss['roundtrip_loss_mwh']:>15,.0f} MWh ({lp['storage_loss_pct']:.2f}%)")
    print(f"Total Losses:        {energy_mix['transmission_losses_mwh'] + ss['roundtrip_loss_mwh']:>15,.0f} MWh ({lp['total_loss_pct']:.2f}%)")
    print("=" * 70)


def get_energy_mix_dataframe(folder_path, period, pjm_only=True, exclusions=None):
    """
    Get energy mix as a flat DataFrame suitable for Streamlit/dashboard display.

    Returns a DataFrame with columns:
    - Category: Resource category
    - Existing_MWh: Generation from existing resources
    - NewBuild_MWh: Generation from newly built resources
    - Total_MWh: Total generation
    - Existing_Pct: Percentage from existing
    - NewBuild_Pct: Percentage from new build
    - Total_Pct: Percentage of total generation
    """
    result = get_energy_mix(folder_path, period, pjm_only, exclusions)

    df = result['by_new_existing'].copy()
    total_gen = result['total_generation_mwh']

    df = df.reset_index()
    df.columns = ['Category', 'Existing_MWh', 'NewBuild_MWh', 'Total_MWh']

    df['Existing_Pct'] = (df['Existing_MWh'] / total_gen * 100).round(2)
    df['NewBuild_Pct'] = (df['NewBuild_MWh'] / total_gen * 100).round(2)
    df['Total_Pct'] = (df['Total_MWh'] / total_gen * 100).round(2)

    return df


def get_energy_mix_summary_dict(folder_path, period, pjm_only=True, exclusions=None):
    """
    Get energy mix summary as a simple dictionary for easy JSON serialization.

    Useful for exporting to Streamlit or comparing across scenarios.

    Returns a dictionary with:
    - total_generation_mwh: Total generation
    - total_load_mwh: Total load (after losses)
    - generation_by_category: Dict of category -> MWh
    - generation_by_category_pct: Dict of category -> percentage
    - existing_vs_new: Dict with 'existing_mwh', 'existing_pct', 'new_mwh', 'new_pct'
    - storage: Dict with discharge, charge, loss
    - transmission_losses_mwh: Transmission losses
    - loss_percentages: Dict with storage_loss_pct, transmission_loss_pct
    """
    result = get_energy_mix(folder_path, period, pjm_only, exclusions)

    # Calculate existing vs new totals
    df = result['by_new_existing']
    total_existing = df['Existing'].sum()
    total_new = df['New Build'].sum()
    total_gen = result['total_generation_mwh']

    return {
        'total_generation_mwh': total_gen,
        'total_load_mwh': result['total_load_mwh'],
        'generation_by_category': result['by_category'].to_dict(),
        'generation_by_category_pct': result['by_category_pct'].to_dict(),
        'existing_vs_new': {
            'existing_mwh': total_existing,
            'existing_pct': round(total_existing / total_gen * 100, 2),
            'new_mwh': total_new,
            'new_pct': round(total_new / total_gen * 100, 2)
        },
        'storage': result['storage_summary'],
        'transmission_losses_mwh': result['transmission_losses_mwh'],
        'loss_percentages': result['loss_percentages']
    }
