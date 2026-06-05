"""
GenX MCP Server

Exposes GenX cluster operations as MCP tools. Run with:
    python server.py
"""

from mcp.server.fastmcp import FastMCP
from typing import Optional

from tools.cases import list_cases as _list_cases
from tools.slurm import preview_case as _preview_case, submit_case as _submit_case
from tools.analytics import npv_for_case, capacity_cost_for_case, capacity_mix_for_case, average_metric_across_scenarios, capacity_mix_plot_for_case

mcp = FastMCP("genx")


@mcp.tool()
def list_cases(scenarios_dir: Optional[str] = None) -> list[dict]:
    """
    List all valid GenX cases and their status.

    Scans the default scenarios directories unless a specific path is given.
    Returns one entry per case with: name, path, has_results, last_modified,
    multi_stage, rep_periods (representative weeks), and num_stages.

    Args:
        scenarios_dir: Optional path to a specific directory to scan.
    """
    return _list_cases(scenarios_dir)


@mcp.tool()
def preview_genx_case(
    case_name: str,
    time_hours: Optional[int] = None,
    mem_gb: Optional[int] = None,
) -> dict:
    """
    Generate the SLURM submission script for a GenX case without submitting it.

    Looks up the case by name in the default scenario directories, infers
    appropriate walltime and memory from the case size (rep_periods × num_stages),
    and returns the script text along with the inferred and final resource values.

    Args:
        case_name:  Name of the case folder (e.g. "PJM_Baseline_Example_copy").
        time_hours: Override the inferred walltime in hours.
        mem_gb:     Override the inferred memory in GB.
    """
    return _preview_case(case_name, time_hours, mem_gb)


@mcp.tool()
def submit_genx_case(
    case_name: str,
    time_hours: Optional[int] = None,
    mem_gb: Optional[int] = None,
) -> dict:
    """
    Submit a GenX case to SLURM via sbatch.

    Looks up the case by name in the default scenario directories, infers
    walltime and memory from the case size, and submits the job. Returns
    the SLURM job ID and the resource values used.

    Args:
        case_name:  Name of the case folder (e.g. "PJM_Baseline_Example_copy").
        time_hours: Override the inferred walltime in hours.
        mem_gb:     Override the inferred memory in GB.
    """
    return _submit_case(case_name, time_hours, mem_gb)


@mcp.tool()
def get_scenario_npv(case_name: str) -> dict:
    """
    Compute total system cost NPV for a completed GenX scenario.

    Discounts each investment stage back to the P1 base year using WACC.
    Adds trailing CAPEX corrections for generation and transmission investments
    whose capital recovery period extends beyond the stage they were built in.

    Args:
        case_name: Name of the case folder (e.g. "PJM_Baseline_52wk").
    """
    return npv_for_case(case_name)


@mcp.tool()
def get_capacity_cost(case_name: str, period: int = 1) -> dict:
    """
    Compute PJM-wide capacity cost in $/MW-day for a completed GenX scenario.

    Follows the BRA-comparable protocol: total annual capacity cost across PJM
    CapRes regions (RTO Main, DOM, E-MAAC) normalized by PJM coincident peak
    demand, scaled to a daily value.

    Args:
        case_name: Name of the case folder (e.g. "PJM_Baseline_52wk").
        period:    Investment stage period (1 or 2). Default: 1.
    """
    return capacity_cost_for_case(case_name, period)


@mcp.tool()
def get_capacity_mix(case_name: str, period: int = 1) -> dict:
    """
    Get new build, retirement, and end capacity by resource category for a
    completed GenX scenario (PJM zones only).

    Resource categories include Solar, Wind, Gas CC, Gas CT, Nuclear, Battery,
    Coal, Hydro, and Other as classified by utils.classify_resource().

    Args:
        case_name: Name of the case folder (e.g. "PJM_Baseline_52wk").
        period:    Investment stage period (1 or 2). Default: 1.
    """
    return capacity_mix_for_case(case_name, period)


@mcp.tool()
def get_average_metric(metric: str, transmission_mw: int, zones: str = "all") -> dict:
    """
    Compute the average of a metric across DC Island scenarios for a given
    transmission level and set of zones.

    Supported metrics (natural language):
      - "total cost" / "total system cost" / "npv" → discounted total system cost NPV
      - "capacity cost" / "capacity price"          → $/MW-day averaged across P1 and P2

    Zone 28 costs are automatically included for DC Island scenarios.
    Skips zones where the scenario folder is missing or has no results.

    Args:
        metric:          Natural language metric name (e.g., "total cost").
        transmission_mw: Transmission capacity in MW (e.g., 0, 250, 500, 750, 1000).
        zones:           Comma-separated zone numbers (e.g., "10, 13, 14") or "all"
                         for all PJM zones. Default: "all".
    """
    return average_metric_across_scenarios(metric, transmission_mw, zones)


@mcp.tool()
def plot_capacity_mix(case_name: str) -> dict:
    """
    Plot net new capacity buildout for a scenario relative to the baseline
    (PJM_Baseline_Example_copy), summed across both investment periods.

    Bars show scenario new builds minus baseline new builds per resource
    category (MW). Positive = more built, negative = less built.
    Saves a PNG to mcp/plots/ and returns the file path.

    Args:
        case_name: Name of the case folder (e.g. "PJM_DC_Island_1000MW_z10").
    """
    return capacity_mix_plot_for_case(case_name)


if __name__ == "__main__":
    mcp.run()
