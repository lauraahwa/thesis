# genx-tested-agent MCP server.
#
# Loads configuration from a .env file (see .env.example) before importing
# anything that reads environment variables, so personal/cluster settings are
# never hardcoded.

from pathlib import Path
from dotenv import load_dotenv

# Load .env sitting next to this file (no-op if it doesn't exist).
load_dotenv(Path(__file__).resolve().parent / ".env")

from typing import Optional

from mcp.server.fastmcp import FastMCP

# Analytics tools
from plot_capacity import (
    resource_colors,
    column_titles,
    classify_resource,
    load_capacity_csv,
    filter_by_zones,
    check_existing,
    aggregate_capacity_by_resource,
    plot_capacity_bar,
)

# SLURM submission (imports os.environ at module load, hence after load_dotenv).
from slurm import preview_case as _preview_case, submit_case as _submit_case

# Two-period NPV of a costs.csv row over a zone set.
from npv_costs import compute_cost_npv as _compute_cost_npv, ZONE_SETS, DEFAULT_WACC

# Weighted average-day (diurnal) generation stacks; wraps <GENX_DIR>/diurnal_generation.py.
from diurnal import plot_diurnal_generation as _plot_diurnal_generation

# PJM-wide capacity cost ($/MW-day) from ReserveMargin_w duals.
from capacity_cost import compute_capacity_cost as _compute_capacity_cost

mcp = FastMCP("genx-tested-agent")


@mcp.tool()
def check_capacity_setting(csv_path: str) -> dict:
    """Detect whether a GenX capacity.csv is a brownfield or greenfield case."""
    df = load_capacity_csv(csv_path)
    # check_existing: whether StartCap > 0 (brownfield) or all StartCap = 0 (greenfield)
    return check_existing(df)


@mcp.tool()
def summarize_capacity(csv_path: str, zones: list[int] | None = None) -> dict:
    """
    Aggregate StartCap, RetCap, NewCap, EndCap, and NetCap by resource type.

    Args:
        csv_path: Path to the capacity.csv file
        zones: Optional list of zone numbers (e.g., [2, 5, 7, 9]) to filter to
        before aggregating. Omit to aggregate over all zones.
    """
    df = load_capacity_csv(csv_path)
    if zones:
        try:
            df = filter_by_zones(df, zones)
        except ValueError as e:
            return {"success": False, "message": str(e)}
    return aggregate_capacity_by_resource(df)


@mcp.tool()
def plot_capacity(
    csv_path: str,
    output_dir: str,
    plot_type: str,
    scenario_name: str,
    period: str,
    zones: list[int] | None = None
) -> dict:
    """
    Plot capacity data and save to PNG file.

    Before calling this tool, ask the user:
    1. Whether they want to specify zones to aggregate over. If they don't
       have specific zones, tell them the default is all zones in the
       capacity.csv file and omit the zones argument.
    2. The scenario name and the period, which go in the plot title.

    Args:
        csv_path: Path to the capacity.csv file
        output_dir: Directory to save the plot PNG files
        plot_type: Type of plot - one of: "StartCap", "RetCap", "NewCap", "EndCap", "NetCap"
        scenario_name: Scenario name for the plot title (ask the user)
        period: Period for the plot title, e.g. "1" (ask the user)
        zones: Optional list of zone numbers (e.g., [2, 5, 7, 9]) to filter to
        before aggregating. Omit to plot all zones.

    Returns:
        dict with success status, message, and file path
    """
    valid_types = ["StartCap", "RetCap", "NewCap", "EndCap", "NetCap"]

    if plot_type not in valid_types:
        return {
            "success": False,
            "message": f"Invalid plot_type '{plot_type}'. Must be one of: {valid_types}",
            "file_path": None
        }

    df = load_capacity_csv(csv_path)
    if zones:
        try:
            df = filter_by_zones(df, zones)
        except ValueError as e:
            return {
                "success": False,
                "message": str(e),
                "file_path": None
            }
    aggregated = aggregate_capacity_by_resource(df)

    setting_info = check_existing(df)
    is_brownfield = setting_info["is_brownfield"]

    if not is_brownfield:
        if plot_type in ["StartCap", "RetCap"]:
            return {
                "success": False,
                "message": f"Not meaningful to plot {plot_type} in greenfield case (all StartCap = 0). NewCap = EndCap = NetCap.",
                "setting": "greenfield"
            }
        elif plot_type == "NetCap":
            plot_type = "EndCap"
            message_suffix = " Note that NewCap = EndCap = NetCap in greenfield case!"
        else:
            message_suffix = ""
    else:
        message_suffix = ""

    output_path = Path(output_dir) / f"{plot_type}.png"
    title = f"{column_titles[plot_type]} {scenario_name} Period {period}"

    result = plot_capacity_bar(
        df=aggregated,
        capacity_column=plot_type,
        output_path=output_path,
        title=title
    )

    if result["success"]:
        result["message"] += message_suffix
        result["setting"] = "greenfield" if not is_brownfield else "brownfield"

    return result


@mcp.tool()
def plot_diurnal_generation(
    case_dir: str,
    output_path: str,
    period: int,
    zones: str,
    labels: str,
    compare_case_dir: Optional[str] = None,
    diff: bool = False,
) -> dict:
    """
    Stacked area chart of average generation by hour of day (0-23), by
    technology group, from a GenX case's results_pN/power.csv. Each model hour
    is weighted by its TDR time weight, so the profile matches the full-year
    average day rather than over-weighting peak representative weeks.

    With only `case_dir`, plots a single stacked chart. With `compare_case_dir`,
    plots side-by-side stacks (or, with diff=True, a line plot of the per-tech
    difference compare - case by hour of day). Comparison is two-way only.

    IMPORTANT — `output_path`, `period`, `zones`, and `labels` must come from
    the user. If the user has not stated them, ask before calling this tool
    rather than guessing (for zones, mention the choices below; "pjm" is the
    usual default).

    Args:
        case_dir: Case folder containing results/results_p{period}/power.csv.
            May be absolute, relative to GENX_DIR, or relative to the cwd.
        output_path: Path for the output PNG (parent dirs are created).
            Ask the user.
        period: Model period N (results_pN), e.g. 1. Ask the user.
        zones: "pjm" (excludes border zones), "island" (DC Island z28), "all",
            or comma-separated zone numbers, e.g. "10,23". Ask the user.
        labels: Comma-separated panel/legend labels, e.g. "Original,DR" (one
            label if plotting a single case). Ask the user.
        compare_case_dir: Optional second case to compare against.
        diff: If True (requires compare_case_dir), plot the per-technology
            difference instead of side-by-side stacks.

    Returns:
        dict with success status, message, file_path, resolved case dirs, and
        the technology groups found.
    """
    return _plot_diurnal_generation(
        case_dir=case_dir,
        output_path=output_path,
        period=period,
        zones=zones,
        labels=labels,
        compare_case_dir=compare_case_dir,
        diff=diff,
    )


@mcp.tool()
def preview_genx_case(
    case_dir: str,
    time_hours: int,
    mem_gb: int,
    cpus: Optional[int] = None,
    case_name: Optional[str] = None,
) -> dict:
    """
    Generate the SLURM submission script for a GenX case without submitting it.

    Resolves the case from the given directory path and returns the script text
    along with the resource values used. Walltime and memory must be supplied by
    the caller; if the user has not stated them, ask before calling this tool.

    Args:
        case_dir:   Path to the case folder. May be absolute, relative to
                    GENX_DIR, or relative to the working directory.
        time_hours: Walltime in hours (required).
        mem_gb:     Memory in GB (required).
        cpus:       Number of CPUs. Defaults to SLURM_CPUS_DEFAULT.
        case_name:  Optional SLURM job name / label. Defaults to the directory basename.
    """
    return _preview_case(case_dir, time_hours, mem_gb, cpus, case_name)


@mcp.tool()
def submit_genx_case(
    case_dir: str,
    time_hours: int,
    mem_gb: int,
    cpus: Optional[int] = None,
    case_name: Optional[str] = None,
) -> dict:
    """
    Submit a GenX case to SLURM via sbatch.

    Resolves the case from the given directory path and submits the job. Returns
    the SLURM job ID and the resource values used. Walltime and memory must be
    supplied by the caller; if the user has not stated them, ask before calling
    this tool.

    Args:
        case_dir:   Path to the case folder. May be absolute, relative to
                    GENX_DIR, or relative to the working directory.
        time_hours: Walltime in hours (required).
        mem_gb:     Memory in GB (required).
        cpus:       Number of CPUs. Defaults to SLURM_CPUS_DEFAULT.
        case_name:  Optional SLURM job name / label. Defaults to the directory basename.
    """
    return _submit_case(case_dir, time_hours, mem_gb, cpus, case_name)


@mcp.tool()
def compute_cost_npv(
    scenario_path: str,
    cost_row: str = "cTotal",
    zone_set: str = "PJM",
    zones: list[int] | None = None,
    wacc: float = DEFAULT_WACC,
    stage_length: int | None = None,
) -> dict:
    """
    Sum a costs.csv cost row over a set of zones in each period, then compute the
    two-period NPV by discounting P2 back to P1:

        NPV = value_p1 + value_p2 / (1 + wacc) ** stage_length

    P1 is the base year (undiscounted). `stage_length` is the number of years
    between the two stages; if omitted it is read from the scenario's
    multi_stage_settings.yml (StageLengths[0]).

    IMPORTANT — before calling this tool, confirm the discount rate with the user:
    the default WACC is 0.045 (4.5%). Ask whether they are happy with 0.045 or
    want a different value, and pass their choice as `wacc`.

    Args:
        scenario_path: Scenario folder containing results/results_p1/costs.csv,
            results/results_p2/costs.csv, and settings/multi_stage_settings.yml.
            May be absolute, relative to GENX_DIR, or relative to the cwd.
        cost_row: Which costs.csv row to sum, e.g. "cTotal", "cFix", "cVar",
            "cFuel", "cStart", "cNSE". Defaults to "cTotal".
        zone_set: Named zone selection: one of "PJM" (21 PJM zones, EXCLUDES the
            DC Island zone 28), "NON_PJM", "ALL" (27 model zones, also excludes
            28), or "DC_ISLAND" (just zone 28). Defaults to "PJM". Ignored if
            `zones` is given.
        zones: Explicit list of zone numbers to sum over, e.g. [1, 2, 23, 28].
            Overrides `zone_set` when provided — use this to include zone 28.
        wacc: Discount rate. Defaults to 0.045. Confirm with the user first.
        stage_length: Years between stages for the P2 discount exponent. Omit to
            read StageLengths[0] from the scenario settings.

    Returns:
        dict with value_p1, value_p2, value_p2_discounted, the discount factor,
        and the combined npv. If `wacc` differs from the scenario's settings
        WACC, `wacc_note` flags it.
    """
    return _compute_cost_npv(
        scenario_path=scenario_path,
        cost_row=cost_row,
        zone_set=zone_set,
        zones=zones,
        wacc=wacc,
        stage_length=stage_length,
    )


@mcp.tool()
def get_capacity_cost(scenario_path: str, period: int = 1) -> dict:
    """
    Compute PJM-wide capacity cost in $/MW-day for a completed GenX scenario.

    Follows the BRA-comparable protocol: reconstructs the total annual value of
    the binding capacity reserve margin constraints across the PJM CapRes
    regions (3=RTO Main, 4=DOM, 5=E-MAAC) from the ReserveMargin_w.csv duals,
    then normalizes by the PJM coincident peak demand and scales to a daily
    value. The DC Island (CapRes_8 / zone 28) is excluded from both the
    numerator and the peak-demand denominator.

    Args:
        scenario_path: Scenario folder containing
            results/results_p{period}/ReserveMargin_w.csv and
            inputs/inputs_p{period}/. May be absolute, relative to GENX_DIR,
            or relative to the cwd.
        period: Investment stage period (1 or 2). Default: 1.

    Returns:
        dict with price_per_mw_day, price_per_mw_yr, pjm_peak_demand_mw, and
        the resolved scenario path.
    """
    return _compute_capacity_cost(scenario_path, period)


if __name__ == "__main__":
    mcp.run()
