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


if __name__ == "__main__":
    mcp.run()
