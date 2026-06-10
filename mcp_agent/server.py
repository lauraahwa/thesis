# Import analytics tools

from plot_capacity import(
    resource_colors,
    column_titles,
    classify_resource,
    load_capacity_csv,
    filter_by_zones,
    check_existing,
    aggregate_capacity_by_resource,
    plot_capacity_bar
)

from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("capacity-plot-agent")


''' Checks if the capacity CSV is a brownfield/greenfield situation,
then aggregates capacity by resource group for plotting.
'''
@mcp.tool()
def check_capacity_setting(csv_path: str) -> dict:
    # Loads in default GenX capacity CSV and groups by resource type
    df = load_capacity_csv(csv_path)
    # check_existing function = whether StartCap > 0 (brownfield) or all StartCap = 0 (greenfield)
    return check_existing(df)

''' Returns aggregated StartCap, RetCap, NewCap, EndCap, and NetCap by resource
group for plotting. '''
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

''' Plot capacity based on what the user wants.
Making sure user states what output directory save the plotted PNGs. '''
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
        ^ user can choose to visualize one of these five
        scenario_name: Scenario name for the plot title (ask the user)
        period: Period for the plot title, e.g. "1" (ask the user)
        zones: Optional list of zone numbers (e.g., [2, 5, 7, 9]) to filter to
        before aggregating. Omit to plot all zones.

    Returns:
        dict with success status, message, and file path
    """
    # Valid plot types
    valid_types = ["StartCap", "RetCap", "NewCap", "EndCap", "NetCap"]

    if plot_type not in valid_types:
        return {
            "success": False,
            "message": f"Invalid plot_type '{plot_type}'. Must be one of: {valid_types}",
            "file_path": None
        }

    # Load data, filtering to the requested zones if given
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

    # Check if it's brownfield/greenfield
    setting_info = check_existing(df)
    is_brownfield = setting_info["is_brownfield"]

    # Handle greenfield cases
    if not is_brownfield:
        if plot_type in ["StartCap", "RetCap"]:
            return {
                "success": False,
                "message": f"Not meaningful to plot {plot_type} in greenfield case (all StartCap = 0). NewCap = EndCap = NetCap.",
                "setting": "greenfield"
            }
        elif plot_type == "NetCap":
            # In greenfield, NetCap = EndCap
            plot_type = "EndCap"
            message_suffix = " Note that NewCap = EndCap = NetCap in greenfield case!"
        else:
            message_suffix = ""
    else:
        message_suffix = ""

    # Create output file path
    output_path = Path(output_dir) / f"{plot_type}.png"

    # Title combines plot type, scenario name, and period (zones are not shown)
    title = f"{column_titles[plot_type]} {scenario_name} Period {period}"

    # Generate plot
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

# Entry point for running the server
if __name__ == "__main__":
    mcp.run()