# MCP wrapper around diurnal_generation.py (weighted average-day generation
# stacks from GenX TDR results). The plotting code lives at the GenX.jl repo
# root; this module locates it via GENX_DIR (or the parent of this package)
# and adds a single-case variant of the two-case comparison plot.

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from npv_costs import resolve_scenario


def _import_diurnal_generation():
    """Import diurnal_generation.py from GENX_DIR or this package's parent dir."""
    candidates = [os.environ.get("GENX_DIR"), str(Path(__file__).resolve().parent.parent)]
    for d in candidates:
        if d and os.path.isfile(os.path.join(d, "diurnal_generation.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            import diurnal_generation
            return diurnal_generation
    raise ImportError(
        "diurnal_generation.py not found in GENX_DIR or the package parent directory"
    )


dg = _import_diurnal_generation()


def resolve_case(case_dir: str, period: int) -> str:
    """Resolve `case_dir` to an absolute path containing
    results/results_p{period}/power.csv (see npv_costs.resolve_scenario)."""
    marker = os.path.join("results", f"results_p{period}", "power.csv")
    return resolve_scenario(case_dir, marker=marker)


def plot_single(df, label, title, out_png):
    """Stacked area chart of an average-day generation profile."""
    vmax = df.sum(axis=1).max()
    unit_div, unit = (1000.0, "GW") if vmax > 10000 else (1.0, "MW")
    cols = [g for g in dg.STACK_ORDER if g in df.columns]
    df = df.reindex(columns=cols, fill_value=0.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor="white")
    dg._stack(ax, df, unit_div)
    dg._style_axis(ax)
    ax.set_title(label, fontsize=11, color=dg.INK, pad=8)
    ax.set_xlabel("Hour of day", fontsize=9.5, color=dg.INK2)
    ax.set_ylabel(f"Average generation ({unit})", fontsize=9.5, color=dg.INK2)

    handles = [Patch(facecolor=dg.COLORS[g], edgecolor="white",
                     hatch="///" if g == "DR" else None, label=g)
               for g in reversed(cols)]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.86, 0.5),
               frameon=False, fontsize=9, labelcolor=dg.INK2)
    fig.suptitle(title, fontsize=12, color=dg.INK, x=0.1, ha="left")
    fig.subplots_adjust(left=0.1, right=0.84, top=0.84, bottom=0.13)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_diurnal_generation(
    case_dir: str,
    output_path: str,
    period: int,
    zones: str,
    labels: str,
    compare_case_dir: str | None = None,
    diff: bool = False,
) -> dict:
    """Implementation behind the MCP tool. Returns a result dict."""
    try:
        case = resolve_case(case_dir, period)
        primary = dg.diurnal_by_tech(case, period, zones)

        zone_lbl = {"pjm": "PJM zones", "island": "DC Island (z28)",
                    "all": "all zones"}.get(zones, f"zones {zones}")
        label_list = [s.strip() for s in labels.split(",")]

        out = Path(os.path.expanduser(output_path))
        out.parent.mkdir(parents=True, exist_ok=True)

        if compare_case_dir is None:
            if diff:
                return {"success": False,
                        "message": "diff=True requires compare_case_dir."}
            title = f"Average day, generation by technology — {zone_lbl}, p{period}"
            plot_single(primary, label_list[0], title, str(out))
        else:
            other_case = resolve_case(compare_case_dir, period)
            other = dg.diurnal_by_tech(other_case, period, zones)
            if len(label_list) < 2:
                label_list = ["Original", "Comparison"]
            kind = "difference" if diff else "generation by technology"
            title = f"Average day, {kind} — {zone_lbl}, p{period}"
            if diff:
                dg.plot_difference(primary, other, label_list, title, str(out))
            else:
                dg.plot_comparison(primary, other, label_list, title, str(out))

        return {
            "success": True,
            "message": f"Wrote {out}",
            "file_path": str(out),
            "case_dir": case,
            "compare_case_dir": None if compare_case_dir is None else other_case,
            "tech_groups": list(primary.columns),
        }
    except Exception as e:  # surface a clean error to the MCP client
        return {"success": False, "message": f"{type(e).__name__}: {e}"}
