import os
import yaml
from datetime import datetime
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DEFAULT_SCENARIOS_DIRS


def list_cases(scenarios_dir: Optional[str] = None) -> list[dict]:
    """
    Scan one or more scenario directories and return metadata for every valid
    GenX case found.

    A valid case must contain:
      - Run.jl
      - settings/genx_settings.yml

    Args:
        scenarios_dir: Path to a specific directory to scan. If omitted,
                       scans all DEFAULT_SCENARIOS_DIRS.

    Returns:
        List of dicts, one per case:
          name          - folder name
          path          - absolute path
          has_results   - True if results/ exists and is non-empty
          last_modified - ISO date string of folder mtime
          rep_periods   - MaxPeriods from TDR settings (None if not found)
          num_stages    - NumStages from multi_stage_settings (None if not found)
    """
    dirs_to_scan = [scenarios_dir] if scenarios_dir else DEFAULT_SCENARIOS_DIRS

    cases = []
    for scan_dir in dirs_to_scan:
        if not os.path.isdir(scan_dir):
            continue
        for name in sorted(os.listdir(scan_dir)):
            path = os.path.join(scan_dir, name)
            if not os.path.isdir(path):
                continue
            if not _is_valid_case(path):
                continue
            cases.append(_describe_case(name, path))

    return cases


def _is_valid_case(path: str) -> bool:
    return (
        os.path.isfile(os.path.join(path, "Run.jl")) and
        os.path.isfile(os.path.join(path, "settings", "genx_settings.yml"))
    )


def _describe_case(name: str, path: str) -> dict:
    results_path = os.path.join(path, "results")
    has_results  = os.path.isdir(results_path) and bool(os.listdir(results_path))

    last_modified = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")

    # TDR settings → representative periods
    tdr_settings = _load_yaml(os.path.join(path, "settings", "time_domain_reduction_settings.yml"))
    rep_periods  = tdr_settings.get("MaxPeriods") if tdr_settings else None

    # multi-stage settings → number of investment stages
    ms_settings = _load_yaml(os.path.join(path, "settings", "multi_stage_settings.yml"))
    num_stages  = ms_settings.get("NumStages") if ms_settings else None

    return {
        "name":          name,
        "path":          path,
        "has_results":   has_results,
        "last_modified": last_modified,
        "rep_periods":   rep_periods,
        "num_stages":    num_stages,
    }

# multi_stage_settings.yml (NumStages) lives here
# calls now so infer_slurm_resources() can use it later
def _load_yaml(path: str) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError):
        return {}
