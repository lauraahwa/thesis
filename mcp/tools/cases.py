import os
import yaml
from datetime import datetime
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tools.slurm import GENX_DIR


def list_cases(scenarios_dir: Optional[str] = None) -> list[dict]:
    """
    Recursively scan a directory tree and return metadata for every valid
    GenX case found, at any depth.

    A valid case must contain:
      - Run.jl
      - settings/genx_settings.yml

    Args:
        scenarios_dir: Root directory to scan. If omitted, scans
                       GENX_DIR/scenarios.

    Returns:
        List of dicts, one per case:
          name          - folder name
          path          - absolute path
          has_results   - True if results/ exists and is non-empty
          last_modified - ISO date string of folder mtime
          rep_periods   - MaxPeriods from TDR settings (None if not found)
          num_stages    - NumStages from multi_stage_settings (None if not found)
    """
    root = scenarios_dir or os.path.join(GENX_DIR, "scenarios")

    cases = []
    if not os.path.isdir(root):
        return cases
    for dirpath, dirnames, _ in os.walk(root):
        if _is_valid_case(dirpath):
            cases.append(_describe_case(os.path.basename(dirpath), dirpath))
            dirnames[:] = []  # prune: don't descend into a case's own subfolders
            continue
        dirnames.sort()

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

def _load_yaml(path: str) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError):
        return {}
