import os
import subprocess
import textwrap
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import GENX_DIR, LOG_DIR, SLURM_DEFAULTS, infer_slurm_resources, DEFAULT_SCENARIOS_DIRS
from tools.cases import _is_valid_case, _load_yaml


def find_case(case_name: str) -> str:
    """
    Resolve a case name to an absolute path by scanning DEFAULT_SCENARIOS_DIRS.
    Raises ValueError if not found or ambiguous.
    """
    matches = []
    for scan_dir in DEFAULT_SCENARIOS_DIRS:
        candidate = os.path.join(scan_dir, case_name)
        if os.path.isdir(candidate) and _is_valid_case(candidate):
            matches.append(candidate)

    if not matches:
        raise ValueError(f"Case '{case_name}' not found in: {DEFAULT_SCENARIOS_DIRS}")
    if len(matches) > 1:
        raise ValueError(f"Case '{case_name}' found in multiple directories: {matches}")
    return matches[0]


def build_script(case_name: str, time_hours: int, mem_gb: int) -> str:
    case_path = find_case(case_name)
    partition = SLURM_DEFAULTS["partition"]
    cpus      = SLURM_DEFAULTS["cpus"]
    mail_user = SLURM_DEFAULTS["mail_user"]

    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={case_name}
        #SBATCH --output={LOG_DIR}/genx_case_%j.out
        #SBATCH --error={LOG_DIR}/genx_case_%j.err
        #SBATCH --time={time_hours}:00:00
        #SBATCH --mem={mem_gb}G
        #SBATCH --cpus-per-task={cpus}
        #SBATCH --partition={partition}
        #SBATCH --mail-type=BEGIN,END,FAIL
        #SBATCH --mail-user={mail_user}

        export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
        export JULIA_CPU_TARGET="generic;skylake=avx512;clone_all;znver2;clone_all;znver3;clone_all"

        echo "=========================================="
        echo "Job ID: $SLURM_JOB_ID"
        echo "Case: {case_name}"
        echo "Case dir: {case_path}"
        echo "Start time: $(date)"
        echo "=========================================="

        module load julia/1.10.5
        module load gurobi/9.0.1

        cd "{case_path}"
        julia --project="{GENX_DIR}" Run.jl
        exit_code=$?

        echo ""
        echo "=========================================="
        echo "Exit code: $exit_code"
        echo "End time: $(date)"
        echo "=========================================="
        exit $exit_code
    """)


def _resolve_resources(case_name: str, time_hours: Optional[int], mem_gb: Optional[int]) -> tuple:
    """Load case settings and infer resources, applying any user overrides."""
    case_path = find_case(case_name)
    genx_settings = _load_yaml(os.path.join(case_path, "settings", "genx_settings.yml"))
    tdr_settings  = _load_yaml(os.path.join(case_path, "settings", "time_domain_reduction_settings.yml"))
    inferred = infer_slurm_resources(genx_settings, tdr_settings)

    final_time = time_hours if time_hours is not None else inferred["time_hours"]
    final_mem  = mem_gb    if mem_gb    is not None else inferred["mem_gb"]
    return final_time, final_mem, inferred


def preview_case(case_name: str, time_hours: Optional[int] = None, mem_gb: Optional[int] = None) -> dict:
    """
    Generate the SLURM script for a case without submitting it.
    Returns the script text and the inferred (and final) resource values.
    """
    final_time, final_mem, inferred = _resolve_resources(case_name, time_hours, mem_gb)
    script = build_script(case_name, final_time, final_mem)
    return {
        "case_name":         case_name,
        "case_path":         find_case(case_name),
        "inferred_time_h":   inferred["time_hours"],
        "inferred_mem_gb":   inferred["mem_gb"],
        "final_time_h":      final_time,
        "final_mem_gb":      final_mem,
        "cpus":              SLURM_DEFAULTS["cpus"],
        "script":            script,
    }


def submit_case(case_name: str, time_hours: Optional[int] = None, mem_gb: Optional[int] = None) -> dict:
    """
    Submit a GenX case to SLURM via sbatch. Returns job_id and resource info.
    """
    final_time, final_mem, inferred = _resolve_resources(case_name, time_hours, mem_gb)
    script = build_script(case_name, final_time, final_mem)

    os.makedirs(LOG_DIR, exist_ok=True)
    result = subprocess.run(
        ["sbatch", "--parsable"],
        input=script,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")

    job_id = result.stdout.strip()
    return {
        "job_id":          job_id,
        "case_name":       case_name,
        "case_path":       find_case(case_name),
        "time_h":          final_time,
        "mem_gb":          final_mem,
        "cpus":            SLURM_DEFAULTS["cpus"],
        "inferred_time_h": inferred["time_hours"],
        "inferred_mem_gb": inferred["mem_gb"],
    }
