import os
import subprocess
import textwrap
from typing import Optional


def _require(name: str) -> str:
    """Read a required environment variable or fail with a clear message."""
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            f"Copy .env.example to .env and fill it in (see README)."
        )
    return val


# Required: absolute path to the GenX.jl checkout this server submits cases from.
GENX_DIR = _require("GENX_DIR")

# Where SLURM logs are written. Defaults to <GENX_DIR>/run_logs.
LOG_DIR = os.environ.get("GENX_LOG_DIR", os.path.join(GENX_DIR, "run_logs"))

SLURM_DEFAULTS = {
    "partition":  os.environ.get("SLURM_PARTITION", "all"),
    "cpus":       int(os.environ.get("SLURM_CPUS_DEFAULT", "4")),
    # Optional: if unset, no SLURM mail lines are emitted.
    "mail_user":  os.environ.get("SLURM_MAIL_USER"),
}

# Cluster module / build settings (all optional, env-driven).
JULIA_MODULE = os.environ.get("JULIA_MODULE")        # e.g. "julia/1.10.5"
GUROBI_MODULE = os.environ.get("GUROBI_MODULE")      # e.g. "gurobi/9.0.1"
JULIA_CPU_TARGET = os.environ.get("JULIA_CPU_TARGET")  # e.g. "generic;skylake=avx512;..."


def _is_valid_case(path: str) -> bool:
    return (
        os.path.isfile(os.path.join(path, "Run.jl")) and
        os.path.isfile(os.path.join(path, "settings", "genx_settings.yml"))
    )


def find_case(case_dir: str) -> str:
    """
    Resolve a case directory to an absolute path.

    `case_dir` may be an absolute path, a path relative to GENX_DIR, or a path
    relative to the current working directory. The target must be a valid GenX
    case (contains Run.jl and settings/genx_settings.yml).

    Raises ValueError if the path does not resolve to a valid case.
    """
    expanded = os.path.expanduser(case_dir)
    candidates = (
        [expanded] if os.path.isabs(expanded)
        else [os.path.join(GENX_DIR, expanded), os.path.abspath(expanded)]
    )
    for candidate in candidates:
        if os.path.isdir(candidate) and _is_valid_case(candidate):
            return os.path.abspath(candidate)

    raise ValueError(
        f"'{case_dir}' is not a valid GenX case directory "
        f"(expected Run.jl + settings/genx_settings.yml). Tried: {candidates}"
    )


def build_script(case_path: str, time_hours: int, mem_gb: int, cpus: int = None, case_name: str = None) -> str:
    job_name  = case_name or os.path.basename(os.path.normpath(case_path))
    partition = SLURM_DEFAULTS["partition"]
    cpus      = cpus if cpus is not None else SLURM_DEFAULTS["cpus"]
    mail_user = SLURM_DEFAULTS["mail_user"]

    # Optional SLURM mail directives — only when a mail user is configured.
    mail_lines = ""
    if mail_user:
        mail_lines = (
            f"#SBATCH --mail-type=BEGIN,END,FAIL\n"
            f"#SBATCH --mail-user={mail_user}\n"
        )

    # Optional JULIA_CPU_TARGET export.
    cpu_target_line = ""
    if JULIA_CPU_TARGET:
        cpu_target_line = f'export JULIA_CPU_TARGET="{JULIA_CPU_TARGET}"\n'

    # Optional module loads.
    module_lines = ""
    if JULIA_MODULE:
        module_lines += f"module load {JULIA_MODULE}\n"
    if GUROBI_MODULE:
        module_lines += f"module load {GUROBI_MODULE}\n"

    header = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --output={LOG_DIR}/genx_case_%j.out
        #SBATCH --error={LOG_DIR}/genx_case_%j.err
        #SBATCH --time={time_hours}:00:00
        #SBATCH --mem={mem_gb}G
        #SBATCH --cpus-per-task={cpus}
        #SBATCH --partition={partition}
        """)

    body = textwrap.dedent(f"""\
        export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
        {cpu_target_line}
        echo "=========================================="
        echo "Job ID: $SLURM_JOB_ID"
        echo "Case: {job_name}"
        echo "Case dir: {case_path}"
        echo "Start time: $(date)"
        echo "=========================================="

        {module_lines}
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

    return header + mail_lines + "\n" + body


def preview_case(
    case_dir: str,
    time_hours: int,
    mem_gb: int,
    cpus: Optional[int] = None,
    case_name: Optional[str] = None,
) -> dict:
    """
    Generate the SLURM script for a case without submitting it.
    Returns the script text and the resource values used.
    """
    case_path  = find_case(case_dir)
    final_cpus = cpus if cpus is not None else SLURM_DEFAULTS["cpus"]
    script     = build_script(case_path, time_hours, mem_gb, final_cpus, case_name=case_name)
    return {
        "case_name":  case_name or os.path.basename(case_path),
        "case_path":  case_path,
        "time_h":     time_hours,
        "mem_gb":     mem_gb,
        "cpus":       final_cpus,
        "script":     script,
    }


def submit_case(
    case_dir: str,
    time_hours: int,
    mem_gb: int,
    cpus: Optional[int] = None,
    case_name: Optional[str] = None,
) -> dict:
    """
    Submit a GenX case to SLURM via sbatch. Returns job_id and resource info.
    """
    case_path  = find_case(case_dir)
    final_cpus = cpus if cpus is not None else SLURM_DEFAULTS["cpus"]
    script     = build_script(case_path, time_hours, mem_gb, final_cpus, case_name=case_name)

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
        "job_id":     job_id,
        "case_name":  case_name or os.path.basename(case_path),
        "case_path":  case_path,
        "time_h":     time_hours,
        "mem_gb":     mem_gb,
        "cpus":       final_cpus,
    }
