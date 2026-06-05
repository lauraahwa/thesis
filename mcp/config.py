import os

GENX_DIR = os.environ.get("GENX_DIR", "/home/yh2673/scratch/GenX.jl")

# Directories to scan when no scenarios_dir is specified
_SCENARIOS_ROOT = os.path.join(GENX_DIR, "scenarios")
DEFAULT_SCENARIOS_DIRS = [_SCENARIOS_ROOT] + [
    os.path.join(_SCENARIOS_ROOT, d)
    for d in sorted(os.listdir(_SCENARIOS_ROOT))
    if os.path.isdir(os.path.join(_SCENARIOS_ROOT, d)) and d.startswith("z")
]

LOG_DIR = os.path.join(GENX_DIR, "run_logs")

SLURM_DEFAULTS = {
    "partition":  os.environ.get("SLURM_PARTITION", "all"),
    "cpus":       2,
    "mail_user":  os.environ.get("SLURM_MAIL_USER", "yh2673@princeton.edu"),
}


def infer_slurm_resources(genx_settings: dict, tdr_settings: dict) -> dict:
    """
    Infer mem_gb and time_hours from MaxPeriods × NumStages.

    Calibration anchors (27-zone PJM):
      18 weeks × 2 stages (36 eff. periods)  →  5h, 128GB
      52 weeks × 2 stages (104 eff. periods) → 24h, 256GB

    Both time and memory are linearly interpolated between anchors.
    Below lower anchor: scales down proportionally.
    Above upper anchor: capped at 48h / 256GB.

    CPUs are always 2 regardless of problem size.
    """
    rep_periods = tdr_settings.get("MaxPeriods", 18)
    multi_stage = genx_settings.get("MultiStage", 0)
    num_stages  = genx_settings.get("NumStages", 1) if multi_stage >= 1 else 1

    eff = rep_periods * num_stages

    LOW_EFF,  LOW_H,  LOW_MEM  = 36,  5,  128
    HIGH_EFF, HIGH_H, HIGH_MEM = 104, 24, 256

    if eff <= LOW_EFF:
        frac = eff / LOW_EFF
        time_hours = max(1, round(LOW_H  * frac))
        mem_gb     = round(LOW_MEM * frac)
    elif eff <= HIGH_EFF:
        frac = (eff - LOW_EFF) / (HIGH_EFF - LOW_EFF)
        time_hours = round(LOW_H   + frac * (HIGH_H   - LOW_H))
        mem_gb     = round(LOW_MEM + frac * (HIGH_MEM - LOW_MEM))
    else:
        time_hours = 48
        mem_gb     = 256

    return {"mem_gb": mem_gb, "time_hours": time_hours}
