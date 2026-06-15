# genx-tested-agent

An MCP server for working with [GenX](https://github.com/GenXProject/GenX.jl)
capacity-expansion runs through Claude, via the
[Model Context Protocol (MCP)](https://modelcontextprotocol.io).

A single server (`genx-tested-agent`) exposes two groups of tools:

- **Cluster** — submit GenX cases to SLURM (and dry-run the generated script) using
  natural-language commands.
- **Results** — plot and summarize GenX `capacity.csv` output.

> *"Submit `scenarios/PJM_Baseline_Example` with 12h, 3 cores, 128GB"*

## Tools

| Tool | Purpose |
|---|---|
| `submit_genx_case` | Build a SLURM batch script for a case and submit it with `sbatch`. Returns the job ID. |
| `preview_genx_case` | Return the generated SLURM script **without** submitting (dry run). |
| `summarize_capacity` | Aggregate `capacity.csv` (StartCap/RetCap/NewCap/EndCap/NetCap) by resource type, optional zone filter. |
| `check_capacity_setting` | Detect whether a run is **greenfield** (all `StartCap = 0`) or **brownfield**. |
| `plot_capacity` | Bar chart of one capacity metric, aggregated by resource type, saved as a PNG. |

Resources are classified into Coal, Natural Gas (incl. petroleum/oil), Solar,
Wind, Battery, Hydro, Nuclear, and Biomass, and always plotted in that order.
Aggregates under 10 MW in magnitude are dropped as solver noise.

A directory is treated as a valid GenX case when it contains both `Run.jl` and
`settings/genx_settings.yml`. `case_dir` may be absolute, relative to `GENX_DIR`,
or relative to your current working directory.

## Setup

Requires Python ≥ 3.10 and (for submission) a SLURM cluster with `sbatch` on
`PATH` plus a local GenX.jl checkout.

```bash
git clone <this-repo> genx-tested-agent
cd genx-tested-agent

# install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# configure your environment
cp .env.example .env
#   then edit .env — at minimum set GENX_DIR
```

### Configuration (`.env`)

All personal/cluster-specific settings come from environment variables, loaded
from `.env` at startup. **`.env` is gitignored — never commit it.** `.env.example`
is the public template; copy it and fill in your own values.

| Variable | Required | Default | Notes |
|---|---|---|---|
| `GENX_DIR` | **yes** | — | Absolute path to your GenX.jl checkout. Server won't start if unset. |
| `SLURM_PARTITION` | no | `all` | Partition to submit to. |
| `SLURM_CPUS_DEFAULT` | no | `4` | CPUs per task when the caller doesn't specify. |
| `SLURM_MAIL_USER` | no | _(none)_ | Email for job notifications. Blank → no mail directives. |
| `GENX_LOG_DIR` | no | `<GENX_DIR>/run_logs` | Where `.out`/`.err` logs go. |
| `JULIA_MODULE` | no | _(none)_ | e.g. `julia/1.10.5`. Blank → no `module load`. |
| `GUROBI_MODULE` | no | _(none)_ | e.g. `gurobi/9.0.1`. Blank → no `module load`. |
| `JULIA_CPU_TARGET` | no | _(none)_ | Optional multi-arch build target. |

Run `module avail` on your cluster to find the correct module names.

## Hooking it up to Claude

> Use **absolute paths** for both the Python interpreter and `server.py`.

### Claude Code

Register the server (run on the cluster, from any directory):

```bash
claude mcp add genx-tested-agent -- /ABSOLUTE/PATH/TO/.venv/bin/python /ABSOLUTE/PATH/TO/genx-tested-agent/server.py
```

Or copy `.mcp.json.example` to `.mcp.json` and edit the paths:

```json
{
  "mcpServers": {
    "genx-tested-agent": {
      "command": "/ABSOLUTE/PATH/TO/.venv/bin/python",
      "args": ["/ABSOLUTE/PATH/TO/genx-tested-agent/server.py"]
    }
  }
}
```

Then run `/mcp` inside Claude Code to verify it connected. Reconnect whenever you
edit the server code.

### Claude Desktop

Edit `claude_desktop_config.json` (macOS:
`~/Library/Application Support/Claude/claude_desktop_config.json`, Windows:
`%APPDATA%\Claude\claude_desktop_config.json`) with the same `mcpServers` block,
then fully restart Claude Desktop. Tools appear under the 🔨 icon.

## Usage

Ask Claude in plain language:

- *"Preview the SLURM script for `scenarios/PJM_Baseline_Example` at 12h, 3 cores, 128GB"* → dry run
- *"Submit that case"* → `sbatch`, returns job ID
- *"Summarize the capacity in `.../results/capacity.csv` for zones 10 and 23"*
- *"Plot NewCap for the baseline scenario, period 1, into `./plots`"*

Walltime and memory are **required** for submission — the tools ask for them if
you don't provide them rather than guessing. Plots are named by plot type only
(`{PlotType}.png`), so plotting a different scenario into the same directory
overwrites earlier PNGs.

## Repo layout

```
server.py          # MCP server: tool definitions (FastMCP)
slurm.py           # case resolution + SLURM script build/submit
plot_capacity.py   # loading, zone filtering, aggregation, plotting
run_server.sh      # run server with the MCP dev inspector (debugging)
.env.example       # config template — copy to .env and fill in
requirements.txt   # Python dependencies
```
