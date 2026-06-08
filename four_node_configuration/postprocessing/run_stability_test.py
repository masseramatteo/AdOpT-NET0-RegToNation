"""
Stability test for decision tree hyperparameters.

Runs decision_tree_local_production.py (REQUIRE_BOTH=True) and
decision_tree_pooled_clusters.py with different DT parameter combinations.

Outputs saved to:
  X:\\3000_simulations_24_may_2026\\Snellius_3000_simulations\\stability\\
    local_both\\<label>\\   ← local file, both clusters must be installed
    pooled\\<label>\\       ← pooled file, one row per cluster
"""

import os
import re
import subprocess
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
STABILITY_BASE = Path(
    r"X:\3000_simulations_24_may_2026\Snellius_3000_simulations\stability"
)

POSTPROCESSING_DIR = Path(__file__).resolve().parent

# (label, DT_MAX_DEPTH, DT_MIN_SAMPLES_LEAF, DT_MIN_SAMPLES_SPLIT, DT_RANDOM_STATE)
PARAM_GRID = [
    ("baseline_rs0",    4, 40, 30,  0),   # baseline
    ("baseline_rs1",    4, 40, 30,  1),   # random state sensitivity
    ("baseline_rs42",   4, 40, 30, 42),
    ("baseline_rs123",  4, 40, 30, 123),
    ("depth3",          3, 40, 30,  0),   # depth sensitivity
    ("depth5",          5, 40, 30,  0),
    ("depth6",          6, 40, 30,  0),
    ("fine_leaf",       4, 20, 15,  0),   # leaf-size sensitivity
    ("coarse_leaf",     4, 60, 50,  0),
]

# Scripts to run and their extra overrides
SCRIPTS = [
    {
        "source":    "decision_tree_local_production.py",
        "subfolder": "local_both",
        "extra": {
            "ELECTROLYZER_REQUIRE_BOTH_CLUSTERS": "True",
            # keep only DT1
            "RUN_DT_2_PIPELINE_SUPPLY":  "False",
            "RUN_DT_4_OUTFLOW":          "False",
            "RUN_DT_5_SMALL_CONNECT":    "False",
            "RUN_DT_6_FULL_CONNECT":     "False",
            "RUN_DT_7_NET_EXPORT":       "False",
            "RUN_DT_8_TWO_CONN":         "False",
            "RUN_DT_9_PRESSURE":         "False",
        },
    },
    {
        "source":    "decision_tree_pooled_clusters.py",
        "subfolder": "pooled",
        "extra": {},   # DT1-only already set in this file
    },
]


def patch_content(content: str, label: str, depth: int, min_leaf: int,
                  min_split: int, rs: int, out_folder: Path, extra: dict) -> str:
    """Apply all parameter patches to the script content."""

    # Add matplotlib Agg backend (headless)
    content = "import matplotlib\nmatplotlib.use('Agg')\n" + content

    # DT hyperparameters
    content = re.sub(r"^DT_MAX_DEPTH\s*=.*$",        f"DT_MAX_DEPTH = {depth}",     content, flags=re.MULTILINE)
    content = re.sub(r"^DT_MIN_SAMPLES_LEAF\s*=.*$",  f"DT_MIN_SAMPLES_LEAF = {min_leaf}",  content, flags=re.MULTILINE)
    content = re.sub(r"^DT_MIN_SAMPLES_SPLIT\s*=.*$", f"DT_MIN_SAMPLES_SPLIT = {min_split}", content, flags=re.MULTILINE)
    content = re.sub(r"^DT_RANDOM_STATE\s*=.*$",      f"DT_RANDOM_STATE = {rs}",    content, flags=re.MULTILINE)

    # Redirect analysis output folder (keep results_folder for data reading)
    out_str = str(out_folder).replace("\\", "\\\\")
    content = content.replace(
        "('combined', data_train, data_test, results_folder)",
        f"('combined', data_train, data_test, r\"{str(out_folder)}\")",
    )

    # Disable archetype-level analysis to keep runs fast
    content = re.sub(r"^ANALYZE_BY_ARCHETYPE\s*=.*$",
                     "ANALYZE_BY_ARCHETYPE = False", content, flags=re.MULTILINE)

    # Extra per-script overrides
    for key, val in extra.items():
        content = re.sub(rf"^{re.escape(key)}\s*=.*$", f"{key} = {val}",
                         content, flags=re.MULTILINE)

    return content


def run_config(script_cfg: dict, label: str, depth: int, min_leaf: int,
               min_split: int, rs: int):
    source_path = POSTPROCESSING_DIR / script_cfg["source"]
    out_folder  = STABILITY_BASE / script_cfg["subfolder"] / label
    out_folder.mkdir(parents=True, exist_ok=True)

    with open(source_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = patch_content(content, label, depth, min_leaf, min_split, rs,
                            out_folder, script_cfg["extra"])

    temp_path = POSTPROCESSING_DIR / f"_stability_temp_{script_cfg['subfolder']}_{label}.py"
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  Running ... ", end="", flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, str(temp_path)],
        capture_output=True, text=True, timeout=600,
        encoding="utf-8", env=env,
    )

    # Save log
    log_path = out_folder / "run_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Config: depth={depth}, min_leaf={min_leaf}, min_split={min_split}, rs={rs}\n")
        f.write(f"Script: {script_cfg['source']}\n")
        f.write("=" * 60 + "\nSTDOUT:\n")
        f.write(result.stdout)
        f.write("\n" + "=" * 60 + "\nSTDERR:\n")
        f.write(result.stderr)

    temp_path.unlink(missing_ok=True)

    if result.returncode == 0:
        print("OK")
    else:
        print(f"FAILED (rc={result.returncode})")
        # Print last 20 lines of stderr for quick diagnosis
        lines = result.stderr.strip().splitlines()
        for ln in lines[-20:]:
            print(f"    {ln}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    total = len(SCRIPTS) * len(PARAM_GRID)
    done  = 0

    for script_cfg in SCRIPTS:
        print(f"\n{'='*70}")
        print(f"Script: {script_cfg['source']}  ->  {script_cfg['subfolder']}/")
        print(f"{'='*70}")

        for label, depth, min_leaf, min_split, rs in PARAM_GRID:
            done += 1
            print(f"[{done:02d}/{total}] {label:25s}  "
                  f"depth={depth}  leaf={min_leaf:3d}  split={min_split:3d}  rs={rs:3d}", end="  ")
            try:
                run_config(script_cfg, label, depth, min_leaf, min_split, rs)
            except subprocess.TimeoutExpired:
                print("TIMEOUT (>10 min)")
            except Exception as e:
                print(f"ERROR: {e}")

    print(f"\n{'='*70}")
    print(f"Done. Results saved to:\n  {STABILITY_BASE}")
