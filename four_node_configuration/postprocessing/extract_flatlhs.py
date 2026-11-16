"""
extracted_results.xlsx for the flat-LHS validation campaign (2026-11).

Runs extract_design_sizes.py unchanged on the flat-LHS results folder, with the
topology metrics joined from preprocess/generated_topology_flatlhs instead of
generated_topology_v5 (topology ids F0001..F1500). Infeasible runs have no h5 and
are skipped by the extractor. parallel_results_summary is not touched (the
watchdog wrote it).

Run:  python extract_flatlhs.py
Out:  <FLATLHS_FOLDER>/extracted_results.xlsx (sheet 'Results', same layout as v5)
"""
import importlib.util
from pathlib import Path

FLATLHS_FOLDER = (
    r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M"
    r"\AdOpT-NET0-RegToNation\four_node_configuration\results"
    r"\flatlhs_validation_20261028_171519"
)
HERE = Path(__file__).resolve().parent
TOPOLOGY_FOLDER = HERE.parent / "preprocess" / "generated_topology_flatlhs"

spec = importlib.util.spec_from_file_location("extract_design_sizes", HERE / "extract_design_sizes.py")
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)

# load_topology_parameters binds its default folder (v5) at definition time
_load_v5_default = ext.load_topology_parameters
ext.load_topology_parameters = lambda topology_folder=TOPOLOGY_FOLDER: _load_v5_default(topology_folder)


if __name__ == "__main__":
    all_results = ext.extract_all_runs(FLATLHS_FOLDER)
    df = ext.create_summary_dataframe(all_results)
    missing = [c for c in ("economy", "slot", "archetype", "d_LL", "d_SS", "d_SL_mean")
               if c not in df.columns or df[c].isna().any()]
    print(f"\n{len(df)} runs extracted; columns missing or incomplete: {missing or 'none'}")
    ext.export_to_excel(df, Path(FLATLHS_FOLDER) / "extracted_results.xlsx")
