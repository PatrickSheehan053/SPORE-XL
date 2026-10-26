"""Records sha256 of every SPORE source file the engine reads or clones (SPORE v3.0 = GitHub 5015dc6), from a checkout of the tag v3.0.
Refuses to overwrite (tests/spore_5015dc6_hashes.json already exists in the repository; this is how it was made).
Run: python -B tests\\record_spore_hashes.py --spore-v3-dir <checkout of the tag v3.0> [--out <new json>]"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

FILES = ["spore_local.py"] + [f"src/{name}" for name in (
    "utils.py", "phase00_sparse_convert.py", "phase01_detection.py", "phase02_cell_triage.py",
    "phase05_escaper_filtering.py", "phase06_gene_triage.py", "phase07_splits.py", "phase08_hvg.py",
    "phase08_worker.py", "phase09_normalization.py", "phase12_metacell.py")]

parser = argparse.ArgumentParser()
parser.add_argument("--spore-v3-dir", required=True)
parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "spore_5015dc6_hashes.json"))
args = parser.parse_args()
out = Path(args.out)
if out.exists():
    raise SystemExit(f"{out} exists; not overwriting")
root = Path(args.spore_v3_dir)
hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES}
out.write_text(json.dumps({"spore_commit": "5015dc6", "sha256": hashes}, indent=2), encoding="utf-8")
print(f"wrote {out}")
