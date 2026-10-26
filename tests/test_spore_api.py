"""Asserts (1) every SPORE name the engine imports still has the signature read on 18 Oct 2026 (SPORE v3.0, 5015dc6);
(2) the 10 SPORE files that SPORE-XL ships (src/phases/) are byte-identical to the recorded 5015dc6 hashes, and, when XL_SPORE_V3_DIR points at a checkout
of the tag v3.0, all 12 recorded files are (the 2 not shipped are spore_local.py and phase00_sparse_convert.py);
(3) the pieces of spore_local.py copied into src/engine/config_v3.py are character-for-character copies (hashes recorded from spore_local.py at 5015dc6),
and with a checkout load_config gives the same result as the real spore_local.load_config on every shipped config.
Run: python -B tests\\test_spore_api.py
Any failure means SPORE changed: re-read the function, update adapted_phases/ and this table together."""
import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "engine"))
import paths  # noqa: E402  (sets dont_write_bytecode and MPLBACKEND before SPORE is imported)

paths.add_spore_to_path()

# (module, name, expected str(inspect.signature))  -- read_notes.md section A
EXPECTED = [
    ("src.phases.utils", "load_protected_genes", "(protected_genes_path, adata, logger=None)"),
    ("src.phases.utils", "get_cell_cycle_genes", "()"),
    ("src.phases.phase01_detection", "run_phase1", "(adata, cfg: dict, logger)"),
    ("src.phases.phase02_cell_triage", "_ensure_qc_metrics", "(adata, cfg, logger)"),
    ("src.phases.phase05_escaper_filtering", "_get_escaper_direction", "(cfg: dict, logger) -> str"),
    ("src.phases.phase05_escaper_filtering", "_parse_target_genes", "(label: str, cfg: dict) -> list"),
    ("src.phases.phase06_gene_triage", "_get_perturbation_targets", "(adata, cfg: dict) -> set"),
    ("src.phases.phase08_hvg", "force_include_into_hvg",
     "(force_ids, hvg_names, hvg_stats, var_names, rank_col, do_not_drop, flag_col, tag, logger=None)"),
    ("src.phases.phase08_hvg", "load_heldout_target_ids",
     "(splits_dir, dataset_name, var_names, ctrl_label, sep='+', is_combinatorial=False, logger=None)"),
    ("src.phases.phase09_normalization", "normalize_split", "(adata, cfg: dict, logger, label: str = '')"),
    ("src.phases.phase12_metacell", "_compute_group_pca", "(X_sub, n_comps: int = 10, seed: int = 42)"),
    ("src.phases.phase12_metacell", "_aggregate_group_mbk", "(X_sub, cluster_emb, k, seed)"),
    ("src.phases.phase12_metacell", "_mean_rows", "(sub)"),
    ("src.phases.phase12_metacell", "_load_split_map", "(split_json_path, ctrl_label, logger)"),
    ("src.phases.phase12_metacell", "_compute_global_embedding", "(adata, n_comps: int = 50, seed: int = 42, logger=None)"),
    ("config_v3", "load_config", "(config_path: str) -> dict"),
    ("config_v3", "report_heldout_coverage", "(splits: dict, cfg: dict, logger) -> dict"),
    # functions the adapted copies reproduce: a signature change here means the copy must be re-read
    ("src.phases.phase02_cell_triage", "filter_cells", "(adata, cfg: dict, logger)"),
    ("src.phases.phase05_escaper_filtering", "filter_escapers", "(adata, cfg: dict, logger)"),
    ("src.phases.phase05_escaper_filtering", "filter_undersized_perturbations", "(adata, cfg: dict, logger)"),
    ("src.phases.phase05_escaper_filtering", "run_phase5", "(adata, cfg: dict, logger)"),
    ("src.phases.phase06_gene_triage", "filter_genes", "(adata, cfg: dict, logger)"),
    ("src.phases.phase07_splits", "split_zero_shot", "(adata, cfg, logger, seed=None)"),
    ("src.phases.phase07_splits", "_compute_mean_shift", "(adata, cfg, logger)"),
    ("src.phases.phase07_splits", "save_splits", "(split_result, cfg, logger, seed=None)"),
    ("src.phases.phase08_hvg", "select_hvgs", "(adata, cfg: dict, logger, split_label: str = 'train')"),
    ("src.phases.phase08_hvg", "run_phase8", "(splits, cfg, logger)"),
    ("src.phases.phase12_metacell", "run_phase12", "(cfg: dict, logger)"),
]
EXPECTED_CONSTANTS = [("src.phases.phase12_metacell", "_PER_GROUP_PCA_THRESHOLD", 50)]

# the 12 recorded SPORE v3.0 paths -> where SPORE-XL ships them (None = not shipped, verified in the v3.0 checkout only)
LOCATION = {"src/utils.py": "src/phases/utils.py", "spore_local.py": None, "src/phase00_sparse_convert.py": None}
for _name in ("phase01_detection", "phase02_cell_triage", "phase05_escaper_filtering", "phase06_gene_triage", "phase07_splits", "phase08_hvg",
              "phase08_worker", "phase09_normalization", "phase12_metacell"):
    LOCATION[f"src/{_name}.py"] = f"src/phases/{_name}.py"


def checkout_dir():
    value = os.environ.get("XL_SPORE_V3_DIR", "").strip()
    return Path(value) if value else None


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def segments_of(text):
    """name -> source text of every piece config_v3.py copies from spore_local.py (same function as the build script)."""
    wanted = {"_DEFAULTS", "_deep_merge", "_resolve_paths", "_apply_light_overrides", "load_config", "_split_targets", "report_heldout_coverage"}
    found = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            found[node.name] = ast.get_source_segment(text, node)
        elif isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) in wanted:
            found[node.target.id] = ast.get_source_segment(text, node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            segment = ast.get_source_segment(text, node)
            if segment.startswith("os.environ.setdefault("):
                found["env:" + node.value.args[0].value] = segment
    return found


def test_signatures():
    problems = []
    for module_name, name, expected in EXPECTED:
        module = importlib.import_module(module_name)
        actual = str(inspect.signature(getattr(module, name)))
        if actual != expected:
            problems.append(f"{module_name}.{name}: expected {expected}  got {actual}")
    for module_name, name, expected in EXPECTED_CONSTANTS:
        actual = getattr(importlib.import_module(module_name), name)
        if actual != expected:
            problems.append(f"{module_name}.{name}: expected {expected} got {actual}")
    assert not problems, "\n".join(problems)
    print(f"  {len(EXPECTED) + len(EXPECTED_CONSTANTS)} imported/cloned SPORE names match their signatures")


def test_files_identical_to_5015dc6():
    recorded = json.loads((HERE / "spore_5015dc6_hashes.json").read_text(encoding="utf-8"))["sha256"]
    assert set(recorded) == set(LOCATION), "the recorded hashes and the location table list different files"
    shipped = {old: new for old, new in LOCATION.items() if new}
    changed = [new for old, new in shipped.items() if sha256_of(paths.SPORE_ROOT / new) != recorded[old]]
    assert not changed, f"SPORE files differ from the recorded 5015dc6 hashes: {changed}"
    note = f"  {len(shipped)} SPORE files byte-identical to the recorded hashes (src/phases/)"
    checkout = checkout_dir()
    if checkout is None:
        note += "; spore_local.py and phase00_sparse_convert.py are verified only in a v3.0 checkout (XL_SPORE_V3_DIR is not set: NOT checked here)"
    else:
        bad = [old for old in recorded if sha256_of(checkout / old) != recorded[old]]
        assert not bad, f"the v3.0 checkout {checkout} differs from the recorded hashes: {bad}"
        note += f"; all {len(recorded)} recorded files verified in the v3.0 checkout {checkout}"
    print(note)


def test_config_v3_pieces_are_verbatim_copies():
    record = json.loads((HERE / "spore_5015dc6_config_segments.json").read_text(encoding="utf-8"))["sha256_of_segments"]
    found = segments_of((paths.BUILD_ROOT / "config_v3.py").read_text(encoding="utf-8"))
    assert set(found) == set(record), f"config_v3.py pieces differ from the record: {sorted(set(found) ^ set(record))}"
    wrong = [name for name, segment in found.items() if hashlib.sha256(segment.encode("utf-8")).hexdigest() != record[name]]
    assert not wrong, f"config_v3.py pieces are not verbatim copies of spore_local.py@5015dc6: {wrong}"
    print(f"  {len(found)} pieces of config_v3.py are verbatim copies")


def test_config_v3_matches_spore_local_in_the_checkout():
    checkout = checkout_dir()
    if checkout is None:
        print("  (skipped: XL_SPORE_V3_DIR is not set)")
        return
    import config_v3
    spec = importlib.util.spec_from_file_location("spore_local_v3_checkout", checkout / "spore_local.py")
    real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real)
    assert config_v3._DEFAULTS == real._DEFAULTS, "_DEFAULTS differ"
    original = segments_of((checkout / "spore_local.py").read_text(encoding="utf-8"))
    copy = segments_of((paths.BUILD_ROOT / "config_v3.py").read_text(encoding="utf-8"))
    assert original == copy, "config_v3.py pieces differ from the checkout's spore_local.py"
    import re
    import yaml
    configs = [paths.SPORE_ROOT / "configs" / "template.yaml"] + sorted((paths.SPORE_ROOT / "configs" / "datasets").glob("*.yaml")) + sorted((paths.SPORE_ROOT / "tests" / "configs").glob("*.yaml"))
    base = Path(tempfile.mkdtemp(prefix="cfgv3_", dir=os.environ.get("XL_TEST_TMP", tempfile.gettempdir())))
    for index, path in enumerate(configs):
        text = re.sub(r"<[A-Z_0-9]+>", (base / "x").as_posix(), path.read_text(encoding="utf-8"))
        outputs = []
        for label, loader in (("a", config_v3.load_config), ("b", real.load_config)):   # each loader gets its own fresh run folder
            root = base / f"{label}{index}"
            root.mkdir()
            loaded = yaml.safe_load(text)
            loaded["paths"]["project_root"] = (root / "run").as_posix()
            config_path = root / "cfg.yaml"
            config_path.write_text(yaml.safe_dump(loaded), encoding="utf-8")
            outputs.append(repr(loader(str(config_path))).replace(root.as_posix(), "ROOT"))
        assert outputs[0] == outputs[1], f"load_config differs on {path.name}"
    print(f"  load_config and _DEFAULTS identical to the checkout's spore_local on {len(configs)} shipped configs")


def test_import_left_no_bytecode_in_spore():
    # the (empty) __pycache__ folder is kept on purpose (Patrick, 2026-10-18); what must never appear is a .pyc file
    stray = list(paths.SPORE_ROOT.rglob("*.pyc"))
    assert not stray, f"bytecode written inside SPORE: {stray}"


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS {name}")
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(error).__name__}: {error}")
    sys.exit(1 if failures else 0)
