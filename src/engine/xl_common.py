"""Run context, unit bookkeeping and small helpers shared by every XL stage.

Unit convention: a task writes into a fresh unit directory <root>/<stage>/<unit>/ and finishes by writing
<root>/<stage>/<unit>.done (sha256 + size of every file). Re-running a finished unit verifies and skips it.
A unit directory without a .done marker (interrupted write) stops the run; nothing is ever deleted or overwritten.
"""
import hashlib
import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import paths  # noqa: F401  (side effects: no bytecode, Agg backend)

paths.add_spore_to_path()
if str(paths.BUILD_ROOT) not in sys.path:
    sys.path.insert(0, str(paths.BUILD_ROOT))  # makes the `adapted_phases` package importable

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config_v3  # noqa: E402


# ---- v4 modes (Amendment 2). `xl.compat: v3` reproduces SPORE v3.0 exactly (test suite); anything else is the production default.
# Explicit keys in the user's config always win; only missing keys get these values.
PRODUCTION_DEFAULTS = {
    "phase5_escaper_filtering": {"escaper_score": "report"},                                # P4: score rule counted in the report, NOT applied
    "phase6_gene_triage": {"rescue_cell_cycle_genes": False},                                # P3
    "phase8_hvg": {"target_node_policy": "budget", "rescue_cell_cycle_genes": False,        # P2, P3
                   "tie_break": "stable"},                                                   # P8
    "xl": {"output_aliases": True, "write_train_hybrid_lognorm": True,                       # P1, P6
           "merge_split_outputs": "never"},                                                  # P10: parts are the product; `spore_xl.py merge` makes merged files
}
COMPAT_V3_DEFAULTS = {
    "phase5_escaper_filtering": {"escaper_score": "off"},
    "phase6_gene_triage": {"rescue_cell_cycle_genes": True},
    "phase8_hvg": {"target_node_policy": "all", "rescue_cell_cycle_genes": True, "tie_break": "spore_set"},
    "xl": {"output_aliases": False, "write_train_hybrid_lognorm": False, "merge_split_outputs": "auto"},
}


def apply_mode_defaults(cfg):
    """Fill in missing keys from the production or the compat-v3 table; returns the mode name."""
    compat = (cfg.get("xl", {}) or {}).get("compat")
    if compat not in (None, False, "v3"):
        raise ValueError(f"xl.compat must be v3 or absent, got {compat!r}")
    table = COMPAT_V3_DEFAULTS if compat == "v3" else PRODUCTION_DEFAULTS
    for section, values in table.items():
        target = cfg.setdefault(section, {})
        for key, value in values.items():
            target.setdefault(key, value)
    return "compat-v3" if compat == "v3" else "production"


class Run:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        import re
        placeholders = sorted({m for line in self.config_path.read_text(encoding="utf-8").split(chr(10)) if not line.lstrip().startswith("#")
                               for m in re.findall(r"<[A-Z_]+>", line)})
        if placeholders:  # a config still holding <WORK_DIR> / <TAR_PATH> would create folders with those names
            raise ValueError(f"{self.config_path.name} still contains placeholders {placeholders}: replace them with real paths first")
        self.cfg = config_v3.load_config(str(self.config_path))  # mkdirs processed/splits/figures/logs inside the run dir
        self.xl = self.cfg.setdefault("xl", {}) or {}
        self.cfg["xl"] = self.xl
        self.mode = apply_mode_defaults(self.cfg)
        self.run_dir = Path(self.cfg["paths"]["_root"]).resolve()
        self.manifest_dir = self.run_dir / "manifest"
        self.stats_dir = self.run_dir / "stats"
        self.splits_dir = Path(self.cfg["paths"]["_splits"])
        self.processed_dir = Path(self.cfg["paths"]["_processed"])
        self.reports_dir = self.run_dir / "reports"
        self.logs_dir = Path(self.cfg["paths"]["_logs"])
        self.config_dir = self.run_dir / "config"
        self.work_dir = self.run_dir / self.xl.get("work_dir", "_xl_work")
        for directory in (self.manifest_dir, self.stats_dir, self.reports_dir, self.config_dir, self.work_dir):
            directory.mkdir(parents=True, exist_ok=True)
        dataset = self.cfg["dataset"]
        self.name = dataset["name"]
        self.pert_col = dataset.get("perturbation_col", "gene")
        self.ctrl_label = dataset.get("control_label", "non-targeting")
        self.n_blocks = int(self.xl.get("n_blocks", 64))
        self.control_block_rows = int(self.xl.get("control_block_rows", 120000))
        self._apply_runtime_updates()

    # -- settings that run_phase1 changed in cfg during `plan` are persisted and re-applied by every later stage
    def _apply_runtime_updates(self):
        path = self.manifest_dir / "cfg_runtime.json"
        if path.exists():
            updates = json.loads(path.read_text(encoding="utf-8"))
            self.cfg["dataset"].update(updates.get("dataset", {}))
            if updates.get("_detection_overrides"):
                self.cfg["_detection_overrides"] = updates["_detection_overrides"]
        gated = self.manifest_dir / "protected_genes_gated.json"   # written by the expression gate of `run`: plan, hvg and every worker read THIS list
        if (self.cfg.get("phase8_hvg", {}) or {}).get("protected_expression_gate") and gated.exists():
            self.cfg["phase8_hvg"]["protected_gene_list"] = str(gated)

    @property
    def escaper_mode(self):
        """phase5_escaper_filtering.escaper_score: off | report | union | replace (P4)."""
        import escaper_score
        return escaper_score.normalize_mode(self.cfg.get("phase5_escaper_filtering", {}).get("escaper_score", "off"))

    def kind_name(self, kind):
        """Directory suffix of an output kind: the normalized panel is `norm` in production (old name `p9` is an alias), `p9` in compat."""
        return "norm" if (kind == "p9" and self.mode == "production") else kind

    def reload_runtime_updates(self):
        self._apply_runtime_updates()

    @property
    def logger(self):
        import logging
        logger = logging.getLogger(f"xl_{self.name}")
        if not logger.handlers:
            formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False  # one line per message even when a library configured the root logger
            logger.setLevel(logging.INFO)
            console_log, parent = os.environ.get("XL_CONSOLE_LOG"), os.environ.get("XL_CONSOLE_PARENT_PID")
            if console_log and parent and parent != str(os.getpid()):  # a pool worker: the parent's tee cannot see this process
                file_handler = logging.FileHandler(console_log, mode="a", encoding="utf-8")
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        return logger

    # -- manifest tables
    def shards(self):
        return pd.read_csv(self.manifest_dir / "shards.tsv", sep="\t", keep_default_na=False)

    def n_shards(self):
        return len(self.shards())

    def layout(self):
        return json.loads((self.manifest_dir / "layout.json").read_text(encoding="utf-8"))

    def source_info(self):
        return json.loads((self.manifest_dir / "source.json").read_text(encoding="utf-8"))

    def var_table(self):
        return pd.read_parquet(self.manifest_dir / "var.parquet")

    def labels_table(self):
        return pd.read_csv(self.manifest_dir / "labels.tsv", sep="\t", keep_default_na=False, dtype={"label": str})

    def ribo_exempt(self):
        text = (self.manifest_dir / "ribo_exempt.txt").read_text(encoding="utf-8")
        return set(line for line in text.split("\n") if line)

    def relpath(self, path):
        """Path as stored in manifest tables: relative to the run directory when inside it (the usual case), else absolute.
        xl.work_dir may be an absolute path on another file system (for example /scratch while the run directory is in /group)."""
        try:
            return str(Path(path).relative_to(self.run_dir)).replace("\\", "/")
        except ValueError:
            return str(Path(path)).replace("\\", "/")

    def unit_dir(self, root, stage, unit):
        return (self.stats_dir if root == "stats" else self.work_dir) / stage / unit

    def marker(self, root, stage, unit):
        return (self.stats_dir if root == "stats" else self.work_dir) / stage / f"{unit}.done"


def sha256_file(path, block_bytes=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _unit_files(directory):
    return sorted(str(path.relative_to(directory)).replace("\\", "/") for path in directory.rglob("*") if path.is_file())


def verify_marker(marker, directory, run_dir=None):
    info = json.loads(marker.read_text(encoding="utf-8"))
    import cleanup  # local import (cleanup imports this module)
    # files removed by the opt-in clean-up are not missing; the log lives in the run directory (which holds the marker's work dir
    # only when xl.work_dir is relative, hence the explicit argument)
    removed = cleanup.removed_set(run_dir if run_dir is not None else Path(marker).resolve().parents[2])
    for relative, expected in info["files"].items():
        path = directory / relative
        if str(path.resolve()) in removed and not path.exists():
            continue
        if not path.exists() or path.stat().st_size != expected["size"] or sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"{path} does not match the checksum in {marker}")
    external = directory / "external_files.json"
    if external.exists():  # outputs that live outside the unit dir (for example under processed/)
        for path_text, expected in json.loads(external.read_text(encoding="utf-8")).items():
            path = Path(path_text)
            if not path.exists() or path.stat().st_size != expected["size"] or sha256_file(path) != expected["sha256"]:
                raise RuntimeError(f"{path} does not match the checksum recorded in {external}")
    return info


def register_external(directory, paths_written):
    """Record outputs a unit wrote outside its own directory; verified on every re-run."""
    record = {str(path): {"size": Path(path).stat().st_size, "sha256": sha256_file(path)} for path in paths_written}
    (Path(directory) / "external_files.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def _tree_bytes(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def quarantine_incomplete(run, stage, unit, paths_to_move):
    """Leftovers of a unit that was interrupted (no .done marker): MOVE them (never delete) to
    <work_dir>/_incomplete/<stage>_<unit>_<YYYYMMDD_HHMMSS>/NN_<name>, log every path (console, log file and
    reports/incomplete_moves.tsv) and return the destinations. The unit is then computed again from scratch."""
    import shutil
    existing = [Path(path) for path in paths_to_move if Path(path).exists() or Path(path).is_symlink()]
    if not existing:
        return []
    stamp = time.strftime("%Y%m%d_%H%M%S")
    folder = run.work_dir / "_incomplete" / f"{stage}_{unit}_{stamp}"
    counter = 1
    while folder.exists():  # two moves of one unit within one second
        counter += 1
        folder = run.work_dir / "_incomplete" / f"{stage}_{unit}_{stamp}_{counter}"
    folder.mkdir(parents=True)
    log_path = run.reports_dir / "incomplete_moves.tsv"
    header = not log_path.exists()
    moved = []
    with open(log_path, "a", encoding="utf-8") as handle:
        if header:
            handle.write("time\tstage\tunit\tfrom\tto\tbytes\n")
        for index, source in enumerate(existing):
            target = folder / f"{index:02d}_{source.name}"
            size = _tree_bytes(source) if not source.is_symlink() else 0
            shutil.move(str(source), str(target))
            run.logger.warning(f"[{stage} {unit}] interrupted earlier, partial output moved (not deleted): {source} -> {target}")
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{stage}\t{unit}\t{source}\t{target}\t{size}\n")
            moved.append(target)
    return moved


def ensure_absent(paths_to_check):
    for path in paths_to_check:
        if Path(path).exists():
            raise FileExistsError(f"{path} exists but its unit is not finished; Patrick decides")


def require_done(run, root, stage, unit):
    marker = run.marker(root, stage, unit)
    if not marker.exists():
        raise FileNotFoundError(f"upstream unit not finished: {marker}")
    return run.unit_dir(root, stage, unit)


def peak_rss_gb():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 1e9
    except Exception:  # noqa: BLE001
        return float("nan")


class _PeakSampler:
    """Tracks the highest resident set size seen while a task runs (peak_wset is per process lifetime, so a serial
    --local run would show the maximum of all earlier tasks)."""

    def __init__(self, interval=0.05):
        import threading
        self.peak = 0.0
        self._stop = threading.Event()
        self._interval = interval
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        import psutil
        process = psutil.Process()
        while not self._stop.is_set():
            self.peak = max(self.peak, process.memory_info().rss / 1e9)
            self._stop.wait(self._interval)

    @staticmethod
    def _linux_high_water_gb():
        try:  # VmHWM is the kernel's own peak resident size of this process
            for line in open("/proc/self/status", encoding="ascii"):
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024 / 1e9
        except OSError:
            pass
        return 0.0

    def __enter__(self):
        try:  # Linux 4.0+: writing 5 resets the peak so that VmHWM describes this task only
            with open("/proc/self/clear_refs", "w", encoding="ascii") as handle:
                handle.write("5")
        except OSError:
            pass
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        import psutil
        self.peak = max(self.peak, psutil.Process().memory_info().rss / 1e9, self._linux_high_water_gb())


@contextmanager
def task(run, root, stage, unit, log=True):
    """Yields the fresh unit directory, or None when the unit is already finished and verified.
    log=False is for the work unit nested inside a stats unit of the same stage: it still gets its directory and .done marker,
    but no sampler, log line or memory_and_time.tsv row (the enclosing unit reports once for both)."""
    directory = run.unit_dir(root, stage, unit)
    marker = run.marker(root, stage, unit)
    if marker.exists() and log:
        verify_marker(marker, directory, run.run_dir)
        run.logger.info(f"[{stage} {unit}] already done, skipped")
        yield None
        return
    # A unit directory without a marker is what a killed run leaves behind. A nested work unit (log=False) that IS finished while
    # its enclosing unit is not (killed between the two markers) is recomputed with its parent. Both are MOVED, never deleted.
    quarantine_incomplete(run, stage, unit, [path for path in (directory, marker) if path.exists()])
    directory.mkdir(parents=True)
    started = time.time()
    sampler = _PeakSampler() if log else None
    if sampler:
        sampler.__enter__()
    try:
        yield directory
    finally:
        if sampler:
            sampler.__exit__(None, None, None)
    files = {relative: {"size": (directory / relative).stat().st_size, "sha256": sha256_file(directory / relative)}
             for relative in _unit_files(directory)}
    elapsed = time.time() - started
    peak = round(sampler.peak, 3) if sampler else None
    marker.write_text(json.dumps({"stage": stage, "unit": unit, "seconds": round(elapsed, 2),
                                  "peak_rss_gb": peak, "files": files}, indent=2), encoding="utf-8")
    if not log:
        return
    log_path = run.reports_dir / "memory_and_time.tsv"
    header = not log_path.exists()
    with open(log_path, "a", encoding="utf-8") as handle:
        if header:
            handle.write("stage\tunit\tseconds\tpeak_rss_gb\n")
        handle.write(f"{stage}\t{unit}\t{elapsed:.2f}\t{sampler.peak:.3f}\n")
    run.logger.info(f"[{stage} {unit}] done in {elapsed:.1f}s, peak RSS {sampler.peak:.2f} GB")


def write_new_text(path, text):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(text, encoding="utf-8")


def write_new_json(path, payload):
    write_new_text(path, json.dumps(payload, indent=2))


def write_new_parquet(path, frame):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    frame.to_parquet(path)


def write_new_tsv(path, frame, index=False):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    frame.to_csv(path, sep="\t", index=index)


def shard_unit(index):
    return f"shard_{int(index):04d}"


def block_unit(index):
    return f"block_{int(index):04d}"


def block_of_label(label, n_blocks):
    # stable across runs, machines and python versions (unlike hash())
    digest = hashlib.blake2b(str(label).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % n_blocks


def control_block_of(global_index, control_cuts, n_blocks):
    return n_blocks + int(np.searchsorted(np.asarray(control_cuts, dtype=np.int64), global_index, side="right"))


def environment_text():
    from importlib.metadata import version  # scanpy.__version__ / anndata.__version__ raise FutureWarnings
    lines = [f"python {platform.python_version()}", f"platform {platform.platform()}",
             f"PYTHONHASHSEED {os.environ.get('PYTHONHASHSEED', '<unset>')}",
             f"numpy {np.__version__}", f"scipy {version('scipy')}", f"pandas {pd.__version__}",
             f"scanpy {version('scanpy')}", f"anndata {version('anndata')}", f"scikit-learn {version('scikit-learn')}"]
    return "\n".join(lines) + "\n"


def alias_map(var_table):
    """phase05_escaper_filtering.py:77-87 on the harmonized SOURCE var (QC columns excluded, blueprint Amendment 1 item 6)."""
    gene_to_idx = {}
    for index, name in enumerate(var_table.index):
        gene_to_idx[str(name)] = index
    for column in var_table.columns:
        for index, value in enumerate(var_table[column]):
            if pd.notna(value):
                gene_to_idx[str(value)] = index
    return gene_to_idx
