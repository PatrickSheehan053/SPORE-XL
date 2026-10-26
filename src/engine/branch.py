"""branch: start a NEW run directory from a finished run, reusing every stage before `--from` (hard links on the same drive,
copies otherwise), so a later stage can be re-run with different settings without touching the original run (nothing is
overwritten or deleted). Example: re-run `hvg` onward with phase8_hvg.target_node_policy: budget on the CD4 run.

  python -B src\\spore_xl.py branch --from-config run_all.yaml --config run_budget.yaml --from-stage hvg

The new config must have its own paths.project_root and the SAME settings for everything the reused stages depend on
(checked below); it may differ in the settings of `--from-stage` and later stages. Needs the work files of the reused stages:
a run whose gather blocks were removed by the opt-in clean-up (xl.cleanup.include_blocks) cannot be branched before map3.
"""
import json
import os
import shutil
from pathlib import Path

from xl_common import Run, sha256_file, verify_marker

ORDER = ["plan", "map1", "reduce1", "map2", "gather", "reduce2", "hvg", "map3", "metacell_prep", "metacell", "finalize"]
ROOT_OF = {"map1": ["stats", "work"], "map2": ["stats", "work"], "gather": ["stats", "work"], "reduce1": ["stats"], "reduce2": ["stats"],
           "hvg": ["stats"], "map3": ["work"], "metacell_prep": ["stats"], "metacell": ["stats"], "finalize": ["stats"]}
# config sections each stage depends on (a reused stage needs ALL sections up to and including its own to match)
SECTIONS = {"plan": ["dataset", "phase1_detection", "phase2_cell_triage", "phase8_hvg.protected_gene_list"], "map1": [],
            "reduce1": ["phase5_escaper_filtering"], "map2": [], "gather": [], "reduce2": ["phase6_gene_triage", "phase7_splits"],
            "hvg": ["phase8_hvg"], "map3": ["phase9_normalization"], "metacell_prep": ["phase12_metacell"], "metacell": [],
            "finalize": []}
XL_KEYS_UPSTREAM = ["source", "n_blocks", "control_block_rows"]


def _get(cfg, dotted):
    node = cfg
    for part in dotted.split("."):
        node = (node or {}).get(part) if isinstance(node, dict) else None
    return node


def check_compatible(source_cfg, target_cfg, from_stage):
    reused = ORDER[:ORDER.index(from_stage)]
    problems = []
    for stage in reused:
        for dotted in SECTIONS.get(stage, []):
            if _get(source_cfg, dotted) != _get(target_cfg, dotted):
                problems.append(f"{dotted} differs but stage {stage} (reused) depends on it")
    for key in XL_KEYS_UPSTREAM:
        if (source_cfg.get("xl") or {}).get(key) != (target_cfg.get("xl") or {}).get(key):
            problems.append(f"xl.{key} differs")
    if problems:
        raise ValueError("branch refused: " + "; ".join(problems))


def _link_or_copy(source, target, force_copy=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    if not force_copy:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def _copy_tree(source, target, only=None):
    for path in sorted(Path(source).rglob("*")):
        if path.is_file() and (only is None or path.name in only):
            small = path.name == "external_files.json" or path.suffix == ".done"  # rewritten later: never share these inodes
            _link_or_copy(path, Path(target) / path.relative_to(source), force_copy=small)


def _rebase_unit(marker, unit_dir, old_run_dir, new_run_dir):
    """A unit that owns files outside its directory (map3 / metacell / finalize) records absolute paths; point them at the new run
    and refresh the checksums of the rewritten record in the marker."""
    external = unit_dir / "external_files.json"
    if not external.exists():
        return
    record = json.loads(external.read_text(encoding="utf-8"))
    rebased = {key.replace(str(old_run_dir), str(new_run_dir)): value for key, value in record.items()}
    external.write_text(json.dumps(rebased, indent=2), encoding="utf-8")  # a COPY made by this branch, not the original run's file
    info = json.loads(marker.read_text(encoding="utf-8"))
    info["files"]["external_files.json"] = {"size": external.stat().st_size, "sha256": sha256_file(external)}
    marker.write_text(json.dumps(info, indent=2), encoding="utf-8")


def branch(source_config, target_config, from_stage):
    if from_stage not in ORDER or from_stage == "plan":
        raise ValueError(f"--from-stage must be one of {ORDER[1:]}")
    source, target = Run(source_config), Run(target_config)
    if source.run_dir == target.run_dir:
        raise ValueError("the new config must have its own paths.project_root")
    if (target.manifest_dir / "plan.done").exists():
        raise FileExistsError(f"{target.run_dir} already holds a planned run")
    check_compatible(source.cfg, target.cfg, from_stage)
    reused = ORDER[:ORDER.index(from_stage)]
    for stage in reused:
        if stage != "plan" and not (source.stats_dir / stage / "all.done").exists() and stage in ("reduce1", "reduce2", "hvg", "metacell_prep", "finalize"):
            raise FileNotFoundError(f"stage {stage} is not finished in {source.run_dir}")
    _copy_tree(source.manifest_dir, target.manifest_dir)
    _copy_tree(source.config_dir, target.config_dir, only={"environment.txt"})  # the target keeps its own run_config.yaml
    if not (target.config_dir / "run_config.yaml").exists():
        shutil.copyfile(target.config_path, target.config_dir / "run_config.yaml")
    for stage in reused:
        for root in ROOT_OF.get(stage, []):
            base_source = (source.stats_dir if root == "stats" else source.work_dir) / stage
            base_target = (target.stats_dir if root == "stats" else target.work_dir) / stage
            if base_source.exists():
                _copy_tree(base_source, base_target)
            for marker in sorted(base_target.glob("*.done")):
                unit_dir = base_target / marker.stem
                if unit_dir.exists():
                    _rebase_unit(marker, unit_dir, source.run_dir, target.run_dir)
    if "plan" in reused:  # shards of the source (h5ad_rows / h5ad_obs) are reused as well
        shard_dir = source.work_dir / "source_shards"
        if shard_dir.exists():
            _copy_tree(shard_dir, target.work_dir / "source_shards")
        intake_dir = source.work_dir / "intake"
        if intake_dir.exists():
            _copy_tree(intake_dir, target.work_dir / "intake")
    for name in ("split_indices.json", "control_assignment.parquet", "hvg_panel.txt"):
        stage_of = {"split_indices.json": "reduce2", "control_assignment.parquet": "reduce2", "hvg_panel.txt": "hvg"}[name]
        if stage_of in reused:
            _link_or_copy(source.splits_dir / f"{source.name}_{name}", target.splits_dir / f"{target.name}_{name}")
    for part_dir in source.processed_dir.glob("*"):
        stage_dir_names = {"hvgraw": "map3", "p9": "map3", "norm": "map3", "allsplits_metacell_ctrlpreserved": "metacell"}
        if part_dir.is_dir() and any(part_dir.name.endswith(key) and stage in reused for key, stage in stage_dir_names.items()):
            _copy_tree(part_dir, target.processed_dir / part_dir.name.replace(source.name, target.name, 1))
    # every reused unit must still verify in the new run
    for stage in reused:
        for root in ROOT_OF.get(stage, []):
            base = (target.stats_dir if root == "stats" else target.work_dir) / stage
            for marker in sorted(base.glob("*.done")) if base.exists() else []:
                verify_marker(marker, marker.with_name(marker.stem), target.run_dir)
    target.logger.info(f"[branch] {target.run_dir.name} reuses {', '.join(reused)} from {source.run_dir.name}; run `{from_stage}` onward")
    return target
