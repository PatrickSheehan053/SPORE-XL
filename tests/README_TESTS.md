# Tests

All tests are plain scripts (`python -B tests/<name>.py`, pytest-collectable). They write only into `XL_TEST_TMP` (default: the system temp folder) and never delete.

## One-time reference material

The equivalence tests compare SPORE-XL with SPORE v3.0, so they need SPORE v3.0's own output on a synthetic 60,000-cell dataset. SPORE v3.0 is not in this repository (it is the tag `v3.0`, branch `v3-archive`); use a checkout outside it (`docs/DEVELOPER.md` section 1b):

```
git clone https://github.com/PatrickSheehan053/SPORE-XL <folder outside the repository>/spore_v3_checkout
git -C <folder outside the repository>/spore_v3_checkout checkout v3.0
python -B tests/prepare_references.py --spore-v3-dir <folder outside the repository>/spore_v3_checkout --out tests/ref     # about 15 minutes; skips what exists
set XL_TEST_DATA=tests\ref                                   # (export XL_TEST_DATA=tests/ref on Linux)
set XL_SPORE_V3_DIR=<folder outside the repository>/spore_v3_checkout     # optional: test_spore_api then also verifies the 2 SPORE files that are not shipped
```

## Running

```
python -B tests/run_all_tests.py --tag t1 --logs <new folder> --scratch <folder> --quick          # no pipeline run, about a minute
python -B tests/run_all_tests.py --tag t1 --logs <new folder> --scratch <folder>                  # everything, about an hour, one test at a time
```
Set `PYTHONHASHSEED=0` (compat mode needs it). `summary_<tag>.tsv` lists every test; re-running with the same tag skips the ones that passed.

## What is tested

| file | what |
|---|---|
| `test_stats`, `test_shard_planner`, `test_spore_api` | exact percentile and mergeable statistics; sharding reproduces the input; every imported SPORE v3.0 name still has its signature and the 12 SPORE files are byte-identical to the recorded hashes |
| `test_stage_*`, `test_equivalence`, `test_robustness` | each stage against SPORE's own intermediates; the whole pipeline against SPORE v3.0 at 1, 3 and 7 shards (26 compare checks); edge cases (empty shards, no controls, a bypassed label, label collisions, refusals) |
| `test_extract`, `test_merge_ondisk`, `test_report`, `test_cleanup`, `test_target_policy`, `test_measure`, `test_gate_protected`, `test_external_workdir`, `test_intake`, `test_intake_genes`, `test_run_command`, `test_spore_xl_sbatch`, `test_preflight_intake`, `test_banner`, `test_shipped_configs`, `test_confounding_null` | the optional tools, the Flex-tar intake and its one gene list, `spore_xl.py run` (exit codes, the gate inside it, automatic workers, SLURM banner), the cluster script (functional runs in bash), the intake RAM calibration, the start-up banner, the shipped configs, the calibrated confounding warning |
| `test_p8_fixes`, `test_production_defaults`, `test_speed_changes` | the v4 production defaults (budget panel, no ghost rescue, node manifest, aliases, lognorm hybrid, diagnostics, escaper counts), the P8 fixes, and that the speed changes give identical results |
| `test_intake_adapters` | every intake adapter on synthetic files, label repair and refusals |
| `test_workers_spawn`, `test_resume_incomplete` | pools on Linux and Windows; a killed run resumes to identical files |
| `stress/` | eleven kinds of difficult synthetic datasets with answer keys: `python -B tests/stress/make_stress_suite.py --out <new folder>` then `python -B tests/stress/run_stress_suite.py --data <folder> --work <scratch> --results <new folder> --tag s2` (`--compat` runs the compat setting; `--reference-tag` also requires identical outputs to an earlier run) |

`tests/make_synthetic.py` makes the 60k dataset (seeded); `make_synthetic_v2.py` makes the 1M dataset of the memory-flatness proof; `make_flex_synthetic.py` makes 10x Flex members for the tar intake.
