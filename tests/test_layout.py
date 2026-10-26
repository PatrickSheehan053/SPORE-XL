"""Layout rules of the SPORE-XL repository (no pipeline run, about 20 seconds):
 - the top level holds exactly the agreed entries
 - no file or folder name contains "sanity"; no spore_local.py / spore_cluster.py / MANIFEST.txt / test_data / v3-format configs
 - no dataset-type files (.h5ad .h5 .tar .gz .parquet .npz .npy .pkl); tests/ref (built reference material) is ignored
 - every .py parses, and no code uses a name from the old layout (spore.py, `src.phaseNN`, `src.utils`, `import spore_local`, `from clone`)
 - every module of src/ imports (one subprocess)
Run: python -B tests\\test_layout.py"""
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parent.parent
TOP = {"README.md", "CHANGELOG.md", "LICENSE", ".gitignore", "requirements.txt", "requirements_pinned.txt", "spore_xl.py", "src", "slurm", "configs",
       "protected_genes", "tests", "tools", "docs", "__pycache__"}
FORBIDDEN_NAMES = {"spore_local.py", "spore_cluster.py", "MANIFEST.txt", "test_data", "spore.py"}
DATA_SUFFIXES = {".h5ad", ".h5", ".tar", ".gz", ".parquet", ".npz", ".npy", ".pkl", ".loom", ".rds"}
OLD_NAMES = [r"\bspore\.py\b", r"\bsrc\.(phase\d\d_\w+|utils|plotting)\b", r"^\s*import spore_local\b", r"\bfrom clone\b", r"\bimport clone\b", r"\bfrom xl\b"]


def repo_files():
    for path in sorted(REPO.rglob("*")):
        relative = path.relative_to(REPO).as_posix()
        if ".git/" in relative + "/" or relative.startswith(".git") or relative.startswith("tests/ref/") or "__pycache__" in path.parts:
            continue
        yield path, relative


def test_top_level_and_names():
    extra = sorted(entry.name for entry in REPO.iterdir() if entry.name not in TOP and entry.name != ".git")
    assert not extra, f"unexpected top-level entries: {extra}"
    for path, relative in repo_files():
        assert "sanity" not in relative.lower(), f"'sanity' in a path: {relative}"
        assert path.name not in FORBIDDEN_NAMES, f"forbidden file or folder: {relative}"
        assert not re.search(r"^configs/spore_config_", relative), f"v3-format config: {relative}"
        if path.is_file():
            assert path.suffix.lower() not in DATA_SUFFIXES, f"dataset-type file in the repository: {relative}"
    print("    top level, names, no datasets: ok")


def test_no_old_layout_names_in_code():
    problems = []
    for path, relative in repo_files():
        if path.suffix == ".py" and relative != "tests/test_layout.py":
            text = path.read_text(encoding="utf-8")
            ast.parse(text)
            for pattern in OLD_NAMES:
                for match in re.finditer(pattern, text, flags=re.MULTILINE):
                    line_number = text.count("\n", 0, match.start()) + 1
                    line = text.splitlines()[line_number - 1]
                    problems.append(f"{relative}:{line_number}: {line.strip()[:100]}")
    assert not problems, "old-layout names left:\n" + "\n".join(problems[:20])
    print("    every .py parses; no old-layout names")


def test_every_module_imports():
    engine = REPO / "src" / "engine"
    modules = sorted(p.stem for p in engine.glob("*.py")) + sorted("adapted_phases." + p.stem for p in (engine / "adapted_phases").glob("*.py") if p.stem != "__init__")
    modules += [f"src.{folder}.{p.stem}" for folder in ("phases", "intake", "report") for p in sorted((REPO / "src" / folder).glob("*.py"))]
    code = ("import sys; sys.dont_write_bytecode = True; sys.path.insert(0, %r); import paths; paths.add_spore_to_path(); import importlib\n"
            "for name in %r:\n    importlib.import_module(name)\nprint(len(%r), 'modules imported')\n") % (str(engine), modules, modules)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", PYTHONUTF8="1", SPORE_NO_BANNER="1")
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO))
    assert result.returncode == 0, result.stderr[-1500:]
    print("    " + result.stdout.strip())


if __name__ == "__main__":
    test_top_level_and_names()
    test_no_old_layout_names_in_code()
    test_every_module_imports()
    print("PASS test_layout")
