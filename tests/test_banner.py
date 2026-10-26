"""Start-up banner (src/report/banner.py): ASCII fallback is pure ASCII, no line is wider than the terminal (80 and 66 columns, block and ASCII art),
every SPORE_FACTS entry has <= 64 characters, SPORE_NO_BANNER=1 silences it, the global random state is untouched, the CLI prints it once.
The once-per-run and no-banner-in-spawned-workers checks on a real run are in test_workers_spawn.py.
Run: python -B tests\\test_banner.py"""
import contextlib
import io
import os
import random
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "engine"))
sys.path.insert(0, str(HERE.parent))

from src.report import banner  # noqa: E402
import cli  # noqa: E402

RUN_NAME = "spore_20261021_p5000_prottahoe_v800t1600_seed42_tnall_xl"


def text_stream(encoding):
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict")


def test_ascii_fallback_is_pure_ascii():
    for width in (80, 66):
        text = banner.render_banner("4.0", run=RUN_NAME, mode="production", workers=3, stream=text_stream("ascii"), width=width)
        text.encode("ascii")  # raises on any non-ASCII character
        assert "spore fact:" in text and "█" not in text and "_____" in text
    stream = text_stream("ascii")
    banner.print_banner("4.0", run=RUN_NAME, mode="compat-v3", workers=1, stream=stream)
    stream.flush()
    assert stream.buffer.getvalue().decode("ascii").count("spore fact:") == 1


def test_no_line_wider_than_the_terminal():
    for encoding in ("utf-8", "ascii"):
        for width in (80, 66, 40):
            for index in range(len(banner.SPORE_FACTS)):
                text = banner.render_banner("4.0", run=RUN_NAME, mode="production", workers=12, stream=text_stream(encoding), width=width, fact_index=index)
                widest = max(len(line) for line in text.split("\n"))
                assert widest <= width, f"{encoding} width {width} fact {index}: line of {widest} characters"
    wide = banner.render_banner("4.0", run=None, stream=text_stream("utf-8"), width=80)
    assert "█" in wide, "block art expected on a UTF-8 terminal of 80 columns"
    mid = banner.render_banner("4.0", run=None, stream=text_stream("utf-8"), width=66)
    assert "█" in mid, "block art (65 columns with the indent) must still fit 66 columns"
    narrow = banner.render_banner("4.0", run=RUN_NAME, stream=text_stream("utf-8"), width=40)
    assert narrow.count("\n") == 1, "one trimmed line below the ASCII art width"


def test_fact_lengths_and_wrapping():
    assert banner.SPORE_FACTS and all(len(fact) <= 64 for fact in banner.SPORE_FACTS), [len(fact) for fact in banner.SPORE_FACTS]
    for index, fact in enumerate(banner.SPORE_FACTS):  # a wrapped fact keeps all its words
        text = banner.render_banner("4.0", run=None, stream=text_stream("utf-8"), width=66, fact_index=index)
        assert all(word in text for word in fact.split()), fact


def test_no_banner_variable_silences_it():
    saved = os.environ.get("SPORE_NO_BANNER")
    try:
        for value in ("1", "true", "YES"):
            os.environ["SPORE_NO_BANNER"] = value
            stream = text_stream("utf-8")
            banner.print_banner("4.0", run=RUN_NAME, stream=stream)
            stream.flush()
            assert stream.buffer.getvalue() == b"", value
        os.environ["SPORE_NO_BANNER"] = "0"
        stream = text_stream("utf-8")
        banner.print_banner("4.0", run=RUN_NAME, stream=stream)
        stream.flush()
        assert stream.buffer.getvalue() != b""
    finally:
        os.environ.pop("SPORE_NO_BANNER", None)
        if saved is not None:
            os.environ["SPORE_NO_BANNER"] = saved


def test_global_random_state_untouched():
    import numpy as np
    random.seed(12345)
    np.random.seed(12345)
    before_python, before_numpy = random.getstate(), np.random.get_state()[1].copy()
    for _ in range(5):
        banner.print_banner("4.0", run=RUN_NAME, mode="production", workers=2, stream=text_stream("utf-8"))
    assert random.getstate() == before_python and (np.random.get_state()[1] == before_numpy).all()
    expected = random.Random(12345).random()
    assert random.random() == expected, "the next global draw must be the one a never-printed run would have drawn"


def test_cli_banner_once_and_silenced():
    script = HERE.parent / "spore_xl.py"
    environment = {key: value for key, value in os.environ.items() if key != "SPORE_NO_BANNER"}
    shown = subprocess.run([sys.executable, "-B", str(script), "--help"], capture_output=True, text=True, env=environment)
    assert shown.returncode == 0 and shown.stdout.count("spore fact:") == 1, shown.stdout[-600:] + shown.stderr[-600:]
    silent = subprocess.run([sys.executable, "-B", str(script), "--help"], capture_output=True, text=True, env=dict(environment, SPORE_NO_BANNER="1"))
    assert silent.returncode == 0 and "spore fact:" not in silent.stdout and "usage:" in silent.stdout


def test_show_banner_passes_run_mode_workers():
    class StubRun:
        name, mode = RUN_NAME, "compat-v3"
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        cli.show_banner(StubRun(), 4)
    text = captured.getvalue()
    assert RUN_NAME in text and "compat-v3" in text and "workers 4 of" in text and f"v{cli.VERSION}" in text, text
    assert "interactive" in text or "SLURM job" in text, "the banner says where the command runs"
    slurm = banner.render_banner("4.0", run=RUN_NAME, mode="production", workers=2, stream=text_stream("utf-8"), width=80, context="SLURM job 4242")
    assert "SLURM job 4242" in slurm and max(len(line) for line in slurm.split("\n")) <= 80


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"    {name} ok")
    print("PASS test_banner")
