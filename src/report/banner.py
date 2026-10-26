# SPORE-XL start-up banner (graduation gift, 21 Oct 2026), v3: spore trivia only. Copied from intermediate/banner_proposal/spore_banner_v3.py
# (the Director's replacement for v2); one change in v4.0: the fact line wraps instead of being cut on a narrow terminal, and the host
# line is not indented when there is no run line.
# Block letters when the terminal can draw them, plain ASCII when it cannot (e.g. Windows PowerShell on a
# legacy code page), one trimmed line on very narrow terminals. Never raises: a banner must not be able to
# break a run. Disable with SPORE_NO_BANNER=1. Uses its own random.Random, so the global random state that
# SPORE relies on is never touched.
import datetime
import os
import random
import shutil
import socket
import sys
import textwrap

BLOCK_ART = [
    '███████╗██████╗  ██████╗ ██████╗ ███████╗    ██╗  ██╗██╗',
    '██╔════╝██╔══██╗██╔═══██╗██╔══██╗██╔════╝    ╚██╗██╔╝██║',
    '███████╗██████╔╝██║   ██║██████╔╝█████╗       ╚███╔╝ ██║',
    '╚════██║██╔═══╝ ██║   ██║██╔══██╗██╔══╝       ██╔██╗ ██║',
    '███████║██║     ╚██████╔╝██║  ██║███████╗    ██╔╝ ██╗███████╗',
    '╚══════╝╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═╝╚══════╝',
]

ASCII_ART = [
    '   _____ ____  ____  ____  ______   _  __ __',
    '  / ___// __ \\/ __ \\/ __ \\/ ____/  | |/ // /',
    '  \\__ \\/ /_/ / / / / /_/ / __/     |   // /',
    ' ___/ / ____/ /_/ / _, _/ /___    /   |/ /___',
    '/____/_/    \\____/_/ |_/_____/   /_/|_/_____/',
]

INDENT = "  "
REPO_URL = "github.com/PatrickSheehan053/SPORE-XL"
ACRONYM = "Systematic Preprocessing and Optimization for Robust Evaluation"
TAGLINE = "sharded, memory-bounded Perturb-seq preprocessing"
BLOCK_TEST_CHARS = "\u2588\u2557\u255a\u2550\u00b7"   # glyphs the block banner and separators use

# one is shown per run; keep every entry true and short (<= 64 characters, tests/test_banner.py)
SPORE_FACTS = [
    "A single giant puffball can release trillions of spores.",
    "Fungi send tens of millions of tonnes of spores aloft each year.",
    "Mushrooms can make their own wind to lift their spores.",
    "Pilobolus fires its spores with accelerations near 20,000 g.",
    "Many mushrooms launch spores with a water droplet catapult.",
    "Fungal spores can help ice crystals and raindrops form.",
    "Mycologists identify mushrooms by the colour of a spore print.",
    "Penicillin was found when a mould spore landed on a petri dish.",
]
FACT_PREFIX = "spore fact: "


def _can_encode(stream, text):
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        text.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _terminal_width():
    # SLURM logs and pipes are not terminals: get_terminal_size then uses $COLUMNS or 80
    return shutil.get_terminal_size(fallback=(80, 24)).columns


def available_cpus():
    # under SLURM the job's allocation is what counts, not the whole node
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if slurm_cpus.isdigit():
        return int(slurm_cpus)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def render_banner(version="4.0", run=None, mode=None, workers=None, stream=None, width=None, fact_index=None, context=None):
    stream = stream or sys.stdout
    width = width or _terminal_width()
    fancy = _can_encode(stream, BLOCK_TEST_CHARS)
    dot = " \u00b7 " if fancy else " | "
    block_width = len(INDENT) + max(len(line) for line in BLOCK_ART)
    ascii_width = len(INDENT) + max(len(line) for line in ASCII_ART)
    if fancy and width >= block_width:
        art = BLOCK_ART
    elif width >= ascii_width:
        art = ASCII_ART
    else:
        line = f"SPORE-XL v{version}" + (f" | {run}" if run else "")
        return (line[:width - 1] if len(line) >= width else line) + "\n"

    lines = [""] + [INDENT + line for line in art] + [""]
    lines += [INDENT + f"v{version}{dot}{TAGLINE}", INDENT + ACRONYM, INDENT + REPO_URL, ""]
    if run:
        lines.append(INDENT + f"run      {run}")
    run_facts = []
    if mode:
        run_facts.append(f"mode {mode}")
    if workers:
        run_facts.append(f"workers {workers} of {available_cpus()} CPUs")
    if context:
        run_facts.append(str(context))          # "interactive" or "SLURM job <id>"
    run_facts.append(f"host {socket.gethostname().split('.')[0]}")
    lines.append(INDENT + ("         " if run else "") + dot.strip().join(f" {item} " for item in run_facts).strip())
    lines.append(INDENT + f"started  {datetime.datetime.now():%Y-%m-%d %H:%M}")
    chooser = random.Random()   # private generator: never disturbs the global random state
    fact = SPORE_FACTS[fact_index % len(SPORE_FACTS)] if fact_index is not None else chooser.choice(SPORE_FACTS)
    room = max(width - 1 - len(INDENT) - len(FACT_PREFIX), 20)
    wrapped = textwrap.wrap(fact, room) or [fact]
    lines += [""] + [INDENT + (FACT_PREFIX if index == 0 else " " * len(FACT_PREFIX)) + part for index, part in enumerate(wrapped)] + [""]
    if not fancy:
        lines = [line.encode("ascii", "replace").decode("ascii") for line in lines]
    # keep every line inside the terminal even if a run name is long
    return "\n".join(line[:width - 1] if len(line) >= width else line for line in lines) + "\n"


def print_banner(version="4.0", run=None, mode=None, workers=None, stream=None, context=None):
    if os.environ.get("SPORE_NO_BANNER", "").strip().lower() in ("1", "true", "yes"):
        return
    stream = stream or sys.stdout
    try:
        stream.write(render_banner(version, run, mode, workers, stream, context=context))
        stream.flush()
    except Exception:
        pass


if __name__ == "__main__":
    print_banner(run="spore_20261021_p5000_prottahoe_v800t1600_seed42_tnall_xl", mode="production", workers=3)
