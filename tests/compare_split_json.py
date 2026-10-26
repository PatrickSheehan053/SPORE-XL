"""Compare split JSONs (SPORE format): sizes, label-set overlap and byte equality. Used for the RPE1 graduation run:
the July 2026 split, SPORE v3.0 (reference) and SPORE-XL.
  python -B tests\\compare_split_json.py --a <json> --b <json> [--c <json>]
Exit 0 only when every pair given is byte-identical; the overlap table is printed either way (July vs v3.0 may differ legitimately)."""
import argparse
import json
import sys
from pathlib import Path


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: set(data[f"{key}_labels"]) for key in ("train", "val", "test")}, data.get("seed"), Path(path).read_bytes()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--c", default=None)
    args = parser.parse_args(argv)
    named = [("a", args.a), ("b", args.b)] + ([("c", args.c)] if args.c else [])
    loaded = {name: load(path) for name, path in named}
    for name, path in named:
        sets, seed, _ = loaded[name]
        print(f"{name}: {path}\n   train {len(sets['train'])} / val {len(sets['val'])} / test {len(sets['test'])}, seed {seed}")
    identical = True
    names = [name for name, _ in named]
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            sets_a, _, bytes_a = loaded[first]
            sets_b, _, bytes_b = loaded[second]
            same_bytes = bytes_a == bytes_b
            identical &= same_bytes
            overlaps = {key: (len(sets_a[key] & sets_b[key]), len(sets_a[key]), len(sets_b[key])) for key in sets_a}
            print(f"{first} vs {second}: byte-identical {same_bytes}; label overlap (shared, in {first}, in {second}): {overlaps}")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
