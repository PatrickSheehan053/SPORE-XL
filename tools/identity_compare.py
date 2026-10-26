"""Compares every file of two run folders by sha256 (task 4, identity proof). Reads only; writes the --out files (new names, never overwrites).

  python -B identity_compare.py --a <run folder A> --b <run folder B> --out <prefix> [--pair <extra folder A> <extra folder B>]...

Writes <prefix>_files.tsv (one row per relative path: sha256 of A and B, status) and <prefix>_diffs.txt (for each differing text file: whether it
differs only by the absolute run / work folder paths, and the first differing lines after that normalization).
status: identical | only_a | only_b | differs. A differing text file is further marked path_only when it is identical once the folder paths
of the two runs are replaced by <RUN>.
"""
import argparse
import difflib
import hashlib
import sys
from pathlib import Path

sys.dont_write_bytecode = True
TEXT_SUFFIXES = {".txt", ".md", ".json", ".tsv", ".csv", ".yaml", ".yml", ".log", ".out", ".done", ".sh", ".py"}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def walk(root):
    return {path.relative_to(root).as_posix(): path for path in sorted(root.rglob("*")) if path.is_file()}


def variants(folder):
    resolved = Path(folder).resolve()
    forms = {str(resolved), resolved.as_posix(), str(resolved).replace("\\", "\\\\")}
    return sorted(forms, key=len, reverse=True)


def normalize(text, folders):
    for folder in folders:
        for form in variants(folder):
            text = text.replace(form, "<RUN>")
    return text


def compare_pair(root_a, root_b, tag, rows, diffs, folders_a, folders_b):
    files_a, files_b = walk(root_a), walk(root_b)
    for rel in sorted(set(files_a) | set(files_b)):
        key = f"{tag}{rel}"
        if rel not in files_b:
            rows.append((key, sha256_of(files_a[rel]), "", "only_a"))
        elif rel not in files_a:
            rows.append((key, "", sha256_of(files_b[rel]), "only_b"))
        else:
            sha_a, sha_b = sha256_of(files_a[rel]), sha256_of(files_b[rel])
            if sha_a == sha_b:
                rows.append((key, sha_a, sha_b, "identical"))
                continue
            status = "differs"
            if files_a[rel].suffix.lower() in TEXT_SUFFIXES:
                text_a = normalize(files_a[rel].read_text(encoding="utf-8", errors="replace"), folders_a)
                text_b = normalize(files_b[rel].read_text(encoding="utf-8", errors="replace"), folders_b)
                if text_a == text_b:
                    status = "path_only"
                else:
                    lines = [line[:240] for line in difflib.unified_diff(text_a.splitlines(), text_b.splitlines(), lineterm="", n=0) if not line.startswith(("---", "+++"))]
                    diffs.append(f"##### {key}  ({len(lines)} changed lines)\n" + "\n".join(lines[:12]))
            rows.append((key, sha_a, sha_b, status))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"))
    args = parser.parse_args()
    out_files, out_diffs = Path(args.out + "_files.tsv"), Path(args.out + "_diffs.txt")
    if out_files.exists() or out_diffs.exists():
        raise SystemExit("output exists; choose a new --out prefix")
    pairs = [("", Path(args.a), Path(args.b))] + [(f"[extra{index}]/", Path(a), Path(b)) for index, (a, b) in enumerate(args.pair, 1)]
    folders_a, folders_b = [pair[1] for pair in pairs], [pair[2] for pair in pairs]
    rows, diffs = [], []
    for tag, root_a, root_b in pairs:
        compare_pair(root_a, root_b, tag, rows, diffs, folders_a, folders_b)
    out_files.write_text("path\tsha256_a\tsha256_b\tstatus\n" + "\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8")
    out_diffs.write_text("\n\n".join(diffs) + "\n", encoding="utf-8")
    counts = {}
    for row in rows:
        counts[row[3]] = counts.get(row[3], 0) + 1
    print("files:", len(rows), counts)
    return 0 if set(counts) <= {"identical", "path_only"} else 1


if __name__ == "__main__":
    sys.exit(main())
