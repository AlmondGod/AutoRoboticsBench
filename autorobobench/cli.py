from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Iterable

from autorobobench.schema import SuiteSpec
from autorobobench.scoring import score_suite


def main() -> None:
    parser = argparse.ArgumentParser(prog="autorobobench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    describe = sub.add_parser("describe", help="Print a compact track summary.")
    describe.add_argument("--config", default="configs/autorobobench_v0.json")

    score = sub.add_parser("score", help="Score an AutoroboBench result JSON.")
    score.add_argument("--config", default="configs/autorobobench_v0.json")
    score.add_argument("--results", required=True)
    score.add_argument("--out", default="")

    hash_manifest = sub.add_parser("hash-manifest", help="Hash immutable files listed by the config.")
    hash_manifest.add_argument("--config", default="configs/autorobobench_v0.json")
    hash_manifest.add_argument("--root", default=".")
    hash_manifest.add_argument("--out", default="")

    args = parser.parse_args()
    if args.cmd == "describe":
        _describe(args)
    elif args.cmd == "score":
        _score(args)
    elif args.cmd == "hash-manifest":
        _hash_manifest(args)
    else:
        raise SystemExit(f"unknown command: {args.cmd}")


def _describe(args: argparse.Namespace) -> None:
    spec = SuiteSpec.from_path(args.config)
    payload = {
        "version": spec.version,
        "total_points": spec.total_points,
        "tracks": [
            {
                "id": track.id,
                "name": track.name,
                "phase": track.phase,
                "weight": track.weight,
                "primary_metric": track.primary_metric,
                "starter_metric": track.starter_metric,
                "reference_metric": track.reference_metric,
            }
            for track in spec.tracks
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _score(args: argparse.Namespace) -> None:
    spec = SuiteSpec.from_path(args.config)
    results = json.loads(Path(args.results).read_text())
    payload = score_suite(spec, results)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text, end="")


def _hash_manifest(args: argparse.Namespace) -> None:
    spec = SuiteSpec.from_path(args.config)
    root = Path(args.root).resolve()
    globs = sorted({pattern for track in spec.tracks for pattern in track.immutable_globs})
    files = list(_iter_matching_files(root, globs))
    payload = {
        "version": spec.version,
        "root": str(root),
        "immutable_globs": globs,
        "files": [
            {
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text, end="")


def _iter_matching_files(root: Path, patterns: Iterable[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _is_generated_python_cache(path):
            continue
        rel = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) for pattern in patterns):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path


def _is_generated_python_cache(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
