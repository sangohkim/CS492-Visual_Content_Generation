#!/usr/bin/env python3
"""Move `run_<datetime>` directories from `outputs_db` to
`outputs_db-color-no_sks` when the `generation_args.json` file inside
contains a `lora_path` whose second-to-last path component equals the
configured match string (default: `nupjuki-new_data-color`).

Usage examples:
  python StoryDiffusion/scripts/move_runs_by_lora.py --dry-run
  python StoryDiffusion/scripts/move_runs_by_lora.py

The script safely handles name collisions by appending a numeric suffix.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PurePath
import sys


def find_run_dirs(src_dir: Path):
    for p in sorted(src_dir.iterdir()):
        if p.is_dir() and p.name.startswith("run_"):
            yield p


def safe_move(src: Path, dst_dir: Path, dry_run: bool = True):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        # find a free name
        i = 1
        while True:
            candidate = dst_dir / f"{src.name}_{i}"
            if not candidate.exists():
                dst = candidate
                break
            i += 1
    if dry_run:
        print(f"DRY-RUN: would move: {src} -> {dst}")
        return dst
    else:
        shutil.move(str(src), str(dst))
        print(f"Moved: {src} -> {dst}")
        return dst


def check_lora_second_last(lora_path_str: str, match: str) -> bool:
    if not lora_path_str:
        return False
    p = PurePath(lora_path_str)
    parts = p.parts
    if len(parts) < 2:
        return False
    return parts[-2] == match


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Move run_* dirs whose generation_args.json references a matching LoRA path."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("./StoryDiffusion/outputs_db"),
        help="Source folder containing run_<datetime> directories",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("./StoryDiffusion/outputs_db-basic"),
        help="Destination folder to move matching run directories into",
    )
    parser.add_argument(
        "--match",
        type=str,
        default="nupjuki-new_data",
        help="Match the second-to-last path component of `lora_path` against this string",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be moved without making changes")
    args = parser.parse_args(argv)

    src_dir = args.src.expanduser().resolve()
    dst_dir = args.dst.expanduser().resolve()

    if not src_dir.exists() or not src_dir.is_dir():
        print(f"Source directory does not exist: {src_dir}")
        return 2

    moved = []
    skipped = []

    for run_dir in find_run_dirs(src_dir):
        gen_file = run_dir / "generation_args.json"
        if not gen_file.exists():
            skipped.append((run_dir, "missing generation_args.json"))
            continue
        try:
            with gen_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            skipped.append((run_dir, f"failed to read json: {e}"))
            continue

        lora_path = data.get("lora_path") or data.get("lora") or ""
        if check_lora_second_last(lora_path, args.match):
            safe_move(run_dir, dst_dir, dry_run=args.dry_run)
            moved.append(run_dir)
        else:
            skipped.append((run_dir, "no match"))

    print("\nSummary:")
    print(f"  Matches moved: {len(moved)}")
    print(f"  Skipped: {len(skipped)}")
    if skipped:
        print("  Skipped details (first 10):")
        for r, reason in skipped[:10]:
            print(f"    {r}  -- {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
