#!/usr/bin/env python3
"""Cleanup repository: remove Python caches and optionally .venv and models directory.

Usage:
  python scripts/cleanup_repo.py [--remove-venv] [--remove-models] [--dry-run]
"""
import argparse
from pathlib import Path
import shutil


def remove_path(p: Path, dry_run: bool = False):
    if not p.exists():
        return
    if dry_run:
        print(f"DRYRUN: would remove {p}")
        return
    if p.is_dir():
        shutil.rmtree(p)
        print(f"Removed directory: {p}")
    else:
        p.unlink()
        print(f"Removed file: {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove-venv", action="store_true", help="Remove .venv directory")
    parser.add_argument("--remove-models", action="store_true", help="Remove models directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    print(f"Repository root: {root}")

    # Remove __pycache__ dirs and .pyc files
    for p in root.rglob("__pycache__"):
        remove_path(p, dry_run=args.dry_run)

    for p in root.rglob("*.pyc"):
        remove_path(p, dry_run=args.dry_run)

    if args.remove_venv:
        venv = root / ".venv"
        remove_path(venv, dry_run=args.dry_run)

    if args.remove_models:
        models = root / "models"
        remove_path(models, dry_run=args.dry_run)

    print("Done. Make sure .gitignore is added and then run 'git status' and commit the changes.")


if __name__ == "__main__":
    main()
