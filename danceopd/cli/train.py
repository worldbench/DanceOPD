"""Command-line entry for DanceOPD training."""
from __future__ import annotations

import argparse

from danceopd.core.config import apply_overrides, load_config
from danceopd.core.engine import DanceOPDEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DanceOPD student.")
    parser.add_argument("--config", required=True, help="YAML config file.")
    parser.add_argument("--set", action="append", default=[], help="Override config values, e.g. training.lr=1e-4")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without loading models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    DanceOPDEngine(cfg, dry_run=args.dry_run).run()


if __name__ == "__main__":
    main()
