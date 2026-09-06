#!/usr/bin/env python3
"""Validate one completed ZStackAnalysis run without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zstack_analysis.validation import validate_run_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    report = validate_run_directory(args.run_dir)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
