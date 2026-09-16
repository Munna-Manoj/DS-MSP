"""Unified DS-MSP command-line entry point."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def run(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # Friendly alias matching the original feature request:
    #   ds-msp --lut ...  ==  ds-msp lut ...
    if args[:1] == ["--lut"]:
        args[0] = "lut"
    if args[:1] == ["lut"]:
        from .isaac_sim.cli import run as run_lut

        return run_lut(args[1:], prog="ds-msp lut")

    parser = argparse.ArgumentParser(
        prog="ds-msp",
        description="DS-MSP camera calibration and projection tools",
    )
    parser.add_argument("command", nargs="?", choices=["lut"])
    parser.print_help()
    return 0 if not args else 2


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
