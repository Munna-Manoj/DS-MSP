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
    if args[:1] == ["export"]:
        from .export_cli import run as run_export

        return run_export(args[1:], prog="ds-msp export")
    if args[:1] == ["ldc"]:
        from .ldc_cli import run as run_ldc

        return run_ldc(args[1:], prog="ds-msp ldc")
    if args[:1] == ["studio"]:
        from .studio.cli import run as run_studio

        return run_studio(args[1:], prog="ds-msp studio")

    parser = argparse.ArgumentParser(
        prog="ds-msp",
        description="DS-MSP camera calibration and projection tools",
    )
    parser.add_argument("command", nargs="?", choices=["export", "ldc", "lut", "studio"])
    parser.print_help()
    return 0 if not args else 2


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
