#!/usr/bin/env python3
"""Verify a DS-MSP rig manifest's inputs and reproduce its calibration in a new directory.

Usage:
  python tools/replay_rig_manifest.py /path/to/calibration_manifest.json --output /path/to/replay

The original output directory is never reused.  Before launching DS-MSP, every recorded input
file is SHA-256 checked.  Afterward the effective observation/target digests and deterministic
MC-Calib output files are compared with the original run.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ds_msp.rig.provenance import MANIFEST_NAME, sha256_file  # noqa: E402

_DYNAMIC_OUTPUT_KEYS = {
    "calibration_manifest_json", "calibration_report_json", "calibration_report_txt",
    "report_json", "report_txt", "interactive_html", "mccalib_output_dir",
    "detection_images", "reprojection_images",
}


def verify_inputs(manifest: dict) -> list[str]:
    errors = []
    for record in manifest.get("inputs", {}).get("files", []):
        path = record.get("path")
        expected = record.get("sha256")
        existed = record.get("exists")
        if not path:
            errors.append(f"recorded input has no path: {record!r}")
        elif existed is False:
            if os.path.exists(path):
                errors.append(f"previously absent input now exists: {path}")
        elif not expected:
            errors.append(f"recorded input has no digest: {path}")
        elif not os.path.isfile(path):
            errors.append(f"recorded input is missing: {path}")
        else:
            actual = sha256_file(path)
            if actual != expected:
                errors.append(f"recorded input changed: {path} ({actual} != {expected})")
    return errors


def replay_args(recorded: list[str], output: str) -> list[str]:
    """Replace only output/display arguments; preserve every estimator-affecting argument."""
    args = []
    is_config = "--config" in recorded
    i = 0
    while i < len(recorded):
        arg = recorded[i]
        if arg in ("--save-dir", "--report") and i + 1 < len(recorded):
            i += 2
            continue
        if arg == "--set" and i + 1 < len(recorded):
            value = recorded[i + 1]
            if value.split("=", 1)[0].strip() == "save_path":
                i += 2
                continue
            args.extend([arg, value])
            i += 2
            continue
        args.append(arg)
        i += 1
    if is_config:
        args.extend(["--set", f"save_path={output}"])
    else:
        args.extend(["--save-dir", output])
    args.extend(["--report", os.path.join(output, "report.html")])
    if "--quiet" not in args:
        args.append("--quiet")
    if "--no-webviewer" not in args:
        args.append("--no-webviewer")
    return args


def compare_manifests(original: dict, reproduced: dict) -> list[str]:
    errors = []
    for key in ("observations", "target_geometry"):
        before = original.get("inputs", {}).get(key, {}).get("sha256")
        after = reproduced.get("inputs", {}).get(key, {}).get("sha256")
        if before != after:
            errors.append(f"effective {key} digest changed: {after} != {before}")
    old_outputs = original.get("outputs", {})
    new_outputs = reproduced.get("outputs", {})
    comparable = (set(old_outputs) & set(new_outputs)) - _DYNAMIC_OUTPUT_KEYS
    for key in sorted(comparable):
        before, after = old_outputs[key].get("sha256"), new_outputs[key].get("sha256")
        if before and after and before != after:
            errors.append(f"output {key!r} changed: {after} != {before}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", help="original calibration_manifest.json")
    ap.add_argument("--output", help="new, empty output directory (required unless --verify-only)")
    ap.add_argument("--verify-only", action="store_true",
                    help="check recorded source files but do not run calibration")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_file():
        print(f"REPLAY REFUSED: manifest not found: {manifest_path}", file=sys.stderr)
        print(
            "Pass the actual calibration_manifest.json created in your calibration "
            "output directory (for example, find . -name calibration_manifest.json -type f).",
            file=sys.stderr,
        )
        return 2
    try:
        original = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"REPLAY REFUSED: cannot read manifest {manifest_path}: {exc}", file=sys.stderr)
        return 2
    errors = verify_inputs(original)
    if errors:
        for error in errors:
            print(f"REPLAY INPUT ERROR: {error}", file=sys.stderr)
        return 2
    print(f"inputs verified: {len(original.get('inputs', {}).get('files', []))} files")
    if args.verify_only:
        return 0
    if not args.output:
        ap.error("--output is required unless --verify-only is used")
    output = os.path.abspath(args.output)
    if os.path.exists(output) and (not os.path.isdir(output) or os.listdir(output)):
        print(f"REPLAY REFUSED: output path is not an empty directory: {output}",
              file=sys.stderr)
        return 2
    os.makedirs(output, exist_ok=True)

    recorded = original.get("invocation", {}).get("replay_argv") or []
    if not recorded:
        print("REPLAY REFUSED: manifest has no replay_argv", file=sys.stderr)
        return 2
    cmd = [sys.executable, "-m", "ds_msp.rig.cli", *replay_args(recorded, output)]
    print(f"replaying: {shlex.join(cmd)}")
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    if completed.returncode:
        return completed.returncode
    new_path = Path(output) / MANIFEST_NAME
    if not new_path.is_file():
        print(f"REPLAY ERROR: reproduced manifest missing: {new_path}", file=sys.stderr)
        return 2
    reproduced = json.loads(new_path.read_text(encoding="utf-8"))
    errors = compare_manifests(original, reproduced)
    if errors:
        for error in errors:
            print(f"REPLAY MISMATCH: {error}", file=sys.stderr)
        return 1
    print(f"reproduction verified: {new_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
