"""``ds-msp studio``: serve DS-MSP Studio over result folders.

Examples::

  ds-msp studio --root ./runs                 # http://127.0.0.1:8765, opens a browser
  ds-msp studio --root /data/calib --no-browser --port 9000
  ds-msp studio --root /data/calib --host 0.0.0.0   # prints a URL with an access token
"""
from __future__ import annotations

import argparse
import secrets
import sys
import webbrowser
from typing import Sequence

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _parser(prog: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="Open DS-MSP Studio, a browser view of "
                                 "rig calibration results.")
    ap.add_argument("--root", action="append", metavar="DIR",
                    help="folder to search for result folders (repeatable; default: current "
                         "folder)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to bind (default 127.0.0.1; any other address needs a token)")
    ap.add_argument("--port", type=int, default=8765, help="port (default 8765)")
    ap.add_argument("--token", default=None,
                    help="access token (default: generated when --host is not local)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser")
    ap.add_argument("--max-depth", type=int, default=6,
                    help="folder levels to search below each root (default 6)")
    return ap


def run(argv: Sequence[str] | None = None, prog: str = "ds-msp studio") -> int:
    args = _parser(prog).parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        import uvicorn

        from .server import create_app
    except ImportError as exc:  # fastapi / uvicorn missing
        print(f"DS-MSP Studio needs the ui extra: pip install \"ds-msp[ui]\" ({exc})",
              file=sys.stderr)
        return 2
    roots = args.root or ["."]
    token = args.token
    if token is None and args.host not in _LOCAL_HOSTS:
        token = secrets.token_urlsafe(18)
    try:
        app = create_app(roots, token=token, max_depth=args.max_depth)
    except NotADirectoryError as exc:
        print(f"not a folder: {exc}", file=sys.stderr)
        return 2
    shown_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{shown_host}:{args.port}/" + (f"?token={token}" if token else "")
    print(f"DS-MSP Studio: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main() -> None:
    raise SystemExit(run())
