"""``ds-msp studio``: serve DS-MSP Studio over result folders.

Examples::

  ds-msp studio --root ./runs                 # http://127.0.0.1:8765, opens a browser
  ds-msp studio --root ./runs --no-browser --port 9000
  ds-msp studio --root ./runs --host 0.0.0.0   # prints a URL with an access token
  ds-msp studio --open ./runs/rig_2026_11_09  # opens straight on that one result
"""
from __future__ import annotations

import argparse
import secrets
import socket
import sys
import webbrowser
from pathlib import Path
from typing import Optional, Sequence, Tuple
from urllib.parse import quote

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _parser(prog: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="Open DS-MSP Studio, a browser view of "
                                 "rig calibration results.")
    ap.add_argument("--root", action="append", metavar="DIR",
                    help="folder to search for result folders (repeatable; default: current "
                         "folder)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to bind (default 127.0.0.1; any other address needs a token)")
    ap.add_argument("--open", metavar="DIR", dest="open_dir",
                    help="one result folder to open directly (its parent folder is served)")
    ap.add_argument("--port", type=int, default=8765,
                    help="port (default 8765; the next free one is used when it is taken)")
    ap.add_argument("--token", default=None,
                    help="access token (default: generated when --host is not local)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser")
    ap.add_argument("--max-depth", type=int, default=6,
                    help="folder levels to search below each root (default 6)")
    return ap


def _free_port(host: str, port: int, tries: int = 20) -> int:
    for p in range(port, port + tries):
        with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as sk:
            try:
                sk.bind((host, p))
                return p
            except OSError:
                continue
    raise OSError(f"no free port in {port}-{port + tries - 1}")


def _result_target(open_dir: str) -> Tuple[str, str]:
    """(root to serve, run id) for one result folder: its parent is served."""
    folder = Path(open_dir).expanduser().resolve()
    if not (folder / "calibration_report.json").is_file():
        raise NotADirectoryError(f"{folder} holds no calibration_report.json")
    return str(folder.parent), f"0:{folder.name}"


def serve(roots: Sequence[str], *, host: str = "127.0.0.1", port: int = 8765,
          token: Optional[str] = None, open_browser: bool = True, run_id: Optional[str] = None,
          max_depth: int = 6) -> int:
    """Serve Studio until Ctrl+C. Prints the link (to ``run_id``'s page when given)."""
    try:
        import uvicorn

        from .server import create_app
    except ImportError as exc:  # fastapi / uvicorn missing
        print(f"DS-MSP Studio needs the ui extra: pip install \"ds-msp[ui]\" ({exc})",
              file=sys.stderr)
        return 2
    if token is None and host not in _LOCAL_HOSTS:
        token = secrets.token_urlsafe(18)
    try:
        app = create_app(roots, token=token, max_depth=max_depth)
    except NotADirectoryError as exc:
        print(f"not a folder: {exc}", file=sys.stderr)
        return 2
    try:
        port = _free_port(host, port)
    except OSError as exc:
        print(f"cannot start DS-MSP Studio: {exc}", file=sys.stderr)
        return 2
    shown_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{shown_host}:{port}/" + (f"?token={token}" if token else "")
    if run_id is not None:
        url += f"#/run/{quote(run_id, safe='')}"
    print(f"DS-MSP Studio: {url}")
    print("Press Ctrl+C to stop.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    return 0


def run(argv: Sequence[str] | None = None, prog: str = "ds-msp studio") -> int:
    args = _parser(prog).parse_args(list(sys.argv[1:] if argv is None else argv))
    roots = args.root or ["."]
    run_id = None
    if args.open_dir:
        try:
            root, run_id = _result_target(args.open_dir)
        except NotADirectoryError as exc:
            print(f"not a result folder: {exc}", file=sys.stderr)
            return 2
        roots = [root]
    return serve(roots, host=args.host, port=args.port, token=args.token,
                 open_browser=not args.no_browser, run_id=run_id, max_depth=args.max_depth)


def main() -> None:
    raise SystemExit(run())
