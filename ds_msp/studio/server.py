"""DS-MSP Studio HTTP server: a read-only JSON API over result folders, plus the bundled UI.

Needs the ``ui`` extra (``pip install "ds-msp[ui]"``). The server only reads result folders
under the roots it was started with; every requested path is resolved and checked to lie
inside one of them. When bound to anything other than localhost it requires a token, given
once as ``?token=`` and then kept in a cookie.
"""
from __future__ import annotations

import hmac
import importlib.resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..rig.config_schema import config_schema
from .runs import KNOWN_FILES, RunError, iter_run_dirs, load_run, summarize_run

API_VERSION = 1
TOKEN_COOKIE = "dsmsp_token"
TOKEN_HEADER = "x-dsmsp-token"

_FALLBACK_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>DS-MSP Studio</title>
<style>body{font:15px/1.5 system-ui,sans-serif;max-width:46rem;margin:3rem auto;padding:0 1rem}
code{background:#eee;padding:0 .3em}</style></head><body>
<h1>DS-MSP Studio</h1>
<p>The server is running, but this installation has no built user interface.</p>
<p>From a source checkout, build it with <code>cd web &amp;&amp; npm ci &amp;&amp; npm run build:studio</code>,
then restart. Released wheels ship the built interface.</p>
<p>The JSON API works without it: <a href="api/runs">api/runs</a>.</p>
</body></html>"""


def static_dir() -> Optional[Path]:
    """The bundled UI folder, or ``None`` when this installation has no built UI."""
    root = importlib.resources.files("ds_msp.studio") / "static"
    path = Path(str(root))
    return path if (path / "index.html").is_file() else None


class _Roots:
    def __init__(self, roots: Sequence[str]):
        self.paths: List[Path] = [Path(r).expanduser().resolve() for r in roots]
        for p in self.paths:
            if not p.is_dir():
                raise NotADirectoryError(str(p))

    def run_id(self, k: int, folder: Path) -> str:
        rel = folder.relative_to(self.paths[k]).as_posix()
        return f"{k}:{'' if rel == '.' else rel}"

    def resolve(self, run_id: str) -> Path:
        """Folder of ``run_id``; raises ``KeyError`` for anything outside the roots."""
        k_str, sep, rel = run_id.partition(":")
        if not sep or not k_str.isdigit() or int(k_str) >= len(self.paths):
            raise KeyError(run_id)
        root = self.paths[int(k_str)]
        folder = (root / rel).resolve()
        if folder != root and root not in folder.parents:
            raise KeyError(run_id)
        return folder


def create_app(roots: Sequence[str], *, token: Optional[str] = None, max_depth: int = 6):
    """The Studio ASGI app over result folders under ``roots``. ``token`` (when given) is
    required on every request."""
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
    from starlette.middleware.base import BaseHTTPMiddleware

    from .. import __version__

    rs = _Roots(roots)
    app = FastAPI(title="DS-MSP Studio", version=__version__, docs_url=None, redoc_url=None,
                  openapi_url="/api/openapi.json")

    if token:
        secret: str = token

        class _TokenGate(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                given = (request.query_params.get("token") or request.headers.get(TOKEN_HEADER)
                         or request.cookies.get(TOKEN_COOKIE) or "")
                if not hmac.compare_digest(given.encode(), secret.encode()):
                    return JSONResponse({"detail": "missing or wrong token"}, status_code=401)
                response = await call_next(request)
                if request.query_params.get("token"):
                    response.set_cookie(TOKEN_COOKIE, secret, httponly=True, samesite="strict")
                return response

        app.add_middleware(_TokenGate)

    def _folder(run_id: str) -> Path:
        try:
            folder = rs.resolve(run_id)
        except KeyError:
            raise HTTPException(404, "unknown run") from None
        if not (folder / "calibration_report.json").is_file():
            raise HTTPException(404, "unknown run")
        return folder

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "version": __version__, "api_version": API_VERSION,
                "ui_bundled": static_dir() is not None}

    @app.get("/api/roots")
    def list_roots() -> List[Dict[str, Any]]:
        return [{"index": k, "name": p.name or str(p)} for k, p in enumerate(rs.paths)]

    @app.get("/api/runs")
    def list_runs() -> List[Dict[str, Any]]:
        out = []
        for k, root in enumerate(rs.paths):
            for folder in iter_run_dirs(root, max_depth=max_depth):
                try:
                    summary = summarize_run(folder)
                except RunError as exc:
                    summary = {"name": folder.name, "error": str(exc)}
                rel = folder.relative_to(root).as_posix()
                out.append({"id": rs.run_id(k, folder), "root": k,
                            "path": "" if rel == "." else rel, **summary})
        out.sort(key=lambda r: (r.get("created_utc") or "", r["id"]), reverse=True)
        return out

    @app.get("/api/run")
    def get_run(id: str = Query(...)) -> Dict[str, Any]:  # noqa: A002 - query name
        folder = _folder(id)
        try:
            doc = load_run(folder)
        except RunError as exc:
            raise HTTPException(422, str(exc)) from None
        doc["id"] = id
        return doc

    @app.get("/api/run/file")
    def get_run_file(id: str = Query(...), name: str = Query(...)):  # noqa: A002
        if name not in KNOWN_FILES:
            raise HTTPException(404, "unknown file")
        path = _folder(id) / name
        if not path.is_file():
            raise HTTPException(404, "file not in this run")
        media = ("text/html" if name.endswith(".html") else "application/json"
                 if name.endswith(".json") else "text/plain")
        return FileResponse(path, media_type=media, filename=name,
                            content_disposition_type="inline")

    @app.get("/api/schema/rig-config")
    def rig_config_schema() -> Dict[str, Any]:
        return config_schema()

    @app.get("/api/{rest:path}", include_in_schema=False)
    def api_not_found(rest: str):
        raise HTTPException(404, "unknown endpoint")

    ui = static_dir()

    @app.get("/{rest:path}", include_in_schema=False)
    def ui_files(rest: str):
        if ui is None:
            return HTMLResponse(_FALLBACK_PAGE)
        target = (ui / rest).resolve() if rest else ui / "index.html"
        if rest and (ui in target.parents) and target.is_file():
            return FileResponse(target)
        index = ui / "index.html"
        return Response(index.read_bytes(), media_type="text/html")

    return app
