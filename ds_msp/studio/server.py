"""DS-MSP Studio HTTP server: a JSON API over result folders, plus the bundled UI.

Needs the ``ui`` extra (``pip install "ds-msp[ui]"``). The server reads result folders under
the roots it was started with, and, unless started read-only, sets up and runs calibrations
whose photos and results lie under those roots (:mod:`ds_msp.studio.jobs`). Every requested
path is resolved and checked to lie inside one of them. When bound to anything other than
localhost it requires a token, given once as ``?token=`` and then kept in a cookie.

Requests that change something are POSTs and must carry the ``x-dsmsp-studio`` header, which
a page on another site cannot send, and come from this server's own page. With
``allowed_hosts`` set, a request addressed to any other host name is refused, so another
site cannot reach a local Studio by pointing its own name at this computer.
"""
from __future__ import annotations

import hmac
import importlib.resources
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..rig.config_schema import config_schema
from . import configio
from .jobs import ACTIVE, JobError, JobManager
from .runs import KNOWN_FILES, RunError, iter_run_dirs, load_diagnostics, load_run, summarize_run

API_VERSION = 1
TOKEN_COOKIE = "dsmsp_token"
TOKEN_HEADER = "x-dsmsp-token"
WRITE_HEADER = "x-dsmsp-studio"
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_LISTED_FILES = (".yml", ".yaml", ".json")
_MAX_LISTED = 400

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


def _host_name(host: str) -> str:
    """The name part of a ``Host`` header value (``[::1]:8765`` -> ``::1``)."""
    host = host.strip().lower()
    if host.startswith("["):
        return host[1:host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if ":" in host else host


def create_app(roots: Sequence[str], *, token: Optional[str] = None, max_depth: int = 6,
               read_only: bool = False, allowed_hosts: Optional[Sequence[str]] = None):
    """The Studio ASGI app over result folders under ``roots``. ``token`` (when given) is
    required on every request. ``read_only`` leaves out everything that writes or starts a
    calibration. ``allowed_hosts`` (host names) refuses requests addressed to another name."""
    from fastapi import Body, FastAPI, HTTPException, Query, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
    from starlette.middleware.base import BaseHTTPMiddleware

    from .. import __version__

    rs = _Roots(roots)
    app = FastAPI(title="DS-MSP Studio", version=__version__, docs_url=None, redoc_url=None,
                  openapi_url="/api/openapi.json")
    jobs = None if read_only else JobManager(rs.paths, max_depth=max_depth)
    app.state.jobs = jobs
    hosts = None if allowed_hosts is None else {h.lower() for h in allowed_hosts}

    class _Guard(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            host = request.headers.get("host", "")
            if hosts is not None and _host_name(host) not in hosts:
                return JSONResponse({"detail": "request addressed to another host"},
                                    status_code=403)
            if request.method not in ("GET", "HEAD", "OPTIONS"):
                origin = request.headers.get("origin")
                same = origin is None or origin.split("://", 1)[-1].lower() == host.lower()
                if request.headers.get(WRITE_HEADER) != "1" or not same:
                    return JSONResponse({"detail": "not sent by the Studio page"},
                                        status_code=403)
            return await call_next(request)

    app.add_middleware(_Guard)

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

    def _manager() -> JobManager:
        if jobs is None:
            raise HTTPException(403, "this Studio was started read-only")
        return jobs

    def _inside(path: str) -> Path:
        """``path`` resolved, refused unless it lies under a root."""
        target = Path(str(path)).expanduser()
        if not target.is_absolute():
            raise HTTPException(400, "give a full path")
        target = target.resolve()
        if not any(target == r or r in target.parents for r in rs.paths):
            raise HTTPException(400, "that folder is outside the folders Studio was started with")
        return target

    def _ref(folder: Path) -> str:
        for k, root in enumerate(rs.paths):
            if folder == root or root in folder.parents:
                return rs.run_id(k, folder)
        raise KeyError(str(folder))

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "version": __version__, "api_version": API_VERSION,
                "ui_bundled": static_dir() is not None, "can_run": jobs is not None}

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

    @app.get("/api/run/diagnostics")
    def get_run_diagnostics(id: str = Query(...), cam: int = Query(..., ge=0)) -> Dict[str, Any]:  # noqa: A002
        folder = _folder(id)
        try:
            return load_diagnostics(folder, cam)
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=f"cannot read diagnostics: {exc}") from exc

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

    # -- a result on its own capture: photos, frames, the board through time, meaning ---------
    @app.get("/api/run/scene")
    def get_run_scene(id: str = Query(...)) -> Dict[str, Any]:  # noqa: A002
        from .scene import run_scene
        return run_scene(_folder(id))

    @app.get("/api/run/frames")
    def get_run_frames(id: str = Query(...), cam: int = Query(..., ge=0)) -> Dict[str, Any]:  # noqa: A002
        from .scene import frame_list
        return frame_list(_folder(id), cam)

    @app.get("/api/run/frame")
    def get_run_frame(id: str = Query(...), cam: int = Query(..., ge=0),  # noqa: A002
                      frame: int = Query(..., ge=0)) -> Dict[str, Any]:
        from .scene import frame_corners, photo_path
        doc = frame_corners(_folder(id), cam, frame)
        doc["photo"] = photo_path(_folder(id), cam, frame, rs.paths) is not None
        return doc

    @app.get("/api/run/photo")
    def get_run_photo(id: str = Query(...), cam: int = Query(..., ge=0),  # noqa: A002
                      frame: int = Query(..., ge=0), w: int = Query(1280, ge=64, le=2000)):
        from .scene import photo_jpeg, photo_path
        path = photo_path(_folder(id), cam, frame, rs.paths)
        if path is None:
            raise HTTPException(404, "this photo is not under the folders Studio was started with")
        try:
            data = photo_jpeg(path, w)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return Response(data, media_type="image/jpeg",
                        headers={"cache-control": "private, max-age=3600"})

    @app.get("/api/run/pair")
    def get_run_pair(id: str = Query(...), a: int = Query(..., ge=0),  # noqa: A002
                     b: int = Query(..., ge=0)) -> Dict[str, Any]:
        from .scene import pair_geometry
        return pair_geometry(_folder(id), a, b)

    @app.get("/api/run/task")
    def get_run_task(id: str = Query(...), a: int = Query(..., ge=0), b: int = Query(..., ge=0),  # noqa: A002
                     distance: float = Query(..., gt=0), pixel_noise: float = Query(0.5, ge=0, le=50)
                     ) -> Dict[str, Any]:
        """What the calibration's uncertainty means for a point seen by two cameras, at
        ``distance`` and over a range of distances around it (``curve``)."""
        import numpy as np

        from .scene import task_curve, task_uncertainty
        folder = _folder(id)
        doc = task_uncertainty(folder, a, b, distance, pixel_noise)
        base = doc.get("baseline") if doc.get("ok") else None
        if base:
            grid = np.geomspace(max(0.5 * base, distance / 20), max(40 * base, distance * 4), 28)
            doc["curve"] = task_curve(folder, a, b, pixel_noise, grid.tolist())
        return doc

    # -- folders ---------------------------------------------------------------------------
    @app.get("/api/browse")
    def browse(ref: Optional[str] = Query(None)) -> Dict[str, Any]:
        """One folder under the roots: its subfolders (with how many images each holds), and
        its config-like files. Without ``ref``: the roots themselves."""
        if ref is None:
            return {"ref": None, "path": None, "name": "", "parent": None, "images": 0,
                    "files": [], "dirs": [{"name": p.name or str(p), "ref": f"{k}:",
                                           "path": str(p), "images": 0, "result": False}
                                          for k, p in enumerate(rs.paths)]}
        try:
            folder = rs.resolve(ref)
        except KeyError:
            raise HTTPException(404, "unknown folder") from None
        if not folder.is_dir():
            raise HTTPException(404, "unknown folder")

        def n_images(d: Path) -> int:
            try:
                return sum(1 for e in os.scandir(d)
                           if e.is_file() and e.name.lower().endswith(_IMAGE_EXT))
            except OSError:
                return 0

        dirs, files = [], []
        try:
            entries = sorted(os.scandir(folder), key=lambda e: e.name.lower())
        except OSError as exc:
            raise HTTPException(422, f"cannot read this folder: {exc}") from None
        for e in entries:
            if e.name.startswith("."):
                continue
            if e.is_dir(follow_symlinks=False):
                if len(dirs) < _MAX_LISTED:
                    child = folder / e.name
                    dirs.append({"name": e.name, "ref": _ref(child), "path": str(child),
                                 "images": n_images(child),
                                 "result": (child / "calibration_report.json").is_file()})
            elif e.name.lower().endswith(_LISTED_FILES) and len(files) < _MAX_LISTED:
                files.append({"name": e.name, "path": str(folder / e.name)})
        k = int(ref.split(":", 1)[0])
        parent = None if folder == rs.paths[k] else _ref(folder.parent)
        return {"ref": ref, "path": str(folder), "name": folder.name, "parent": parent,
                "is_root": folder == rs.paths[k], "images": n_images(folder),
                "dirs": dirs, "files": files}

    # -- setting up a calibration ----------------------------------------------------------
    @app.post("/api/photos/scan")
    def photos_scan(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        from .photos import scan_photos
        _manager()
        root = _inside(str(body.get("root_path") or ""))
        return scan_photos(str(root), body.get("cam_prefix") or None,
                           body.get("frame_pattern") or None)

    @app.post("/api/photos/check-board")
    def photos_check_board(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        from .photos import check_board
        _manager()
        values, problems = configio.check_values(dict(body.get("values") or {}))
        board_keys = ("number_x_square", "number_y_square", "number_board", "length_square",
                      "length_marker", "boards_index", "number_x_square_per_board",
                      "number_y_square_per_board")
        bad = [p for p in problems if p["key"] in board_keys]
        if bad:
            raise HTTPException(400, f"{bad[0]['key']} {bad[0]['message']}")
        root = _inside(str(values.get("root_path") or ""))
        try:
            return check_board(str(root), values)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, f"this board cannot be built: {exc}") from None

    @app.post("/api/config/check")
    def config_check(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Check form values the way a run would: the schema, the paths, then the CLI's own
        loader on the file that would be written."""
        import tempfile

        out = _manager().prepare(dict(body.get("values") or {}))
        if out["ok"]:
            with tempfile.TemporaryDirectory(prefix="dsmsp-studio-") as tmp:
                path = Path(tmp) / configio.CONFIG_NAME
                configio.write_config(path, out["values"])
                loaded = configio.load_check(path)
            out["warnings"] = loaded["warnings"]
            if not loaded["ok"]:
                out["ok"] = False
                out["problems"] = [{"key": "", "message": loaded["error"]}]
        return out

    @app.post("/api/config/import")
    def config_import(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        text = body.get("text")
        if not isinstance(text, str) or len(text) > 400_000:
            raise HTTPException(400, "send the file's text")
        try:
            return configio.read_values(text)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.get("/api/config")
    def config_of(id: str = Query(...)) -> Dict[str, Any]:  # noqa: A002
        """The settings a folder was calibrated with, to start a new run from them."""
        try:
            folder = rs.resolve(id)
        except KeyError:
            raise HTTPException(404, "unknown folder") from None
        path = folder / configio.CONFIG_NAME
        if not path.is_file():
            raise HTTPException(404, "this folder has no calib_param.yml")
        try:
            doc = configio.read_values(path.read_text(encoding="utf-8", errors="replace"))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        doc["values"].pop("save_path", None)
        doc["name"] = folder.name
        return doc

    # -- running ---------------------------------------------------------------------------
    @app.get("/api/jobs")
    def list_jobs() -> List[Dict[str, Any]]:
        return [] if jobs is None else jobs.list()

    @app.post("/api/jobs")
    def start_job(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            job = _manager().submit(int(body.get("root", 0)), str(body.get("name") or ""),
                                    dict(body.get("values") or {}),
                                    parent=str(body.get("parent") or ""))
        except JobError as exc:
            return JSONResponse({"detail": str(exc), "problems": exc.problems}, status_code=400)
        except (TypeError, ValueError):
            raise HTTPException(400, "bad request") from None
        return _manager().snapshot(job.id, log_lines=0)

    @app.get("/api/job")
    def get_job(id: str = Query(...)) -> Dict[str, Any]:  # noqa: A002
        try:
            return _manager().snapshot(id)
        except KeyError:
            raise HTTPException(404, "unknown calibration") from None

    @app.post("/api/job/cancel")
    def cancel_job(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            job = _manager().cancel(str(body.get("id")))
        except KeyError:
            raise HTTPException(404, "unknown calibration") from None
        return {"id": job.id, "state": job.state, "active": job.state in ACTIVE}

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
