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
from .sources import settings_of
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


#: Browse references outside the roots start with this; the rest is a full path.
_ABSOLUTE = "abs:"


class _Roots:
    def __init__(self, roots: Sequence[str]):
        self.paths: List[Path] = [Path(r).expanduser().resolve() for r in roots]
        for p in self.paths:
            if not p.is_dir():
                raise NotADirectoryError(str(p))

    def add(self, folder: Path) -> int:
        """Index of ``folder`` among the roots, adding it when it is not one yet."""
        if folder not in self.paths:
            self.paths.append(folder)
        return self.paths.index(folder)

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
               read_only: bool = False, allowed_hosts: Optional[Sequence[str]] = None,
               anywhere: bool = False):
    """The Studio ASGI app over result folders under ``roots``. ``token`` (when given) is
    required on every request. ``read_only`` leaves out everything that writes or starts a
    calibration. ``allowed_hosts`` (host names) refuses requests addressed to another name.
    ``anywhere`` lets the page browse and read any folder on this computer and add one as a root;
    results are still only written under the roots."""
    from fastapi import Body, FastAPI, HTTPException, Query, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
    from starlette.middleware.base import BaseHTTPMiddleware

    from .. import __version__

    rs = _Roots(roots)
    app = FastAPI(title="DS-MSP Studio", version=__version__, docs_url=None, redoc_url=None,
                  openapi_url="/api/openapi.json")
    anywhere = anywhere and not read_only
    readable = None if anywhere else rs.paths          # None: no limit on what may be read
    jobs = None if read_only else JobManager(rs.paths, max_depth=max_depth, anywhere=anywhere)
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

    @app.exception_handler(PermissionError)
    async def _no_permission(_request: Request, exc: PermissionError) -> JSONResponse:
        """A path this account may not read is the user's to fix, not a server fault."""
        where = f": {exc.filename}" if exc.filename else ""
        return JSONResponse({"detail": f"Your account does not have permission to read this{where}"},
                            status_code=403)

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
        if not anywhere and not any(target == r or r in target.parents for r in rs.paths):
            raise HTTPException(400, "that folder is outside the folders Studio was started with")
        return target

    def _ref(folder: Path) -> str:
        for k, root in enumerate(rs.paths):
            if folder == root or root in folder.parents:
                return rs.run_id(k, folder)
        if anywhere:
            return f"{_ABSOLUTE}{folder}"
        raise KeyError(str(folder))

    def _browse_target(ref: str) -> Path:
        if anywhere and ref.startswith(_ABSOLUTE):
            return Path(ref[len(_ABSOLUTE):]).expanduser().resolve()
        return rs.resolve(ref)

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "version": __version__, "api_version": API_VERSION,
                "ui_bundled": static_dir() is not None, "can_run": jobs is not None}

    @app.get("/api/roots")
    def list_roots() -> List[Dict[str, Any]]:
        return [{"index": k, "name": p.name or str(p)} for k, p in enumerate(rs.paths)]

    @app.post("/api/roots/add")
    def add_root(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Serve one more folder: its results are listed and new results can be saved in it."""
        if not anywhere:
            raise HTTPException(403, "this Studio only uses the folders it was started with")
        folder = _inside(str(body.get("path") or ""))
        if not folder.is_dir():
            raise HTTPException(404, "no such folder")
        if folder == Path(folder.anchor):
            raise HTTPException(400, "choose a folder inside the disk, not the whole disk")
        k = rs.add(folder)
        if jobs is not None and folder not in jobs.roots:
            jobs.roots.append(folder)
        return {"index": k, "path": str(folder)}

    @app.get("/api/runs")
    def list_runs() -> List[Dict[str, Any]]:
        out = []
        listed: set = set()                             # an added root may lie inside another
        for k, root in enumerate(rs.paths):
            for folder in iter_run_dirs(root, max_depth=max_depth):
                if folder in listed:
                    continue
                listed.add(folder)
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

    @app.get("/api/run/errors")
    def get_run_errors(id: str = Query(...)) -> Dict[str, Any]:  # noqa: A002
        from .scene import error_spread
        return error_spread(_folder(id))

    @app.get("/api/run/frame")
    def get_run_frame(id: str = Query(...), cam: int = Query(..., ge=0),  # noqa: A002
                      frame: int = Query(..., ge=0)) -> Dict[str, Any]:
        from .scene import frame_corners, photo_path
        doc = frame_corners(_folder(id), cam, frame)
        doc["photo"] = photo_path(_folder(id), cam, frame, readable) is not None
        return doc

    @app.get("/api/run/photo")
    def get_run_photo(id: str = Query(...), cam: int = Query(..., ge=0),  # noqa: A002
                      frame: int = Query(..., ge=0), w: int = Query(1280, ge=64, le=2000)):
        from .scene import photo_jpeg, photo_path
        path = photo_path(_folder(id), cam, frame, readable)
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

    @app.get("/api/run/precision")
    def get_run_precision(id: str = Query(...), rotation: Optional[float] = Query(None, gt=0),  # noqa: A002
                          centre: Optional[float] = Query(None, gt=0)) -> Dict[str, Any]:
        """How far this result is from the limits (the run's own, unless given), and how many
        more photos of the same kind would close the gap."""
        from .precision import precision_gap
        from .runs import ACCEPTANCE_FILE, _read_json
        acc = _read_json(_folder(id) / ACCEPTANCE_FILE) or {}
        return precision_gap(acc.get("extrinsic_uncertainty"), rotation, centre)

    @app.get("/api/run/task")
    def get_run_task(id: str = Query(...), a: int = Query(..., ge=0), b: int = Query(..., ge=0),  # noqa: A002
                     distance: float = Query(..., gt=0), pixel_noise: float = Query(0.5, ge=0, le=50)
                     ) -> Dict[str, Any]:
        """What the calibration's uncertainty means for a point at ``distance``, located by
        both cameras or handed from one to the other, and over a range of distances around it
        (``curve``)."""
        import numpy as np

        from .scene import task_answer, task_curve
        folder = _folder(id)
        doc = task_answer(folder, a, b, distance, pixel_noise)
        base = doc.get("baseline") if doc.get("ok") else None
        if base:
            near, far = (0.05, 4) if doc["mode"] == "transfer" else (0.5, 40)
            grid = np.geomspace(max(near * base, distance / 20), max(far * base, distance * 4), 28)
            doc["curve"] = task_curve(folder, a, b, pixel_noise, grid.tolist(), doc["mode"])
        return doc

    # -- folders ---------------------------------------------------------------------------
    @app.get("/api/browse")
    def browse(ref: Optional[str] = Query(None)) -> Dict[str, Any]:
        """One folder under the roots: its subfolders (with how many images each holds), and
        its config-like files. Without ``ref``: the roots themselves."""
        if ref is None:
            home = Path.home()
            places = [] if not anywhere else [
                {"name": "Home folder", "ref": _ref(home), "path": str(home)},
                {"name": "Whole computer", "ref": f"{_ABSOLUTE}{home.anchor}", "path": home.anchor}]
            return {"ref": None, "path": None, "name": "", "parent": None, "images": 0,
                    "files": [], "anywhere": anywhere, "places": places,
                    "dirs": [{"name": p.name or str(p), "ref": f"{k}:",
                              "path": str(p), "images": 0, "result": False}
                             for k, p in enumerate(rs.paths)]}
        try:
            folder = _browse_target(ref)
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

        def is_result(d: Path) -> bool:
            try:
                return (d / "calibration_report.json").is_file()
            except OSError:                              # a folder this account may not open
                return False

        dirs, files = [], []
        try:
            entries = sorted(os.scandir(folder), key=lambda e: e.name.lower())
        except PermissionError:
            raise HTTPException(403, "Your account does not have permission to open this folder.") from None
        except OSError as exc:
            raise HTTPException(422, f"cannot read this folder: {exc}") from None
        for e in entries:
            try:
                e.name.encode("utf-8")
            except UnicodeEncodeError:                   # a name the page could not show
                continue
            if e.name.startswith("."):
                continue
            if e.is_dir(follow_symlinks=False):
                if len(dirs) < _MAX_LISTED:
                    child = folder / e.name
                    dirs.append({"name": e.name, "ref": _ref(child), "path": str(child),
                                 "images": n_images(child), "result": is_result(child),
                                 "locked": not os.access(child, os.R_OK | os.X_OK)})
            elif e.name.lower().endswith(_LISTED_FILES) and len(files) < _MAX_LISTED:
                files.append({"name": e.name, "path": str(folder / e.name)})
        is_root = folder in rs.paths
        at_top = folder == Path(folder.anchor) if anywhere else is_root
        parent = None if at_top else _ref(folder.parent)
        return {"ref": ref, "path": str(folder), "name": folder.name, "parent": parent,
                "is_root": is_root, "anywhere": anywhere, "images": n_images(folder),
                "dirs": dirs, "files": files}

    # -- other things a calibration can start from -----------------------------------------
    def _source(path: str, reader):
        target = _inside(path)
        if not target.is_file():
            raise HTTPException(404, "no such file")
        try:
            return reader(target)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/source/config")
    def source_config(path: str = Query(...)) -> Dict[str, Any]:
        from .sources import config_here
        return _source(path, config_here)

    @app.get("/api/source/keypoints")
    def source_keypoints(path: str = Query(...)) -> Dict[str, Any]:
        from .sources import scan_keypoints
        return _source(path, scan_keypoints)

    @app.get("/api/source/intrinsics")
    def source_intrinsics(path: str = Query(...)) -> Dict[str, Any]:
        from .sources import scan_intrinsics
        return _source(path, scan_intrinsics)

    # -- a result's fingerprint, and repeating it --------------------------------------------
    @app.get("/api/run/fingerprint")
    def run_fingerprint(id: str = Query(...)) -> Dict[str, Any]:  # noqa: A002
        from .replay import fingerprint
        return fingerprint(_folder(id))

    @app.get("/api/run/fingerprint/check")
    def run_fingerprint_check(id: str = Query(...)) -> Dict[str, Any]:  # noqa: A002
        """Re-hash every input of the run and say which changed. Slow on a large capture."""
        from .replay import check_inputs
        return check_inputs(_folder(id))

    @app.post("/api/jobs/repeat")
    def repeat_job(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        _folder(str(body.get("id")))
        try:
            job = _manager().submit_repeat(str(body.get("id")), int(body.get("root", 0)),
                                           str(body.get("name") or ""))
        except JobError as exc:
            return JSONResponse({"detail": str(exc), "problems": exc.problems}, status_code=400)
        except (TypeError, ValueError):
            raise HTTPException(400, "bad request") from None
        return _manager().snapshot(job.id, log_lines=0)

    # -- setting up a calibration ----------------------------------------------------------
    @app.post("/api/photos/scan")
    def photos_scan(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        from .photos import scan_photos
        from .sources import companions
        _manager()
        root = _inside(str(body.get("root_path") or ""))
        doc = scan_photos(str(root), body.get("cam_prefix") or None,
                          body.get("frame_pattern") or None)
        doc["found"] = companions(root)
        return doc

    def _work_root() -> Path:
        if not rs.paths:
            raise HTTPException(400, "choose where to save results first")
        return rs.paths[0]

    @app.post("/api/photos/lineup")
    def photos_lineup(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Link the chosen subfolders, in order, under the names the calibration reads."""
        from . import lineup
        _manager()
        root = _inside(str(body.get("root_path") or ""))
        try:
            dest = lineup.stage(root, [str(f) for f in body.get("folders") or []], _work_root())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"root_path": str(dest)}

    @app.post("/api/lens/values")
    def lens_values(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Turn lens values typed on the setup page into a lens file, and read it back."""
        from . import lineup
        from .sources import scan_intrinsics
        _manager()
        try:
            path = lineup.write_lens_values(list(body.get("cameras") or []), _work_root())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return scan_intrinsics(path)

    @app.get("/api/photos/folder-thumb")
    def photos_folder_thumb(path: str = Query(...), folder: str = Query(...)) -> Response:
        """The first photo of one subfolder, before it is known to be a camera."""
        from .photos import folder_thumbnail
        if Path(folder).name != folder:
            raise HTTPException(400, "not a folder directly inside this one")
        data = folder_thumbnail(_inside(path) / folder)
        if data is None:
            raise HTTPException(404, "no such photo")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=600"})

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

    @app.post("/api/photos/find-board")
    def photos_find_board(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Work out the number of squares of the board from the markers in a few photos."""
        from .photos import find_board
        values = _values_with_photos(body)
        try:
            return find_board(values["root_path"], values)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, f"could not look for the board: {exc}") from None

    @app.get("/api/photos/thumb")
    def photos_thumb(path: str = Query(...), cam: int = Query(..., ge=0),
                     photo: int = Query(0, ge=0), prefix: Optional[str] = Query(None)) -> Response:
        """One photo of one camera, scaled down, for the setup page."""
        from .photos import thumbnail
        data = thumbnail(str(_inside(path)), cam, photo, prefix)
        if data is None:
            raise HTTPException(404, "no such photo")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=600"})

    def _values_with_photos(body: Dict[str, Any]) -> Dict[str, Any]:
        values, _problems = configio.check_values(dict(body.get("values") or {}))
        values["root_path"] = str(_inside(str(values.get("root_path") or "")))
        return values

    def _values_with_views(body: Dict[str, Any]) -> Dict[str, Any]:
        """Form values whose photo folder, or else detections file, lies under a root."""
        values, _problems = configio.check_values(dict(body.get("values") or {}))
        if values.get("root_path"):
            values["root_path"] = str(_inside(str(values["root_path"])))
            return values
        path = _inside(str(values.get("keypoints_path") or ""))
        if not path.is_file():
            raise HTTPException(404, "no such detections file")
        values["keypoints_path"] = str(path)
        return values

    @app.post("/api/photos/lens-advice")
    def photos_lens_advice(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Measure one camera's field of view on a few photos, or on saved detections when no
        photo folder is set, and say whether the lens type chosen for it fits them."""
        from .advisor import lens_advice
        _manager()
        values = _values_with_views(body)
        try:
            return lens_advice(values, int(body.get("cam", 0)))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, f"could not check this lens: {exc}") from None

    @app.post("/api/photos/lens-health")
    def photos_lens_health(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Whether the lens values in the chosen lens file explain one camera's photos, or its
        saved detections when no photo folder is set."""
        from .advisor import lens_file_health
        _manager()
        values = _values_with_views(body)
        lens = _inside(str(values.get("cam_params_path") or ""))
        try:
            return lens_file_health(values, int(body.get("cam", 0)), str(lens))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, f"could not check these lens values: {exc}") from None

    @app.post("/api/capture/history")
    def capture_history(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """The calibrations already made from a photo folder, and whether the detections of
        one of them fit the settings in ``values``."""
        from .capture import history
        values = _values_with_photos(body)
        return history(values["root_path"], rs.paths, values, run_id=rs.run_id, max_depth=max_depth)

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
        try:
            doc = settings_of(folder, readable)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        doc["name"] = folder.name
        return doc

    # -- exporting LDC meshes and Isaac LUTs ------------------------------------------------
    def _export_paths():
        from .exporting import ExportProblem

        def inside(path: str) -> Path:
            try:
                return _inside(path)
            except HTTPException as exc:
                raise ExportProblem(str(exc.detail), [{"key": "source",
                                                       "message": str(exc.detail)}]) from None

        def result_folder(run_id: str) -> Path:
            try:
                return _folder(run_id)
            except HTTPException:
                raise ExportProblem("That result is not in the folders Studio was started with.",
                                    [{"key": "source", "message": "unknown result"}]) from None

        return inside, result_folder

    def _export_call(fn, *args):
        from .exporting import ExportProblem
        try:
            return fn(*args, *_export_paths())
        except ExportProblem as exc:
            return JSONResponse({"detail": str(exc), "problems": exc.problems}, status_code=400)

    @app.post("/api/export/cameras")
    def export_cameras(body: Dict[str, Any] = Body(...)) -> Any:
        """The cameras of a file, a result or typed values, as the export page lists them."""
        from .exporting import list_source
        return _export_call(list_source, dict(body.get("source") or {}))

    @app.post("/api/export/check")
    def export_check(body: Dict[str, Any] = Body(...)) -> Any:
        """A preview of each chosen camera's LDC mesh accuracy; nothing is written."""
        from .exporting import check
        return _export_call(check, dict(body.get("spec") or {}))

    @app.post("/api/export/view")
    def export_view(body: Dict[str, Any] = Body(...)) -> Any:
        """Pictures of one camera's LDC mesh or LUT as the spec would write it."""
        from .exporting import ExportProblem, view
        inside, result_folder = _export_paths()
        try:
            return view(dict(body.get("spec") or {}), str(body.get("camera") or ""),
                        str(body.get("kind") or "ldc"), inside, result_folder,
                        use_photo=bool(body.get("photo")),
                        frame=int(body["frame"]) if body.get("frame") is not None else None,
                        overlay=bool(body.get("overlay")),
                        display=max(160, min(int(body.get("width") or 720), 1600)), readable=readable)
        except ExportProblem as exc:
            return JSONResponse({"detail": str(exc), "problems": exc.problems}, status_code=400)

    @app.get("/api/run/export-view")
    def run_export_view(id: str = Query(...), camera: str = Query(...),  # noqa: A002
                        kind: str = Query("ldc"), photo: bool = Query(False),
                        frame: Optional[int] = Query(None), overlay: bool = Query(False),
                        width: int = Query(720, ge=160, le=1600)) -> Any:
        """Pictures of one camera's written LDC mesh or LUT, read from the files."""
        from .exporting import ExportProblem, view_files
        try:
            folder = rs.resolve(id)
        except KeyError:
            raise HTTPException(404, "unknown folder") from None
        try:
            return view_files(folder, camera, kind if kind in ("ldc", "lut") else "ldc",
                              use_photo=photo, frame=frame, overlay=overlay, display=width,
                              readable=readable)
        except ExportProblem as exc:
            return JSONResponse({"detail": str(exc), "problems": exc.problems}, status_code=400)
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(422, f"could not read the exported files: {exc}") from None

    @app.post("/api/jobs/export")
    def start_export(body: Dict[str, Any] = Body(...)) -> Any:
        inside, result_folder = _export_paths()
        try:
            job = _manager().submit_export(int(body.get("root", 0)), str(body.get("name") or ""),
                                           dict(body.get("spec") or {}), inside, result_folder,
                                           parent=str(body.get("parent") or ""))
        except JobError as exc:
            return JSONResponse({"detail": str(exc), "problems": exc.problems}, status_code=400)
        except (TypeError, ValueError):
            raise HTTPException(400, "bad request") from None
        return _manager().snapshot(job.id, log_lines=0)

    @app.get("/api/run/exports")
    def run_exports(id: str = Query(...)) -> Dict[str, Any]:  # noqa: A002
        """The exports a result or an export folder holds (its ``exports.json``)."""
        from .exporting import summary_of
        try:
            folder = rs.resolve(id)
        except KeyError:
            raise HTTPException(404, "unknown folder") from None
        return {"exports": summary_of(folder)}

    @app.get("/api/run/export-file")
    def run_export_file(id: str = Query(...), path: str = Query(...)) -> Response:  # noqa: A002
        """One exported file, for download."""
        try:
            folder = rs.resolve(id)
        except KeyError:
            raise HTTPException(404, "unknown folder") from None
        target = (folder / path).resolve()
        if folder not in target.parents or target.parent.name not in ("ldc", "isaac_lut") \
                or not target.is_file():
            raise HTTPException(404, "unknown file")
        return FileResponse(target, filename=target.name)

    # -- running ---------------------------------------------------------------------------
    @app.get("/api/jobs")
    def list_jobs() -> List[Dict[str, Any]]:
        return [] if jobs is None else jobs.list()

    @app.post("/api/jobs")
    def start_job(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            job = _manager().submit(int(body.get("root", 0)), str(body.get("name") or ""),
                                    dict(body.get("values") or {}),
                                    parent=str(body.get("parent") or ""),
                                    reuse=str(body["reuse"]) if body.get("reuse") else None)
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
