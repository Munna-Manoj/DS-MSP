"""Calibrations started from Studio.

A job is one run of the rig CLI in its own process: Studio writes ``calib_param.yml`` into a
new result folder, starts ``python -m ds_msp.rig.cli --config <that file>`` there, and follows
it through the CLI's progress log (:mod:`ds_msp.rig.progress`) and its terminal output. The
calibration is therefore the CLI's own, with the files the CLI writes; Studio adds three files
of its own to the folder: the job record, the captured output and the progress log.

Jobs run one at a time, in the order they were started: a calibration uses every core, and two
at once only slow each other down. Cancelling stops the process; what it had written stays in
the folder.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import configio

JOB_FILE = "studio_job.json"
LOG_FILE = "studio_run.log"
PROGRESS_FILE = "studio_progress.jsonl"
REPORT_FILE = "calibration_report.json"

ACTIVE = ("queued", "running")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_COUNTER_LINE = re.compile(r"^(\[\w+\] cam \d+:) \d+/\d+")
_ERROR_LINE = re.compile(r"^(?:[\w.]+\.)?(\w*(?:Error|Exception)): (.+)$")
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "Detection", "Reprojection"}


class JobError(ValueError):
    """A job could not be created; ``problems`` lists ``{"key", "message"}`` entries."""

    def __init__(self, message: str, problems: Optional[List[Dict[str, str]]] = None):
        super().__init__(message)
        self.problems = problems or []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def folder_name(name: str) -> str:
    """A result-folder name from what the user typed: spaces become ``_``; anything else that
    is not a letter, digit, ``.``, ``_`` or ``-`` is refused, not silently changed."""
    text = re.sub(r"\s+", "_", str(name).strip())
    if not _NAME.match(text) or text in (".", ".."):
        raise JobError("Give the calibration a name made of letters, digits, '.', '_' or '-' "
                       "(up to 80 characters, not starting with a dot).",
                       [{"key": "name", "message": "is not a usable folder name"}])
    return text


@dataclass
class Job:
    id: str
    name: str
    state: str                       # queued, running, done, failed, cancelled, interrupted
    created_utc: str
    command: List[str]
    started_utc: Optional[str] = None
    finished_utc: Optional[str] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    pid: Optional[int] = None
    repeat_of: Optional[str] = None      # id of the result this job repeats, if it is a repeat
    reused_from: Optional[str] = None    # id of the result whose detections this job reuses
    folder: Path = field(default=Path("."), repr=False, compare=False)

    def record(self) -> Dict[str, Any]:
        doc = asdict(self)
        doc.pop("folder")
        return doc

    def save(self) -> None:
        tmp = self.folder / (JOB_FILE + ".tmp")
        tmp.write_text(json.dumps(self.record(), indent=1), encoding="utf-8")
        os.replace(tmp, self.folder / JOB_FILE)


class _Progress:
    """Reads a progress log a piece at a time and keeps the summary a page shows."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.rest = b""
        self.summary: Dict[str, Any] = {"phase": "starting", "stage": None, "stage_index": 0,
                                        "stages": [], "detect": None, "save": None,
                                        "iter": None, "elapsed": None, "history": []}

    def update(self) -> Dict[str, Any]:
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                data = f.read()
        except OSError:
            return self.summary
        self.offset += len(data)
        lines = (self.rest + data).split(b"\n")
        self.rest = lines.pop()
        s = self.summary
        for raw in lines:
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            s["elapsed"] = ev.get("t", s["elapsed"])
            kind = ev.get("event")
            if kind == "detect":
                s["phase"], s["detect"] = "detecting", ev.get("cameras")
            elif kind == "scene":
                s["phase"] = "solving"
                s["scene"] = {"cameras": ev.get("cameras"), "observations": ev.get("observations")}
            elif kind == "stage":
                name = str(ev.get("name", ""))
                s["phase"] = "saving" if name.startswith("(save)") else "solving"
                s["stage"], s["stage_index"], s["iter"] = name, ev.get("index", 0), None
                s["stages"].append({"name": name, "t": ev.get("t")})
            elif kind == "iter":
                s["iter"] = {k: ev.get(k) for k in ("it", "max_iter", "rms", "cost")}
                if ev.get("rms") is not None:
                    s["history"].append([ev.get("t"), ev["rms"]])
                    if len(s["history"]) > 400:               # keep the curve light to send
                        s["history"] = s["history"][::2]
            elif kind == "save":
                s["phase"], s["save"] = "saving", ev.get("cameras")
            elif kind == "done":
                s["phase"], s["final_rms"] = "done", ev.get("rms")
        return s


def log_tail(path: Path, max_lines: int = 200, max_bytes: int = 48_000) -> List[str]:
    """The last lines of captured terminal output, as a terminal would show them: each line
    keeps what followed its last carriage return, colour codes are removed, and a run of
    per-camera counter lines (``[detect] cam 0: 7/30 ...``) keeps each camera's latest."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    out: List[str] = []
    counters: Dict[str, int] = {}            # counter prefix -> its place in the current run
    for line in text.split("\n"):
        line = _ANSI.sub("", line.rstrip("\r").rsplit("\r", 1)[-1]).rstrip()
        if not line:
            continue
        m = _COUNTER_LINE.match(line)
        if m is None:
            counters = {}
            out.append(line)
        elif m.group(1) in counters:
            out[counters[m.group(1)]] = line
        else:
            counters[m.group(1)] = len(out)
            out.append(line)
    return out[-max_lines:]


def error_headline(lines: Sequence[str]) -> Optional[str]:
    """The message of the exception that ended a run, from its captured output."""
    for line in reversed(lines):
        m = _ERROR_LINE.match(line.strip())
        if m:
            text = m.group(2).strip()
            return text[:1].upper() + text[1:]
    return None


class JobManager:
    """Creates, runs, follows and cancels calibration jobs under ``roots``."""

    def __init__(self, roots: Sequence[Path], *, python: Optional[str] = None,
                 max_depth: int = 6):
        self.roots = [Path(r) for r in roots]
        self.python = python or sys.executable
        self._jobs: Dict[str, Job] = {}
        self._progress: Dict[str, _Progress] = {}
        self._procs: Dict[str, subprocess.Popen] = {}
        self._cancel: set = set()
        self._lock = threading.RLock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._discover(max_depth)

    # -- ids and folders -------------------------------------------------------------------
    def _id(self, folder: Path) -> str:
        for k, root in enumerate(self.roots):
            if folder == root or root in folder.parents:
                rel = folder.relative_to(root).as_posix()
                return f"{k}:{'' if rel == '.' else rel}"
        raise KeyError(str(folder))

    def _discover(self, max_depth: int) -> None:
        for root in self.roots:
            stack = [(root, 0)]
            while stack:
                folder, depth = stack.pop()
                record = folder / JOB_FILE
                if record.is_file():
                    try:
                        doc = json.loads(record.read_text(encoding="utf-8"))
                        job = Job(folder=folder, **{k: doc.get(k) for k in Job.__dataclass_fields__
                                                    if k != "folder"})
                        job.id = self._id(folder)
                    except (ValueError, TypeError, KeyError):
                        continue
                    if job.state in ACTIVE:              # its Studio stopped before it ended
                        job.state, job.pid = "interrupted", None
                        job.error = "Studio was stopped while this calibration was running."
                        job.save()
                    self._jobs[job.id] = job
                if depth >= max_depth:
                    continue
                try:
                    stack.extend((p, depth + 1) for p in folder.iterdir()
                                 if p.is_dir() and not p.is_symlink()
                                 and not p.name.startswith(".") and p.name not in _SKIP_DIRS)
                except OSError:
                    continue

    # -- creating --------------------------------------------------------------------------
    def prepare(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Check form values without creating anything. Returns ``{"ok", "problems",
        "warnings", "yaml"}``."""
        clean, problems = configio.check_values(values)
        problems += configio.path_problems(clean, self.roots, skip=("save_path",))
        if not problems and not (clean.get("root_path") or clean.get("keypoints_path")):
            problems.append({"key": "root_path", "message": "choose the folder with the photos"})
        return {"ok": not problems, "problems": problems, "warnings": [],
                "yaml": configio.render_config(clean), "values": clean}

    def submit(self, root: int, name: str, values: Dict[str, Any], parent: str = "",
               reuse: Optional[str] = None) -> Job:
        """Create and queue a calibration. ``reuse`` names an earlier result made from the
        same photos: its corner detections are copied into the new folder and used, after
        checking again that they still fit (:mod:`ds_msp.studio.capture`)."""
        folder_name(name)                                # a bad name is reported first
        checked = self.prepare(values)
        if not checked["ok"]:
            raise JobError("The settings have problems.", checked["problems"])
        folder = self._target(root, name, parent)
        clean = dict(checked["values"])
        clean["save_path"] = str(folder) + os.sep
        source = None
        if reuse:
            from . import capture
            hist = capture.history(str(clean.get("root_path") or ""), self.roots, clean)
            row = next((r for r in hist["runs"] if r["id"] == reuse), None)
            if row is None or not (row.get("reuse") or {}).get("ok"):
                why = (row or {}).get("reuse", {}).get("why", "it was not made from these photos")
                raise JobError(f"The detections of that result cannot be reused: {why}.",
                               [{"key": "reuse", "message": why}])
            k_str, _sep, rel = reuse.partition(":")
            source = (self.roots[int(k_str)] / rel).resolve()
        folder.mkdir(parents=True)
        if source is not None:
            from . import capture
            clean["keypoints_path"] = str(capture.copy_detections(source, folder))
        config = folder / configio.CONFIG_NAME
        configio.write_config(config, clean)
        loaded = configio.load_check(config)
        if not loaded["ok"]:
            import shutil
            shutil.rmtree(folder)
            raise JobError(loaded["error"], [{"key": "", "message": loaded["error"]}])
        return self._enqueue(folder, ["--config", str(config)], reused_from=reuse or None)

    def _target(self, root: int, name: str, parent: str = "") -> Path:
        """The new result folder for a job: under a served root, named ``name``, not there yet."""
        if not 0 <= int(root) < len(self.roots):
            raise JobError("Unknown folder.", [{"key": "root", "message": "is not a served folder"}])
        base = (self.roots[int(root)] / parent).resolve()
        if base != self.roots[int(root)] and self.roots[int(root)] not in base.parents:
            raise JobError("Unknown folder.", [{"key": "parent", "message": "is outside the roots"}])
        folder = base / folder_name(name)
        if folder.exists():
            raise JobError(f"A folder named {folder.name} already exists there. Pick another name.",
                           [{"key": "name", "message": "already exists"}])
        return folder

    def submit_repeat(self, original_id: str, root: int, name: str) -> Job:
        """Repeat the calibration of result ``original_id`` into a new folder, from the
        configuration and inputs its manifest records (:mod:`ds_msp.studio.replay`)."""
        from . import replay

        k_str, _sep, rel = original_id.partition(":")
        if not k_str.isdigit() or int(k_str) >= len(self.roots):
            raise JobError("Unknown result.", [{"key": "id", "message": "is not a result here"}])
        original = (self.roots[int(k_str)] / rel).resolve()
        if not any(original == r or r in original.parents for r in self.roots):
            raise JobError("Unknown result.", [{"key": "id", "message": "is not a result here"}])
        folder = self._target(root, name)
        try:
            args = replay.replay_command(original, folder, self.roots)
        except ValueError as exc:
            raise JobError(f"This result cannot be repeated: {exc}.") from None
        folder.mkdir(parents=True)
        return self._enqueue(folder, args, repeat_of=original_id)

    def _enqueue(self, folder: Path, cli_args: List[str], repeat_of: Optional[str] = None,
                 reused_from: Optional[str] = None) -> Job:
        command = [self.python, "-m", "ds_msp.rig.cli", *cli_args,
                   "--no-webviewer", "--progress-file", str(folder / PROGRESS_FILE)]
        job = Job(id=self._id(folder), name=folder.name, state="queued", created_utc=_now(),
                  command=command, repeat_of=repeat_of, reused_from=reused_from, folder=folder)
        job.save()
        with self._lock:
            self._jobs[job.id] = job
            self._queue.put(job.id)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run_queue, name="dsmsp-studio-jobs",
                                                daemon=True)
                self._worker.start()
        return job

    # -- running ---------------------------------------------------------------------------
    def _run_queue(self) -> None:
        while True:
            try:
                job_id = self._queue.get(timeout=5.0)
            except queue.Empty:
                with self._lock:
                    if self._queue.empty():
                        self._worker = None
                        return
                continue
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.state != "queued":
                    continue
                job.state, job.started_utc = "running", _now()
            self._run(job)

    def _run(self, job: Job) -> None:
        # The calibration must be this Studio's own code, whatever the folder it starts in.
        package_home = str(Path(__file__).resolve().parents[2])
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        env["PYTHONPATH"] = os.pathsep.join(
            [package_home] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        try:
            with open(job.folder / LOG_FILE, "wb") as log:
                proc = subprocess.Popen(job.command, cwd=job.folder, stdout=log,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        env=env, start_new_session=os.name == "posix")
                with self._lock:
                    self._procs[job.id] = proc
                    job.pid = proc.pid
                    job.save()
                    stop_now = job.id in self._cancel      # cancelled while it was starting
                if stop_now:
                    _stop(proc)
                code = proc.wait()
        except OSError as exc:
            code = None
            job.error = f"could not start the calibration: {exc}"
        with self._lock:
            self._procs.pop(job.id, None)
            cancelled = job.id in self._cancel
            self._cancel.discard(job.id)
            job.exit_code, job.finished_utc, job.pid = code, _now(), None
            if cancelled:
                job.state, job.error = "cancelled", None
            elif (job.folder / REPORT_FILE).is_file():
                job.state = "done"
            else:
                job.state = "failed"
                job.error = job.error or error_headline(log_tail(job.folder / LOG_FILE)) or \
                    "The calibration stopped before writing a result."
            job.save()

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs[job_id]
            if job.state == "queued":
                job.state, job.finished_utc = "cancelled", _now()
                job.save()
                return job
            if job.state != "running":
                return job
            self._cancel.add(job_id)
            proc = self._procs.get(job_id)
        if proc is not None:
            _stop(proc)
        return job

    def shutdown(self) -> None:
        """Stop the running calibration, if any (called when Studio exits)."""
        with self._lock:
            procs = list(self._procs.items())
            for job_id, _p in procs:
                self._cancel.add(job_id)
        for _job_id, proc in procs:
            _stop(proc)

    # -- reading ---------------------------------------------------------------------------
    def get(self, job_id: str) -> Job:
        with self._lock:
            return self._jobs[job_id]

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_utc, reverse=True)
            waiting = [j.id for j in sorted(self._jobs.values(), key=lambda j: j.created_utc)
                       if j.state == "queued"]
            return [self._brief(j, waiting) for j in jobs]

    def _brief(self, job: Job, waiting: Sequence[str]) -> Dict[str, Any]:
        doc = job.record()
        doc.pop("command")
        doc["has_result"] = (job.folder / REPORT_FILE).is_file()
        doc["queue_position"] = waiting.index(job.id) + 1 if job.id in waiting else None
        return doc

    def snapshot(self, job_id: str, log_lines: int = 200) -> Dict[str, Any]:
        """Everything a page needs to show one job: its record, progress and recent output."""
        with self._lock:
            job = self._jobs[job_id]
            waiting = [j.id for j in sorted(self._jobs.values(), key=lambda j: j.created_utc)
                       if j.state == "queued"]
            doc = self._brief(job, waiting)
            reader = self._progress.setdefault(job_id, _Progress(job.folder / PROGRESS_FILE))
            doc["progress"] = dict(reader.update())
        doc["command"] = self._terminal_command(job)
        doc["log"] = log_tail(job.folder / LOG_FILE, max_lines=log_lines)
        if job.repeat_of and job.state == "done":
            from . import replay
            k_str, _sep, rel = job.repeat_of.partition(":")
            doc["repeat"] = replay.compare((self.roots[int(k_str)] / rel).resolve(), job.folder)
        return doc

    @staticmethod
    def _terminal_command(job: Job) -> str:
        """The same calibration as a terminal command: the job's own arguments without the two
        Studio adds to follow it."""
        args = list(job.command[3:])
        if "--progress-file" in args:
            k = args.index("--progress-file")
            del args[k:k + 2]
        args = [a for a in args if a != "--no-webviewer"]
        return "ds-msp-calibrate-rig " + shlex.join(args)


def _stop(proc: subprocess.Popen, grace: float = 5.0) -> None:
    """End ``proc`` and the processes it started: ask first, then kill."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except (OSError, ProcessLookupError):
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except (OSError, ProcessLookupError):
            pass
