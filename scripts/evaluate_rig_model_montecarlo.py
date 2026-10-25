"""Camera-model Monte Carlo on the MC-Calib real image sequences.

Every rig is calibrated from raw ChArUco images many times, each time with a different camera
model per camera (every single-model assignment plus seeded random mixes), intrinsics estimated
from scratch. Each result is compared with MC-Calib's calibration of the *same* training images
and checked on held-out frames two ways that do not depend on MC-Calib:

* held-out reprojection: target poses refit with cameras and target fixed;
* held-out triangulation: every target corner seen by two or more cameras (rays >= 3 deg apart)
  is triangulated with the calibrated rig and compared with the printed board -- similarity
  scale error and every corner-to-corner distance. Wrong extrinsics show up as a scale or
  shape error no reprojection fit can hide.

Stages (``work`` is any scratch directory)::

  split   <Real_Images> <configs> <work>   frame-number split (every 3rd frame held out)
  detect  <work> <seq>                     detect once; cache training + held-out corners
  plan    <work> [seed]                    write the trial list
  run     <work> <seq> <trial>             calibrate one trial and evaluate it
  ref     <work> <seq> <mccalib_out_dir>   evaluate MC-Calib's result on the same held-out data
  report  <work>                           markdown tables

``<configs>`` holds MC-Calib's ``calib_param_<seq>.yml`` files (``configs/Real_images`` in the
MC-Calib repository). Run MC-Calib itself on ``<work>/<seq>/train`` for the reference.
Lengths are reported in millimetres, assuming the configs' ``square_size`` is in centimetres
(true for the MC-Calib real sequences).
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import itertools
import json
import os
import pickle
import re
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from ds_msp.data.observations import RigState                              # noqa: E402
from ds_msp.detect.charuco import board_object_points                     # noqa: E402
from ds_msp.io.mccalib import _load_cameras, _load_object                 # noqa: E402
from ds_msp.rig import bundle, cli                                        # noqa: E402
from ds_msp.rig.calib_param import _source_model, load_config             # noqa: E402
from ds_msp.rig.pose_init import robust_pose_irls                         # noqa: E402
from ds_msp.rig.reconstruct import (detect_board_obs_images,              # noqa: E402
                                    object_obs_from_board_obs_multi)

PINHOLE = ["radtan", "kb", "ucm", "eucm", "ds", "dsplus", "ocam"]
FISHEYE = ["kb", "ucm", "eucm", "ds", "dsplus", "ocam"]          # radtan: negative control
N_MIXED = 13
MM_PER_UNIT = 10.0
_FRAME = re.compile(r"(\d+)\D*$")


def _frame(name):
    return int(_FRAME.search(os.path.splitext(name)[0]).group(1))


def _cfg_path(work, seq):
    return Path(work) / seq / "calib_param.yml"


# ----------------------------------------------------------------------------- split / detect
def split(real_images, configs, work):
    """Hold out every 3rd frame *by frame number*, the same frames in every camera."""
    for seq in sorted(os.listdir(real_images)):
        src = Path(real_images) / seq
        cfg = Path(configs) / f"calib_param_{seq}.yml"
        if not cfg.exists():
            continue
        cams = sorted(p.name for p in src.iterdir() if p.is_dir() and re.match(r"(?i)cam_\d+", p.name))
        frames = sorted({_frame(f) for f in os.listdir(src / cams[0])})
        held = set(frames[0::3])
        for c in cams:
            for part in ("train", "holdout"):
                (Path(work) / seq / part / c).mkdir(parents=True, exist_ok=True)
            for f in os.listdir(src / c):
                part = "holdout" if _frame(f) in held else "train"
                shutil.copy2(src / c / f, Path(work) / seq / part / c / f)
        shutil.copy2(cfg, _cfg_path(work, seq))
        print(seq, len(frames), "frames,", len(held), "held out")


def _overrides(work, seq, **extra):
    base = {"root_path": f"{Path(work) / seq / 'train'}/", "cam_params_path": "None",
            "keypoints_path": "None", "save_detection": "0", "save_reprojection": "0",
            "webviewer": "0", "verbose": "0"}
    base.update(extra)
    return [f"{k}={v}" for k, v in base.items()]


def detect(work, seq):
    """One default calibration from the training images (its detections are the cache every
    trial reuses -- detection does not depend on the camera model) + held-out detections."""
    out = Path(work) / seq / "detect"
    with contextlib.redirect_stdout(io.StringIO()):
        cli._run_config(str(_cfg_path(work, seq)), _overrides(work, seq, save_path=f"{out}/"),
                        pass_px=1.0, warn_px=2.0, report_path="")
    cfg = load_config(str(_cfg_path(work, seq)))
    boards, _ = detect_board_obs_images(f"{Path(work) / seq / 'holdout'}/",
                                        list(range(cfg.number_camera)), cfg.boards,
                                        cam_prefix=cfg.cam_prefix, frame_pattern=cfg.frame_pattern)
    (Path(work) / seq / "holdout_boards.pkl").write_bytes(pickle.dumps(boards))
    print(seq, "cached", len(boards), "held-out board observations")


# ----------------------------------------------------------------------------- evaluation
def heldout_reprojection(rig, boards):
    obs = object_obs_from_board_obs_multi(copy.deepcopy(boards), list(rig.objects.values()))
    obs = [o for o in obs if o.cam_id in rig.cameras]
    by_key = {}
    for o in obs:
        by_key.setdefault((o.object_id, o.frame_id), []).append(o)
    poses, failed = {}, 0
    for key, group in by_key.items():
        for o in sorted(group, key=lambda o: -len(o.point_rows)):
            xyz = rig.objects[o.object_id].pts_3d[o.point_rows]
            pose = robust_pose_irls(rig.cameras[o.cam_id], xyz, o.pts_2d, max_iter=12, gnc_iters=2)
            if pose is not None:
                poses[key] = np.linalg.inv(rig.T_c_g[o.cam_id]) @ pose
                break
        else:
            failed += 1
    r = copy.deepcopy(rig)
    r.object_poses = poses
    used = [o for o in obs if (o.object_id, o.frame_id) in poses]
    r = bundle.refine(r, used, fix_intrinsics=True, fix_extrinsics=True, max_iter=30,
                      robust_kernel="cauchy", robust_scale="auto")
    err = []
    for o in used:
        T = r.T_c_g[o.cam_id] @ r.object_poses[(o.object_id, o.frame_id)]
        X = r.objects[o.object_id].pts_3d[o.point_rows]
        uv, valid = r.cameras[o.cam_id].project(X @ T[:3, :3].T + T[:3, 3])
        err.extend(np.linalg.norm(uv[valid] - o.pts_2d[valid], axis=1).tolist())
    e = np.asarray(err)
    return {"corners": int(e.size), "failed_groups": failed,
            "rms_px": float(np.sqrt(np.mean(e ** 2))), "median_px": float(np.median(e)),
            "p95_px": float(np.percentile(e, 95))}


def _triangulate(rig, cams, uvs, iters=5):
    """Midpoint of bearing rays, then Gauss-Newton on pixel reprojection (rig fixed)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c, uv in zip(cams, uvs):
        f, _v = rig.cameras[c].unproject(np.asarray(uv, float)[None])
        f = np.asarray(f, float).reshape(3)
        f /= np.linalg.norm(f)
        R, t = rig.T_c_g[c][:3, :3], rig.T_c_g[c][:3, 3]
        d = R.T @ f
        Pm = np.eye(3) - np.outer(d, d)
        A += Pm
        b += Pm @ (-R.T @ t)
    X = np.linalg.solve(A, b)
    for _ in range(iters):
        H, g = np.zeros((3, 3)), np.zeros(3)
        for c, uv in zip(cams, uvs):
            R, t = rig.T_c_g[c][:3, :3], rig.T_c_g[c][:3, 3]
            uvp, Jp, _jp, valid = rig.cameras[c].project_jacobian((R @ X + t)[None])
            if not np.asarray(valid).ravel()[0]:
                continue
            J = Jp[0] @ R
            H += J.T @ J
            g += J.T @ (uvp[0] - np.asarray(uv, float))
        X = X - np.linalg.solve(H + 1e-12 * np.eye(3), g)
    dirs = [(X + rig.T_c_g[c][:3, :3].T @ rig.T_c_g[c][:3, 3]) for c in cams]
    dirs = [v / np.linalg.norm(v) for v in dirs]
    ang = max(np.degrees(np.arccos(np.clip(a @ b_, -1, 1)))
              for a, b_ in itertools.combinations(dirs, 2))
    return X, ang


def heldout_triangulation(rig, boards, specs):
    by = {}
    for o in boards:
        if o.cam_id in rig.cameras:
            by.setdefault((o.frame_id, o.board_id), []).append(o)
    scale, dist, n_pts, n_boards = [], [], 0, 0
    for (_fr, bid), obs in by.items():
        if len({o.cam_id for o in obs}) < 2:
            continue
        nominal = board_object_points(specs[bid])
        views = {}
        for o in obs:
            for cid, uv in zip(o.corner_ids, o.pts_2d):
                views.setdefault(int(cid), []).append((o.cam_id, uv))
        pts, ids = [], []
        for cid, v in views.items():
            if len(v) < 2:
                continue
            try:
                X, ang = _triangulate(rig, [x[0] for x in v], [x[1] for x in v])
            except np.linalg.LinAlgError:
                continue
            if ang >= 3.0 and np.all(np.isfinite(X)):
                pts.append(X)
                ids.append(cid)
        if len(ids) < 6:
            continue
        Pp, Q = np.asarray(pts), nominal[ids]
        pc, qc = Pp - Pp.mean(0), Q - Q.mean(0)
        U, S, Vt = np.linalg.svd(qc.T @ pc)
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        scale.append((S * np.diag(D)).sum() / (qc ** 2).sum() - 1.0)
        for i, j in itertools.combinations(range(len(ids)), 2):
            dist.append(np.linalg.norm(Pp[i] - Pp[j]) - np.linalg.norm(Q[i] - Q[j]))
        n_pts += len(ids)
        n_boards += 1
    if not n_boards:
        return {"boards": 0}
    se, de = np.asarray(scale), np.asarray(dist)
    return {"boards": n_boards, "corners": n_pts,
            "scale_err_pct_mean": float(se.mean() * 100),
            "dist_err_mm_rms": float(np.sqrt(np.mean(de ** 2)) * MM_PER_UNIT)}


def mccalib_rig(d, number_camera):
    cams, _ = _load_cameras(str(Path(d) / "calibrated_cameras_data.yml"))
    cameras = {c: _source_model(g) for c, g in cams.items() if c < number_camera}
    T = {c: np.linalg.inv(np.asarray(cams[c].pose, float)) for c in cameras}
    return RigState(cameras=cameras, T_c_g=T, ref_cam_id=min(cameras), object_poses={},
                    objects={0: _load_object(str(Path(d) / "calibrated_objects_data.yml"))})


def extrinsic_diff(rig, ref):
    out = {}
    for a, b in itertools.combinations(sorted(set(rig.T_c_g) & set(ref.T_c_g)), 2):
        Ta = rig.T_c_g[b] @ np.linalg.inv(rig.T_c_g[a])
        Tb = ref.T_c_g[b] @ np.linalg.inv(ref.T_c_g[a])
        R = Ta[:3, :3].T @ Tb[:3, :3]
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        dc = np.linalg.norm(Ta[:3, :3].T @ Ta[:3, 3] - Tb[:3, :3].T @ Tb[:3, 3])
        out[f"{a}-{b}"] = {"rot_deg": ang, "centre_mm": float(dc * MM_PER_UNIT)}
    return out


def _evaluate(rig, work, seq):
    cfg = load_config(str(_cfg_path(work, seq)))
    boards = pickle.loads((Path(work) / seq / "holdout_boards.pkl").read_bytes())
    return {"heldout": heldout_reprojection(rig, boards),
            "triangulation": heldout_triangulation(rig, boards, cfg.boards)}


# ----------------------------------------------------------------------------- trials
def plan(work, seed=0):
    trials = []
    for seq in sorted(p.name for p in Path(work).iterdir() if _cfg_path(work, p.name).exists()):
        cfg = load_config(str(_cfg_path(work, seq)))
        n = cfg.number_camera
        # each camera draws from the models valid for its own lens (config: Brown -> pinhole
        # pool, Kannala -> fisheye pool); a rig with a fisheye adds all-radtan as a control
        pools = [FISHEYE if m == "kb" else PINHOLE for m in cfg.camera_models]
        common = [m for m in pools[0] if all(m in p for p in pools)]
        key = int(hashlib.sha256(seq.encode()).hexdigest()[:8], 16) + seed
        rng = np.random.default_rng(key)
        trials += [{"dataset": seq, "trial": f"all_{m}", "kind": "single-model", "models": [m] * n}
                   for m in common]
        if any(p is FISHEYE for p in pools):
            trials.append({"dataset": seq, "trial": "all_radtan", "kind": "negative-control",
                           "models": ["radtan"] * n})
        seen = {tuple([m] * n) for m in common}
        space = int(np.prod([len(p) for p in pools]))
        k = 0
        while k < N_MIXED:
            if all(p is pools[0] for p in pools):
                draw = tuple(str(x) for x in rng.choice(pools[0], size=n))
            else:
                draw = tuple(str(rng.choice(p)) for p in pools)
            if draw in seen and space > len(seen):
                continue
            seen.add(draw)
            trials.append({"dataset": seq, "trial": f"mix_{k:02d}", "kind": "mixed",
                           "models": list(draw)})
            k += 1
    (Path(work) / "plan.json").write_text(json.dumps(trials, indent=1))
    print(len(trials), "trials")


def run(work, seq, trial_id):
    trial = next(t for t in json.loads((Path(work) / "plan.json").read_text())
                 if t["dataset"] == seq and t["trial"] == trial_id)
    out = Path(work) / "runs" / seq / trial_id
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(work) / seq / "detect" / "detected_keypoints_data.yml"
    sets = _overrides(work, seq, root_path="None", keypoints_path=str(cache), fix_intrinsic="0",
                      save_path=f"{out}/calib/") + ["camera_models=" + ",".join(trial["models"])]
    rec = dict(trial)
    t0 = time.monotonic()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = cli._run_config(str(_cfg_path(work, seq)), sets, pass_px=1.0, warn_px=2.0,
                                  report_path="")
        rig = res["rig"]
        rec.update(status="ok", cameras=len(rig.cameras), train_max_rms=res["metrics"]["max_rms_px"],
                   audit_weak=(res.get("audit") or {}).get("n_weak"),
                   worst_sigma=(res.get("uncertainty") or {}).get("worst"), **_evaluate(rig, work, seq))
        mcc = Path(work) / seq / "mccalib"
        if (mcc / "calibrated_cameras_data.yml").exists():
            diff = extrinsic_diff(rig, mccalib_rig(mcc, len(rig.cameras)))
            rec.update(extrinsic_vs_mccalib=diff,
                       max_rot_deg=max(v["rot_deg"] for v in diff.values()),
                       max_centre_mm=max(v["centre_mm"] for v in diff.values()))
    except Exception as exc:                                       # noqa: BLE001
        rec.update(status="error", error=repr(exc), traceback=traceback.format_exc()[-2000:])
    rec["seconds"] = time.monotonic() - t0
    (out / "result.json").write_text(json.dumps(rec, indent=1, default=float))
    print(seq, trial_id, rec["status"], rec.get("max_rot_deg"), (rec.get("heldout") or {}).get("rms_px"))


def ref(work, seq, mccalib_out):
    """Copy MC-Calib's result next to the sequence and evaluate it on the same held-out data."""
    dst = Path(work) / seq / "mccalib"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("calibrated_cameras_data.yml", "calibrated_objects_data.yml"):
        shutil.copy2(Path(mccalib_out) / f, dst / f)
    cfg = load_config(str(_cfg_path(work, seq)))
    res = {"dataset": seq, **_evaluate(mccalib_rig(dst, cfg.number_camera), work, seq)}
    (dst / "evaluation.json").write_text(json.dumps(res, indent=1))
    print(seq, "MC-Calib held-out", res["heldout"]["rms_px"])


def report(work):
    plan_ = json.loads((Path(work) / "plan.json").read_text())
    rows = []
    for t in plan_:
        p = Path(work) / "runs" / t["dataset"] / t["trial"] / "result.json"
        rows.append(json.loads(p.read_text()) if p.exists() else {**t, "status": "pending"})

    def span(xs, fmt):
        xs = [x for x in xs if x is not None]
        return "—" if not xs else (f"{fmt.format(statistics.median(xs))} "
                                   f"[{fmt.format(min(xs))}, {fmt.format(max(xs))}]")
    print("| rig | trials OK | Δrot vs MC-Calib deg | Δcentre vs MC-Calib mm | held-out RMS px "
          "(MC-Calib) | triangulated scale error % (MC-Calib) | corner distance error mm (MC-Calib) |")
    print("|---|---|---|---|---|---|---|")
    for seq in sorted({t["dataset"] for t in plan_}):
        rs = [r for r in rows if r["dataset"] == seq and r["kind"] != "negative-control"]
        ok = [r for r in rs if r["status"] == "ok"]
        mp = Path(work) / seq / "mccalib" / "evaluation.json"
        m = json.loads(mp.read_text()) if mp.exists() else {}
        mh = (m.get("heldout") or {}).get("rms_px")
        mt = m.get("triangulation") or {}
        print(f"| {seq} | {len(ok)}/{len(rs)} | {span([r.get('max_rot_deg') for r in ok], '{:.3f}')} | "
              f"{span([r.get('max_centre_mm') for r in ok], '{:.2f}')} | "
              f"{span([r['heldout']['rms_px'] for r in ok], '{:.3f}')} ({mh or float('nan'):.3f}) | "
              f"{span([r['triangulation'].get('scale_err_pct_mean') for r in ok], '{:+.3f}')} "
              f"({mt.get('scale_err_pct_mean', float('nan')):+.3f}) | "
              f"{span([r['triangulation'].get('dist_err_mm_rms') for r in ok], '{:.3f}')} "
              f"({mt.get('dist_err_mm_rms', float('nan')):.3f}) |")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"split": lambda: split(*args), "detect": lambda: detect(*args),
     "plan": lambda: plan(args[0], int(args[1]) if len(args) > 1 else 0),
     "run": lambda: run(*args), "ref": lambda: ref(*args), "report": lambda: report(*args)}[cmd]()
