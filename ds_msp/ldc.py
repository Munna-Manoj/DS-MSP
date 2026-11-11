"""
Texas Instruments (TI) Jacinto LDC displacement-mesh export for **any** camera model.

Generates a quantized displacement-mesh lookup table for the on-chip Lens
Distortion Correction (LDC) hardware accelerator on TI Jacinto J7/TDA4 SoCs
from any calibrated :class:`~ds_msp.core.contracts.CameraModel`, plus a point
undistorter that simulates the hardware's own mesh-inversion behavior. See
[Export a TI Jacinto LDC displacement mesh](../how-to/export_ldc_mesh.md).

The generator depends only on the model contract -- ``project()`` for the
distorted source location of every mesh node and ``K`` for the pinhole focal
that seeds the rectified intrinsics -- so every model that satisfies the
contract (the registered ones, models added in the future, and the legacy
:class:`~ds_msp.model.DoubleSphereCamera`) exports a mesh with no
model-specific code.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .core.contracts import CameraModel
from .core.pinhole import balanced_pinhole_K

#: Q3 fixed point: the hardware stores ``round(delta_px * 8)`` as ``int16``.
Q3_SCALE = 8.0
Q3_INT16_MIN, Q3_INT16_MAX = -32768, 32767


def _pinhole_focal(camera: Any) -> Tuple[float, float]:
    """``(fx, fy)`` of the model's pinhole-equivalent ``K`` (contract property).

    Every :class:`CameraModel` exposes ``K``; a model without a native focal
    returns its pinhole-equivalent focal there, so the rectified
    ``K_new`` seeds identically for every model. Falls back to ``fx``/``fy``
    attributes for duck-typed legacy objects.
    """
    K = getattr(camera, "K", None)
    if K is not None:
        K = np.asarray(K, dtype=np.float64)
        return float(K[0, 0]), float(K[1, 1])
    if hasattr(camera, "fx") and hasattr(camera, "fy"):
        return float(camera.fx), float(camera.fy)
    raise TypeError(
        f"{type(camera).__name__} exposes neither a pinhole K nor fx/fy; it does not "
        "satisfy the DS-MSP CameraModel contract"
    )


def _describe_camera(camera: Any) -> Dict[str, Any]:
    """Self-describing record of the source camera for the flashed config."""
    to_dict = getattr(camera, "to_dict", None)
    if callable(to_dict):
        record = dict(to_dict())
        name = record.pop("model", None) or getattr(camera, "name", type(camera).__name__)
        return {"name": str(name), "params": {k: float(v) for k, v in record.items()}}
    # Legacy duck-typed camera (no contract serialization): record what it exposes.
    record: Dict[str, Any] = {"name": type(camera).__name__}
    K = getattr(camera, "K", None)
    if K is not None:
        record["K"] = np.asarray(K, dtype=np.float64).tolist()
    D = getattr(camera, "D", None)
    if D is not None:
        record["distortion"] = np.asarray(D, dtype=np.float64).ravel().tolist()
    return record


class TI_LDC_MeshGenerator:
    """
    Texas Instruments (TI) Lens Distortion Correction (LDC) Mesh LUT Generator.

    Generates downsampled displacement mesh lookup tables compatible with
    TI Jacinto J7/TDA4 hardware accelerators from **any** calibrated
    :class:`~ds_msp.core.contracts.CameraModel` -- Double Sphere, UCM, EUCM,
    Kannala-Brandt, RadTan, and any model added later that
    implements the contract. See
    [Export a TI Jacinto LDC displacement mesh](../how-to/export_ldc_mesh.md)
    for a worked example with real mesh numbers.

    Parameters
    ----------
    camera : CameraModel
        The calibrated camera to generate a mesh for. Only ``project()`` and
        ``K`` are used. Any ``width``/``height`` attributes on the model are
        **not** used by this class — the mesh is sized from the
        ``output_width``/``output_height`` arguments passed to
        :meth:`generate_mesh_and_intrinsics`.
    """
    def __init__(self, camera: CameraModel) -> None:
        if not callable(getattr(camera, "project", None)):
            raise TypeError(
                f"{type(camera).__name__} has no project(); it does not satisfy the "
                "DS-MSP CameraModel contract"
            )
        _pinhole_focal(camera)  # fail fast with a clear message
        self.cam = camera

    def generate_mesh_and_intrinsics(
        self,
        output_width: int,
        output_height: int,
        downsample_factor: int = 4,
        balance: float = 0.5,
    ) -> Dict:
        """
        Generate the LDC displacement mesh (Q3 fixed point) and rectified intrinsics.

        Samples one mesh node every ``2**downsample_factor`` output pixels (plus a
        one-node border), computing at each node the pixel displacement between
        the undistorted (pinhole) location and its distorted source location
        under the camera model, quantized to Q3 fixed point
        (``round(delta_px * 8)``, ``int16``).

        Parameters
        ----------
        output_width, output_height : int
            Size of the undistorted (on-chip output) image, pixels. Independent
            of the camera's own sensor size.
        downsample_factor : int, default=4
            Power-of-two mesh node spacing: nodes are sampled every
            ``2**downsample_factor`` output pixels. Larger values give a
            coarser, smaller LUT; smaller values a denser, more accurate one.
        balance : float, default=0.5
            Field-of-view/border trade-off in ``[0, 1]`` passed to
            :func:`~ds_msp.core.pinhole.balanced_pinhole_K` to build ``K_new``
            from the model's pinhole focal (``K[0, 0]``, ``K[1, 1]``) -- the
            same ``K_new`` :class:`~ds_msp.ops.undistort.Undistorter` builds
            for that model and balance; ``0.0`` widest FOV, ``1.0`` tightest
            crop.

        Returns
        -------
        dict
            With keys:

            - ``mesh_lut`` : ndarray of shape (mesh_h, mesh_w, 2), int16 —
              quantized Q3 ``(h, v)`` displacements (pixels x 8, rounded,
              clipped to the ``int16`` range).
            - ``mesh_lut_float`` : ndarray of shape (mesh_h, mesh_w, 2), float64 —
              the same displacements before quantization.
            - ``K_new`` : ndarray of shape (3, 3), float64 — rectified pinhole
              intrinsics of the undistorted output image.
            - ``valid_mask`` : ndarray of shape (mesh_h, mesh_w), bool —
              ``False`` at nodes whose pinhole ray the model cannot project
              (outside its field of view); those nodes hold zero displacement.
            - ``config`` : dict — the call parameters, resulting ``mesh_size``,
              the source ``camera_model`` (name + parameters),
              ``n_invalid_nodes`` and ``q3_overflow``, for a self-describing
              record to flash alongside the mesh.

        Warns
        -----
        UserWarning
            If any mesh node is invalid for the model (raise ``balance`` to
            crop the periphery), or if any displacement exceeds the ``int16``
            Q3 range (``[-4096, 4095.875]`` px) -- such values are clipped, not
            wrapped, and ``config["q3_overflow"]`` is set.
        """
        K_new = self._compute_K_new(output_width, output_height, balance)
        mesh_lut_int, mesh_lut_float, valid, overflow = self._generate_mesh(
            output_width, output_height, K_new, downsample_factor
        )
        n_invalid = int((~valid).sum())
        if n_invalid:
            warnings.warn(
                f"{n_invalid} of {valid.size} LDC mesh nodes are invalid for this camera "
                "model (outside its field of view); they hold zero displacement. Raise "
                "`balance` to crop the periphery.",
                stacklevel=2,
            )
        if overflow:
            warnings.warn(
                "LDC mesh displacements exceed the int16 Q3 range and were clipped; "
                "raise `balance` (or lower the output resolution) before flashing.",
                stacklevel=2,
            )
        config: Dict[str, Any] = {
            "output_width": output_width,
            "output_height": output_height,
            "downsample_factor": downsample_factor,
            "balance": balance,
            "mesh_size": mesh_lut_int.shape,
            "camera_model": _describe_camera(self.cam),
            "n_invalid_nodes": n_invalid,
            "q3_overflow": bool(overflow),
        }
        params = config["camera_model"].get("params")
        if isinstance(params, dict) and {"xi", "alpha"} <= set(params):
            # Backward-compatible alias for consumers that read the pre-generalization
            # Double Sphere record; ``camera_model`` is the canonical entry.
            config["double_sphere_params"] = {
                k: params[k] for k in ("fx", "fy", "cx", "cy", "xi", "alpha") if k in params
            }
        return {
            "mesh_lut": mesh_lut_int,
            "mesh_lut_float": mesh_lut_float,
            "K_new": K_new,
            "valid_mask": valid,
            "config": config,
        }

    def _compute_K_new(self, width: int, height: int, balance: float) -> np.ndarray:
        # Same seed as ops.undistort.Undistorter.new_K (the model's pinhole focal),
        # so the mesh and the software undistorter share one rectified frame; LDC may
        # target an output resolution different from the sensor, so width/height
        # are explicit.
        fx, fy = _pinhole_focal(self.cam)
        return balanced_pinhole_K(fx, fy, width, height, balance)

    def _generate_mesh(
        self, width: int, height: int, K_new: np.ndarray, m: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        fx_new, fy_new = K_new[0, 0], K_new[1, 1]
        cx_new, cy_new = K_new[0, 2], K_new[1, 2]
        step = 2**m

        # Pad grid to next multiple of step to prevent out-of-bounds at image boundaries
        padded_width = ((width + step - 1) // step) * step
        padded_height = ((height + step - 1) // step) * step

        # Only the node locations are evaluated (every `step` pixels, plus the border).
        h_undist, v_undist = np.meshgrid(
            np.arange(0, padded_width + 1, step, dtype=np.float64),
            np.arange(0, padded_height + 1, step, dtype=np.float64),
            indexing="xy",
        )
        mx = (h_undist - cx_new) / fx_new
        my = (v_undist - cy_new) / fy_new
        rays = np.stack([mx, my, np.ones_like(mx)], axis=-1)
        # Not normalised: every contract model's project() is scale-invariant.

        distorted_pts, valid = self.cam.project(rays)
        distorted_pts = np.asarray(distorted_pts, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool) & np.all(np.isfinite(distorted_pts), axis=-1)

        mesh_float = np.zeros(h_undist.shape + (2,), dtype=np.float64)
        mesh_float[..., 0] = distorted_pts[..., 0] - h_undist
        mesh_float[..., 1] = distorted_pts[..., 1] - v_undist
        mesh_float[~valid] = 0.0

        q3 = np.round(mesh_float * Q3_SCALE)
        overflow = bool((q3 < Q3_INT16_MIN).any() or (q3 > Q3_INT16_MAX).any())
        mesh_int = np.clip(q3, Q3_INT16_MIN, Q3_INT16_MAX).astype(np.int16)
        return mesh_int, mesh_float, valid, overflow


class TI_LDC_PointUndistorter:
    """
    Simulates TI J7 LDC hardware displacement interpolation to undistort points.

    Inverts a mesh produced by :meth:`TI_LDC_MeshGenerator.generate_mesh_and_intrinsics`
    to recover undistorted (pinhole) coordinates for distorted input pixels, by the
    same bilinear-interpolation + fixed-point-iteration approach the LDC hardware
    itself performs. **Prefer the closed form** for anything other than
    reproducing hardware behavior:
    :meth:`ds_msp.ops.undistort.Undistorter.undistort_points` for any model (or
    :meth:`~ds_msp.model.DoubleSphereCamera.undistort_points` on the legacy
    class) — the mesh inverse is exact only near the image center and diverges
    sharply toward the periphery (measured ~0.08 px median disagreement
    overall, ~80 px at the corners in a representative 1920x1080
    configuration; see
    [Export a TI Jacinto LDC displacement mesh](../how-to/export_ldc_mesh.md)).

    Parameters
    ----------
    mesh_lut_float : ndarray of shape (mesh_h, mesh_w, 2), float64
        The unquantized displacement mesh, i.e. ``generate_mesh_and_intrinsics``'s
        ``mesh_lut_float`` output.
    K_new : ndarray of shape (3, 3)
        The rectified pinhole intrinsics the mesh was generated for (same
        ``generate_mesh_and_intrinsics`` call's ``K_new``).
    downsample_factor : int
        The power-of-two mesh node spacing used to generate ``mesh_lut_float``
        (node spacing is ``2**downsample_factor`` output pixels).
    output_width, output_height : int
        Size of the undistorted (output) image the mesh targets.
    """
    def __init__(
        self,
        mesh_lut_float: np.ndarray,
        K_new: np.ndarray,
        downsample_factor: int,
        output_width: int,
        output_height: int,
    ) -> None:
        self.mesh = mesh_lut_float
        self.K_new = K_new
        self.step = 2**downsample_factor
        self.output_width = output_width
        self.output_height = output_height
        self.mesh_height, self.mesh_width = mesh_lut_float.shape[:2]

    def undistort_points(
        self, points_distorted: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Invert the LDC displacement mesh for a batch of distorted points.

        Fully vectorized fixed-point solve: every point iterates together, and a
        point drops out of the active set once it converges (residual < 0.01 px)
        or its current estimate leaves the mesh. Equivalent to the per-point
        Newton-free iteration the J7 LDC performs, but array-wide. Runs at most
        10 iterations.

        Parameters
        ----------
        points_distorted : ndarray of shape (N, 2)
            Distorted pixel coordinates in the original fisheye image.

        Returns
        -------
        guess : ndarray of shape (N, 2)
            Estimated undistorted (pinhole, ``K_new``-frame) coordinates. Rows
            marked invalid still hold the last iterate, not NaN.
        valid : ndarray of shape (N,), bool
            ``False`` where the fixed-point iterate left the mesh bounds before
            converging, did not converge within the 10 iterations, or the result falls outside
            ``[0, output_width) x [0, output_height)``.
        """
        pts = np.asarray(points_distorted, dtype=np.float64)
        N = len(pts)
        target = pts.copy()                 # distorted coords we want to match
        guess = pts.copy()                  # current undistorted estimate
        valid = np.ones(N, dtype=bool)
        active = np.ones(N, dtype=bool)

        for _ in range(10):
            if not active.any():
                break
            delta, in_bounds = self._interpolate_mesh_batch(guess)

            # Points whose estimate fell outside the mesh are unrecoverable.
            lost = active & ~in_bounds
            valid[lost] = False
            active[lost] = False

            err = target - (guess + delta)          # residual for all points
            guess[active] += err[active]            # update only active points

            converged = active & (np.hypot(err[:, 0], err[:, 1]) < 0.01)
            active[converged] = False

        valid[active] = False               # still moving after the last iteration
        # Final estimate must land inside the output (undistorted) image.
        out_of_frame = (
            (guess[:, 0] < 0) | (guess[:, 0] >= self.output_width)
            | (guess[:, 1] < 0) | (guess[:, 1] >= self.output_height)
        )
        valid[out_of_frame] = False
        return guess, valid

    def _interpolate_mesh_batch(
        self, pts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bilinearly sample the displacement mesh at (N, 2) point locations."""
        m = pts / self.step
        m0 = np.floor(m).astype(np.int64)
        frac = m - m0
        mx0, my0 = m0[:, 0], m0[:, 1]
        fx, fy = frac[:, 0], frac[:, 1]

        in_bounds = (
            (mx0 >= 0) & (mx0 < self.mesh_width - 1)
            & (my0 >= 0) & (my0 < self.mesh_height - 1)
        )
        # Clamp indices so the gather is always safe; out-of-bounds rows are
        # discarded by the in_bounds mask returned to the caller.
        mx0c = np.clip(mx0, 0, self.mesh_width - 2)
        my0c = np.clip(my0, 0, self.mesh_height - 2)

        Q00 = self.mesh[my0c, mx0c]
        Q10 = self.mesh[my0c, mx0c + 1]
        Q01 = self.mesh[my0c + 1, mx0c]
        Q11 = self.mesh[my0c + 1, mx0c + 1]

        w00 = ((1.0 - fx) * (1.0 - fy))[:, None]
        w10 = (fx * (1.0 - fy))[:, None]
        w01 = ((1.0 - fx) * fy)[:, None]
        w11 = (fx * fy)[:, None]

        delta = w00 * Q00 + w10 * Q10 + w01 * Q01 + w11 * Q11
        return delta, in_bounds


#: Version of the files :func:`write_ldc_mesh` writes; bump on any layout change.
LDC_FILE_FORMAT = 1

#: How :func:`write_ldc_mesh` lays out the mesh bytes. Not yet checked on a board.
LDC_MESH_LAYOUT = (
    "int16 little-endian, (h, v) interleaved per node, nodes row by row from the "
    "top-left, no header and no row padding; values are Q3 (pixels x 8)"
)


def mesh_accuracy(
    camera: CameraModel,
    result: Dict,
    *,
    input_size: Optional[Tuple[int, int]] = None,
    stride: Optional[int] = None,
) -> Dict[str, Any]:
    """Measure how far the stored mesh sends each output pixel from where it should.

    For each output pixel the LDC hardware reads the input at the pixel plus the
    bilinearly interpolated Q3 displacement. The exact source is
    ``camera.project`` of the pixel's ray under ``K_new``. The error is the
    distance between the two, in input pixels, so it includes both the mesh
    spacing and the Q3 rounding.

    Parameters
    ----------
    camera : CameraModel
        The camera the mesh was generated for.
    result : dict
        The output of :meth:`TI_LDC_MeshGenerator.generate_mesh_and_intrinsics`.
    input_size : (width, height), optional
        The camera's own image size. When given, ``in_view_fraction`` reports the
        share of output pixels whose source lies inside the input image.
    stride : int, optional
        Evaluate every ``stride``-th output pixel in each direction. Defaults to 1,
        or 2 above 4 megapixels.

    Returns
    -------
    dict
        ``median_px``, ``p99_px`` and ``max_px`` over the measured pixels,
        ``measured_fraction`` (output pixels the camera can see and the mesh
        covers with four valid nodes), ``in_view_fraction`` (or ``None``),
        ``stride``, and ``worst_output_px`` (where ``max_px`` occurs).
    """
    cfg = result["config"]
    width, height = int(cfg["output_width"]), int(cfg["output_height"])
    step = 2 ** int(cfg["downsample_factor"])
    if stride is None:
        stride = 1 if width * height <= 4_000_000 else 2
    K_new = np.asarray(result["K_new"], dtype=np.float64)
    mesh = np.asarray(result["mesh_lut"], dtype=np.float64) / Q3_SCALE
    valid = np.asarray(result["valid_mask"], dtype=bool)

    grid_u, grid_v = np.meshgrid(
        np.arange(0, width, stride, dtype=np.float64),
        np.arange(0, height, stride, dtype=np.float64),
        indexing="xy",
    )
    u, v = grid_u.ravel(), grid_v.ravel()
    rays = np.stack(
        [(u - K_new[0, 2]) / K_new[0, 0], (v - K_new[1, 2]) / K_new[1, 1], np.ones_like(u)],
        axis=-1,
    )
    exact, seen = camera.project(rays)
    exact = np.asarray(exact, dtype=np.float64)
    seen = np.asarray(seen, dtype=bool) & np.all(np.isfinite(exact), axis=-1)

    mx, my = u / step, v / step
    x0 = np.minimum(np.floor(mx).astype(np.int64), mesh.shape[1] - 2)
    y0 = np.minimum(np.floor(my).astype(np.int64), mesh.shape[0] - 2)
    fx, fy = (mx - x0)[:, None], (my - y0)[:, None]
    corners_valid = valid[y0, x0] & valid[y0, x0 + 1] & valid[y0 + 1, x0] & valid[y0 + 1, x0 + 1]
    delta = (
        (1 - fx) * (1 - fy) * mesh[y0, x0] + fx * (1 - fy) * mesh[y0, x0 + 1]
        + (1 - fx) * fy * mesh[y0 + 1, x0] + fx * fy * mesh[y0 + 1, x0 + 1]
    )
    hardware = np.stack([u, v], axis=-1) + delta
    measured = seen & corners_valid
    error = np.linalg.norm(hardware - exact, axis=-1)

    in_view = None
    if input_size is not None:
        iw, ih = int(input_size[0]), int(input_size[1])
        inside = (
            seen & (exact[:, 0] >= -0.5) & (exact[:, 0] <= iw - 0.5)
            & (exact[:, 1] >= -0.5) & (exact[:, 1] <= ih - 0.5)
        )
        in_view = float(inside.mean())

    report: Dict[str, Any] = {
        "median_px": None, "p99_px": None, "max_px": None,
        "measured_fraction": float(measured.mean()),
        "in_view_fraction": in_view,
        "stride": int(stride),
        "worst_output_px": None,
    }
    if measured.any():
        e = error[measured]
        worst = int(np.flatnonzero(measured)[int(np.argmax(e))])
        report.update(
            median_px=float(np.median(e)),
            p99_px=float(np.percentile(e, 99)),
            max_px=float(e.max()),
            worst_output_px=[float(u[worst]), float(v[worst])],
        )
    return report


@dataclass(frozen=True)
class LdcFiles:
    """The two files :func:`write_ldc_mesh` writes."""

    mesh: Path
    record: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_ldc_mesh(
    result: Dict,
    output_dir: str | Path,
    stem: str,
    *,
    accuracy: Optional[Dict[str, Any]] = None,
    source: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> LdcFiles:
    """Write a mesh as ``<stem>_ldc_mesh.bin`` plus a JSON record ``<stem>_ldc.json``.

    The binary holds the Q3 mesh in the layout :data:`LDC_MESH_LAYOUT` describes.
    The record holds everything needed to use or check it: mesh and output sizes,
    node spacing, ``K_new``, the camera, invalid nodes, Q3 overflow, the measured
    accuracy, where the camera came from, and the binary's SHA-256. Writing the
    same mesh twice gives byte-identical files.

    Raises
    ------
    FileExistsError
        If either file exists and ``overwrite`` is false.
    """
    if not stem or any(ch in stem for ch in "/\\") or stem in (".", ".."):
        raise ValueError(f"stem must be a plain file name, got {stem!r}")
    out = Path(output_dir).expanduser()
    mesh_path = out / f"{stem}_ldc_mesh.bin"
    record_path = out / f"{stem}_ldc.json"
    existing = [p for p in (mesh_path, record_path) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{existing[0]} already exists; choose another folder or name, or allow overwrite"
        )
    out.mkdir(parents=True, exist_ok=True)
    mesh = np.ascontiguousarray(np.asarray(result["mesh_lut"], dtype="<i2"))
    mesh_path.write_bytes(mesh.tobytes())

    cfg = result["config"]
    record = {
        "format": "ds-msp ti-ldc mesh",
        "format_version": LDC_FILE_FORMAT,
        "mesh_file": mesh_path.name,
        "mesh_sha256": _sha256(mesh_path),
        "layout": LDC_MESH_LAYOUT,
        "checked_on_board": False,
        "mesh_width": int(mesh.shape[1]),
        "mesh_height": int(mesh.shape[0]),
        "output_width": int(cfg["output_width"]),
        "output_height": int(cfg["output_height"]),
        "downsample_factor": int(cfg["downsample_factor"]),
        "node_spacing_px": 2 ** int(cfg["downsample_factor"]),
        "balance": float(cfg["balance"]),
        "K_new": _plain(np.asarray(result["K_new"], dtype=np.float64)),
        "camera_model": _plain(cfg["camera_model"]),
        "n_invalid_nodes": int(cfg["n_invalid_nodes"]),
        "q3_overflow": bool(cfg["q3_overflow"]),
        "accuracy": _plain(accuracy) if accuracy is not None else None,
        "source": _plain(source) if source is not None else None,
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return LdcFiles(mesh_path, record_path)


def read_ldc_mesh(record_path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a mesh written by :func:`write_ldc_mesh`, checking its size and checksum.

    Returns the ``(mesh_height, mesh_width, 2)`` ``int16`` mesh and the record.
    """
    record_path = Path(record_path).expanduser()
    record = json.loads(record_path.read_text())
    if record.get("format") != "ds-msp ti-ldc mesh":
        raise ValueError(f"{record_path} is not a DS-MSP LDC mesh record")
    if int(record.get("format_version", 0)) > LDC_FILE_FORMAT:
        raise ValueError(
            f"{record_path} uses format {record['format_version']}; this DS-MSP reads up to "
            f"{LDC_FILE_FORMAT}"
        )
    mesh_path = record_path.parent / record["mesh_file"]
    if _sha256(mesh_path) != record["mesh_sha256"]:
        raise ValueError(f"{mesh_path} does not match the checksum in {record_path.name}")
    h, w = int(record["mesh_height"]), int(record["mesh_width"])
    raw = np.frombuffer(mesh_path.read_bytes(), dtype="<i2")
    if raw.size != h * w * 2:
        raise ValueError(f"{mesh_path} holds {raw.size} values; the record says {h * w * 2}")
    return raw.reshape(h, w, 2).astype(np.int16), record


@dataclass(frozen=True)
class LdcExport:
    """What :func:`export_ldc` wrote and measured."""

    files: LdcFiles
    accuracy: Dict[str, Any]
    config: Dict[str, Any]
    notes: Tuple[str, ...]


def export_ldc(
    camera: CameraModel,
    input_size: Tuple[int, int],
    output_dir: str | Path,
    stem: str,
    *,
    output_size: Optional[Tuple[int, int]] = None,
    downsample_factor: int = 4,
    balance: float = 0.5,
    overwrite: bool = False,
    source: Optional[Dict[str, Any]] = None,
) -> LdcExport:
    """Generate, measure and write one camera's LDC mesh.

    ``input_size`` is the camera's own image size; ``output_size`` (default: the
    same) is the corrected image the board outputs. The generator's warnings
    (invalid nodes, Q3 overflow) are returned as ``notes`` instead of raised, so
    a caller exporting many cameras can report each one.
    """
    if not 0 <= int(downsample_factor) <= 7:
        raise ValueError("downsample_factor must be between 0 and 7")
    if not 0.0 <= float(balance) <= 1.0:
        raise ValueError("balance must be between 0 and 1")
    out_w, out_h = output_size or input_size
    if min(int(out_w), int(out_h)) <= 0:
        raise ValueError("output size must be positive")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = TI_LDC_MeshGenerator(camera).generate_mesh_and_intrinsics(
            int(out_w), int(out_h), int(downsample_factor), float(balance)
        )
    accuracy = mesh_accuracy(camera, result, input_size=input_size)
    record_source = dict(source or {})
    record_source.update(input_width=int(input_size[0]), input_height=int(input_size[1]))
    files = write_ldc_mesh(
        result, output_dir, stem, accuracy=accuracy, source=record_source, overwrite=overwrite
    )
    return LdcExport(files, accuracy, result["config"], tuple(str(w.message) for w in caught))
