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

import warnings
from typing import Any, Dict, Tuple

import numpy as np

from .core.contracts import CameraModel
from .core.pinhole import balanced_pinhole_K

#: Q3 fixed point: the hardware stores ``round(delta_px * 8)`` as ``int16``.
Q3_SCALE = 8.0
Q3_INT16_MIN, Q3_INT16_MAX = -32768, 32767


def _pinhole_focal(camera: Any) -> Tuple[float, float]:
    """``(fx, fy)`` of the model's pinhole-equivalent ``K`` (contract property).

    Every :class:`CameraModel` exposes ``K``; models without a native focal
    (e.g. OCam) return their pinhole-equivalent focal there, so the rectified
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
    Kannala-Brandt, RadTan, OCam, DS⁺, and any model added later that
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
            converging, or the converged result falls outside
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
