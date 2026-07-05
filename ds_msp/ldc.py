"""
Texas Instruments (TI) Jacinto LDC displacement-mesh export.

Generates a quantized displacement-mesh lookup table for the on-chip Lens
Distortion Correction (LDC) hardware accelerator on TI Jacinto J7/TDA4 SoCs,
from a calibrated :class:`~ds_msp.model.DoubleSphereCamera`, plus a point
undistorter that simulates the hardware's own mesh-inversion behavior. See
[Export a TI Jacinto LDC displacement mesh](../how-to/export_ldc_mesh.md).
"""

import numpy as np
from typing import Tuple, Dict
from .model import DoubleSphereCamera, balanced_pinhole_K

class TI_LDC_MeshGenerator:
    """
    Texas Instruments (TI) Lens Distortion Correction (LDC) Mesh LUT Generator.

    Generates downsampled displacement mesh lookup tables compatible with
    TI Jacinto J7/TDA4 hardware accelerators, from a calibrated
    :class:`~ds_msp.model.DoubleSphereCamera`. See
    [Export a TI Jacinto LDC displacement mesh](../how-to/export_ldc_mesh.md)
    for a worked example with real mesh numbers.

    Parameters
    ----------
    ds_camera : DoubleSphereCamera
        The calibrated fisheye camera to generate a mesh for. Its
        ``width``/``height`` attributes are **not** used by this class — the
        mesh is sized from the ``output_width``/``output_height`` arguments
        passed to :meth:`generate_mesh_and_intrinsics`.
    """
    def __init__(self, ds_camera: DoubleSphereCamera) -> None:
        self.cam = ds_camera

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
        the undistorted (pinhole) location and its Double Sphere-distorted
        source location, quantized to Q3 fixed point (``round(delta_px * 8)``,
        ``int16``).

        Parameters
        ----------
        output_width, output_height : int
            Size of the undistorted (on-chip output) image, pixels. Independent
            of ``ds_camera``'s own ``width``/``height``.
        downsample_factor : int, default=4
            Power-of-two mesh node spacing: nodes are sampled every
            ``2**downsample_factor`` output pixels. Larger values give a
            coarser, smaller LUT; smaller values a denser, more accurate one.
        balance : float, default=0.5
            Field-of-view/border trade-off in ``[0, 1]`` passed to
            :func:`~ds_msp.core.pinhole.balanced_pinhole_K` to build ``K_new``;
            ``0.0`` widest FOV, ``1.0`` tightest crop.

        Returns
        -------
        dict
            With keys:

            - ``mesh_lut`` : ndarray of shape (mesh_h, mesh_w, 2), int16 —
              quantized Q3 ``(h, v)`` displacements (pixels x 8, rounded).
            - ``mesh_lut_float`` : ndarray of shape (mesh_h, mesh_w, 2), float64 —
              the same displacements before quantization.
            - ``K_new`` : ndarray of shape (3, 3), float64 — rectified pinhole
              intrinsics of the undistorted output image.
            - ``config`` : dict — the call parameters, resulting ``mesh_size``,
              and the source Double Sphere intrinsics, for a self-describing
              record to flash alongside the mesh.

        Notes
        -----
        Overflow of the ``int16`` Q3 range (``[-32768, 32767]``) is **not**
        checked or raised here — a large enough field of view produces
        displacements that silently wrap to the wrong sign. Inspect
        ``mesh_lut.min()``/``.max()`` after generation, especially for
        aggressive fields of view (see the how-to guide's Warning).
        """
        K_new = self._compute_K_new(output_width, output_height, balance)
        mesh_lut_int, mesh_lut_float = self._generate_mesh(
            output_width, output_height, K_new, downsample_factor
        )
        return {
            "mesh_lut": mesh_lut_int,
            "mesh_lut_float": mesh_lut_float,
            "K_new": K_new,
            "config": {
                "output_width": output_width,
                "output_height": output_height,
                "downsample_factor": downsample_factor,
                "balance": balance,
                "mesh_size": mesh_lut_int.shape,
                "double_sphere_params": {
                    "fx": self.cam.fx,
                    "fy": self.cam.fy,
                    "cx": self.cam.cx,
                    "cy": self.cam.cy,
                    "xi": self.cam.xi,
                    "alpha": self.cam.alpha,
                },
            },
        }

    def _compute_K_new(self, width: int, height: int, balance: float) -> np.ndarray:
        # Shared with DoubleSphereCamera.compute_K_new; LDC may target an output
        # resolution different from the sensor, so width/height are explicit.
        return balanced_pinhole_K(self.cam.fx, self.cam.fy, width, height, balance)

    def _generate_mesh(
        self, width: int, height: int, K_new: np.ndarray, m: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        fx_new, fy_new = K_new[0, 0], K_new[1, 1]
        cx_new, cy_new = K_new[0, 2], K_new[1, 2]
        step = 2**m

        # Pad grid to next multiple of step to prevent out-of-bounds at image boundaries
        padded_width = ((width + step - 1) // step) * step
        padded_height = ((height + step - 1) // step) * step

        h_undist, v_undist = np.meshgrid(
            np.arange(0, padded_width + 1, dtype=np.float64),
            np.arange(0, padded_height + 1, dtype=np.float64),
            indexing="xy",
        )
        mx = (h_undist - cx_new) / fx_new
        my = (v_undist - cy_new) / fy_new
        rays = np.stack([mx, my, np.ones_like(mx)], axis=-1)
        # Omit redundant norm computation since ds_project is scale-invariant

        distorted_pts, _ = self.cam.project(rays)
        h_distorted = distorted_pts[..., 0]
        v_distorted = distorted_pts[..., 1]

        delta_h_float = h_distorted - h_undist
        delta_v_float = v_distorted - v_undist

        delta_h_q3 = np.round(delta_h_float * 8.0).astype(np.int16)
        delta_v_q3 = np.round(delta_v_float * 8.0).astype(np.int16)

        h_down = delta_h_q3[::step, ::step]
        v_down = delta_v_q3[::step, ::step]
        h_float_down = delta_h_float[::step, ::step]
        v_float_down = delta_v_float[::step, ::step]

        mesh_height, mesh_width = h_down.shape
        mesh_int = np.zeros((mesh_height, mesh_width, 2), dtype=np.int16)
        mesh_int[..., 0] = h_down
        mesh_int[..., 1] = v_down

        mesh_float = np.zeros((mesh_height, mesh_width, 2), dtype=np.float64)
        mesh_float[..., 0] = h_float_down
        mesh_float[..., 1] = v_float_down
        return mesh_int, mesh_float


class TI_LDC_PointUndistorter:
    """
    Simulates TI J7 LDC hardware displacement interpolation to undistort points.

    Inverts a mesh produced by :meth:`TI_LDC_MeshGenerator.generate_mesh_and_intrinsics`
    to recover undistorted (pinhole) coordinates for distorted input pixels, by the
    same bilinear-interpolation + fixed-point-iteration approach the LDC hardware
    itself performs. **Prefer the closed-form**
    :meth:`~ds_msp.model.DoubleSphereCamera.undistort_points` for anything other
    than reproducing hardware behavior — the mesh inverse is exact only near the
    image center and diverges sharply toward the periphery (measured ~0.08 px
    median disagreement overall, ~80 px at the corners in a representative
    1920x1080 configuration; see
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
