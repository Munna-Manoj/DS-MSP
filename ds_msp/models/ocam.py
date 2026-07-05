"""OCamCalib / Scaramuzza omnidirectional polynomial model."""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .ocam_math import R_REF, ocam_project, ocam_project_jacobian, ocam_unproject


class OCamModel:
    """Scaramuzza polynomial omni-camera. Satisfies ``CameraModel``.

    Parameters: centre (cx, cy), affine (c, d, e), world polynomial (a0..a4).
    Has no native fx/fy; ``K`` exposes a pinhole-equivalent focal ~ |a0|.
    """

    name: ClassVar[str] = "ocam"
    param_names: ClassVar[Tuple[str, ...]] = (
        "cx", "cy", "c", "d", "e", "a0", "a1", "a2", "a3", "a4")

    def __init__(self, cx, cy, c=1.0, d=0.0, e=0.0,
                 a0=-200.0, a1=0.0, a2=0.0, a3=0.0, a4=0.0) -> None:
        self.cx, self.cy = float(cx), float(cy)
        self.c, self.d, self.e = float(c), float(d), float(e)
        self.a0, self.a1, self.a2, self.a3, self.a4 = map(float, (a0, a1, a2, a3, a4))

    @classmethod
    def sample(cls) -> "OCamModel":
        """Realistic instance for contract testing (a synthetic omni-camera).

        Polynomial argument is normalized by ``R_REF=100``, so ``a2..a4``
        are O(1).
        """
        return cls(320.0, 240.0, 1.0, 0.0, 0.0, -220.0, 0.0, 6.0, -1.5, 0.25)

    def _p(self):
        return (self.cx, self.cy, self.c, self.d, self.e,
                self.a0, self.a1, self.a2, self.a3, self.a4)

    @property
    def params(self) -> np.ndarray:
        """Flat parameter vector ``[cx, cy, c, d, e, a0, a1, a2, a3, a4]``."""
        return np.array(self._p(), dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        """Pinhole-equivalent 3x3 intrinsic matrix; focal is ``|a0|`` (OCam has no native fx/fy)."""
        f = abs(self.a0)
        return np.array([[f, 0.0, self.cx], [0.0, f, self.cy], [0.0, 0.0, 1.0]],
                        dtype=np.float64)

    @property
    def distortion(self) -> np.ndarray:
        """Distortion tail ``[c, d, e, a0, a1, a2, a3, a4]`` (affine + world polynomial)."""
        return np.array([self.c, self.d, self.e,
                         self.a0, self.a1, self.a2, self.a3, self.a4], dtype=np.float64)

    def project(self, P):
        """Project camera-frame points via the Scaramuzza world polynomial.

        OCam has no closed-form projection: instead of a distortion formula
        applied to a normalized ray, it solves for the radial image distance
        ``rho`` satisfying ``w(rho) + m*rho = 0`` by Newton iteration, where
        ``w`` is a degree-4 "world" polynomial in ``rho / R_REF`` and
        ``m = Z / sqrt(X^2 + Y^2)``. The 2x2 affine stretch ``[[c, d], [e,
        1]]`` is then applied to model sensor non-squareness / skew before
        translation by the principal point. This general polynomial radial
        profile (rather than a fixed functional form) is what lets OCam fit
        distortion curves other closed-form models cannot.

        Parameters
        ----------
        P : ndarray, shape (..., 3)
            Camera-frame points (meters), +Z forward.

        Returns
        -------
        uv : ndarray, shape (..., 2)
            Projected pixel coordinates ``(u, v)``, origin top-left.
            On-axis points (``sqrt(X^2+Y^2) < 1e-12``) map to the principal
            point.
        valid : ndarray, shape (...,)
            ``True`` iff the Newton solve for ``rho`` converged
            (``|w(rho) + m*rho| < 1e-6``) or the point is on-axis. See
            ``ds_msp.models.ocam_math.ocam_project``.

        References
        ----------
        Scaramuzza, D., Martinelli, A., Siegwart, R. "A Toolbox for Easily
        Calibrating Omnidirectional Cameras." IROS 2006.

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import OCamModel
        >>> m = OCamModel.sample()
        >>> uv, valid = m.project(np.array([[0.0, 0.0, 1.0]]))
        >>> np.round(uv, 2)
        array([[320., 240.]])
        """
        u, v, valid = ocam_project(np.asarray(P, dtype=np.float64), *self._p())
        return np.stack([u, v], axis=-1), valid

    def unproject(self, uv):
        """Unproject pixels to unit bearing rays (direct polynomial evaluation).

        Inverts the affine stretch linearly, then evaluates the world
        polynomial ``w(rho)`` directly (no root-finding needed in this
        direction) to form ``ray = normalize([x, y, -w(rho)])``. See
        ``ds_msp.models.ocam_math.ocam_unproject``.

        Parameters
        ----------
        uv : ndarray, shape (..., 2)
            Pixel coordinates ``(u, v)``, origin top-left.

        Returns
        -------
        rays : ndarray, shape (..., 3)
            Unit-norm camera-frame bearing vectors, +Z forward.
        valid : ndarray, shape (...,)
            ``True`` iff the resulting ray is finite. Invalid rays are
            zeroed, never NaN.

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import OCamModel
        >>> m = OCamModel.sample()
        >>> rays, valid = m.unproject(np.array([[m.cx, m.cy]]))
        >>> np.round(rays, 4)
        array([[0., 0., 1.]])
        """
        return ocam_unproject(np.asarray(uv, dtype=np.float64), *self._p())

    def project_jacobian(self, P):
        """Project with analytic derivatives via the implicit function theorem.

        ``rho`` is defined implicitly by ``w(rho) + m*rho = 0``; its
        derivatives w.r.t. both the 3D point and the polynomial coefficients
        are obtained without differentiating through the Newton loop, via
        the implicit function theorem (``drho = -dF / (dF/drho)``).

        Parameters
        ----------
        P : ndarray, shape (..., 3)
            Camera-frame points (meters), +Z forward.

        Returns
        -------
        uv : ndarray, shape (..., 2)
            Projected pixel coordinates, identical to `project`.
        J_point : ndarray, shape (..., 2, 3)
            ``d(u, v) / d(X, Y, Z)``.
        J_param : ndarray, shape (..., 2, 10)
            ``d(u, v) / d(cx, cy, c, d, e, a0, a1, a2, a3, a4)``, columns in
            `param_names` order.
        valid : ndarray, shape (...,)
            Projectability mask, identical condition to `project`.

        References
        ----------
        Scaramuzza, D., Martinelli, A., Siegwart, R. IROS 2006 (Jacobian is
        this repo's own derivation via the implicit function theorem;
        verified by finite-difference check, relative error <= 1e-6, see
        ``pytest -m jac``).

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import OCamModel
        >>> m = OCamModel.sample()
        >>> uv, J_point, J_param, valid = m.project_jacobian(np.array([[0.0, 0.0, 1.0]]))
        >>> J_point.shape, J_param.shape
        ((1, 2, 3), (1, 2, 10))
        """
        u, v, J_point, J_param, valid = ocam_project_jacobian(
            np.asarray(P, dtype=np.float64), *self._p())
        return np.stack([u, v], axis=-1), J_point, J_param, valid

    @classmethod
    def from_params(cls, p):
        """Build from a flat ``[cx, cy, c, d, e, a0, a1, a2, a3, a4]`` vector."""
        return cls(*np.asarray(p, dtype=np.float64).ravel())

    @classmethod
    def param_bounds(cls):
        """Optimizer bounds: ``c in (0, 10]``, ``d, e in [-1, 1]``, ``a0 < 0`` (forward-facing focal sign)."""
        lb = np.array([-1e5, -1e5, 0.1, -1.0, -1.0, -1e5, -1e3, -1e3, -1e3, -1e3],
                      dtype=np.float64)
        ub = np.array([1e5, 1e5, 10.0, 1.0, 1.0, -1.0, 1e3, 1e3, 1e3, 1e3],
                      dtype=np.float64)
        return lb, ub

    def initialize_from_correspondences(self, K_seed, rays, pixels) -> None:
        """Seed ``cx,cy`` from `K_seed`; fix the affine to identity and solve
        ``a0..a4`` by linear least squares against the observed world-polynomial values."""
        self.cx, self.cy = float(K_seed[0, 2]), float(K_seed[1, 2])
        self.c, self.d, self.e = 1.0, 0.0, 0.0
        rays = np.asarray(rays, dtype=np.float64)
        x = pixels[:, 0] - self.cx
        y = pixels[:, 1] - self.cy
        rho = np.sqrt(x*x + y*y)
        rxy = np.maximum(np.sqrt(rays[:, 0]**2 + rays[:, 1]**2), 1e-9)
        # ray = [x, y, -w]/n  =>  w(rho) = -rho * rz / |rxy|
        w = -rho * rays[:, 2] / rxy
        rn = rho / R_REF
        A = np.stack([np.ones_like(rn), rn, rn**2, rn**3, rn**4], axis=1)
        coeffs, *_ = np.linalg.lstsq(A, w, rcond=None)
        self.a0, self.a1, self.a2, self.a3, self.a4 = (float(v) for v in coeffs)

    def to_dict(self) -> dict:
        """Serialize to ``{"model": "ocam", "cx": ..., ..., "a4": ...}``."""
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OCamModel":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(**{k: d[k] for k in cls.param_names})

    def __repr__(self) -> str:
        return ("OCamModel(cx={:.2f}, cy={:.2f}, affine=[{:.3f},{:.3f},{:.3f}], "
                "a=[{:.4g},{:.4g},{:.4g},{:.4g},{:.4g}])").format(
                    self.cx, self.cy, self.c, self.d, self.e,
                    self.a0, self.a1, self.a2, self.a3, self.a4)
