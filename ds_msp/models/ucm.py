"""Unified Camera Model implementing the CameraModel contract."""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .ucm_math import ucm_project, ucm_project_jacobian, ucm_unproject


class UCMModel:
    """Unified Camera Model (single sphere). Satisfies ``CameraModel``."""

    name: ClassVar[str] = "ucm"
    param_names: ClassVar[Tuple[str, ...]] = ("fx", "fy", "cx", "cy", "alpha")

    def __init__(self, fx: float, fy: float, cx: float, cy: float, alpha: float) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.alpha = float(alpha)

    @classmethod
    def sample(cls) -> "UCMModel":
        """Realistic instance for contract testing (the bundled calibration)."""
        return cls(711.57, 711.24, 949.18, 518.81, 0.62)

    @property
    def params(self) -> np.ndarray:
        """Flat parameter vector ``[fx, fy, cx, cy, alpha]``."""
        return np.array([self.fx, self.fy, self.cx, self.cy, self.alpha], dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        """3x3 pinhole intrinsic matrix built from ``fx, fy, cx, cy``."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def distortion(self) -> np.ndarray:
        """Distortion tail ``[alpha]`` (perspective blend between sphere and plane)."""
        return np.array([self.alpha], dtype=np.float64)

    def project(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project camera-frame points via the single unit-sphere composition.

        UCM re-centers the point onto a single unit sphere (radius
        ``d = sqrt(x^2+y^2+z^2)``), then perspective-divides from a point
        blended between the sphere's surface and the pinhole plane by
        ``alpha``: ``den = alpha*d + (1-alpha)*z``. It is the one-sphere,
        one-parameter special case of `DoubleSphereModel` (``xi=0``).

        Parameters
        ----------
        P : ndarray, shape (..., 3)
            Camera-frame points (meters), +Z forward.

        Returns
        -------
        uv : ndarray, shape (..., 2)
            Projected pixel coordinates ``(u, v)``, origin top-left.
        valid : ndarray, shape (...,)
            ``True`` iff the point lies in the tilted half-space
            ``z > -w(alpha) * d`` (*not* the naive ``z > 0``) and the
            perspective-division denominator is bounded away from zero. See
            ``ds_msp.models.ucm_math.ucm_project``.

        References
        ----------
        Geyer, C., Daniilidis, K. "A Unifying Theory for Central Panoramic
        Systems." ECCV 2000; Mei, C., Rives, P. "Single View Point
        Omnidirectional Camera Calibration from Planar Grids." ICRA 2007.

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import UCMModel
        >>> m = UCMModel.sample()
        >>> uv, valid = m.project(np.array([[0.0, 0.0, 1.0]]))
        >>> np.round(uv, 2)
        array([[949.18, 518.81]])
        """
        u, v, valid = ucm_project(np.asarray(P, dtype=np.float64),
                                  self.fx, self.fy, self.cx, self.cy, self.alpha)
        return np.stack([u, v], axis=-1), valid

    def unproject(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unproject pixels to unit bearing rays (closed form).

        Parameters
        ----------
        uv : ndarray, shape (..., 2)
            Pixel coordinates ``(u, v)``, origin top-left.

        Returns
        -------
        rays : ndarray, shape (..., 3)
            Unit-norm camera-frame bearing vectors, +Z forward.
        valid : ndarray, shape (...,)
            ``True`` iff the pixel lies within the sphere's reachable disc
            and the perspective-division denominator is bounded away from
            zero. Invalid rays are zeroed, never NaN. See
            ``ds_msp.models.ucm_math.ucm_unproject``.

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import UCMModel
        >>> m = UCMModel.sample()
        >>> rays, valid = m.unproject(np.array([[m.cx, m.cy]]))
        >>> np.round(rays, 4)
        array([[0., 0., 1.]])
        """
        return ucm_unproject(np.asarray(uv, dtype=np.float64),
                             self.fx, self.fy, self.cx, self.cy, self.alpha)

    def project_jacobian(self, P):
        """Project with analytic derivatives (no autodiff, no finite differences).

        Parameters
        ----------
        P : ndarray, shape (..., 3)
            Camera-frame points (meters), +Z forward.

        Returns
        -------
        uv : ndarray, shape (..., 2)
            Projected pixel coordinates, identical to `project`.
        J_point : ndarray, shape (..., 2, 3)
            ``d(u, v) / d(x, y, z)``.
        J_param : ndarray, shape (..., 2, 5)
            ``d(u, v) / d(fx, fy, cx, cy, alpha)``, columns in `param_names`
            order.
        valid : ndarray, shape (...,)
            Projectability mask, identical condition to `project`.

        References
        ----------
        Geyer, C., Daniilidis, K. ECCV 2000; Mei, C., Rives, P. ICRA 2007
        (closed-form Jacobian derived from the forward map; verified here by
        finite-difference check, relative error <= 1e-6, see ``pytest -m jac``).

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import UCMModel
        >>> m = UCMModel.sample()
        >>> uv, J_point, J_param, valid = m.project_jacobian(np.array([[0.0, 0.0, 1.0]]))
        >>> J_point.shape, J_param.shape
        ((1, 2, 3), (1, 2, 5))
        """
        u, v, J_point, J_param, valid = ucm_project_jacobian(
            np.asarray(P, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, self.alpha)
        return np.stack([u, v], axis=-1), J_point, J_param, valid

    @classmethod
    def from_params(cls, p: np.ndarray) -> "UCMModel":
        """Build from a flat ``[fx, fy, cx, cy, alpha]`` vector."""
        return cls(*np.asarray(p, dtype=np.float64).ravel())

    @classmethod
    def param_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        """Optimizer bounds: ``alpha in (0, 1)``, focal/center wide-open."""
        lb = np.array([1.0, 1.0, -1e5, -1e5, 1e-6], dtype=np.float64)
        ub = np.array([1e5, 1e5, 1e5, 1e5, 1.0 - 1e-6], dtype=np.float64)
        return lb, ub

    def initialize_from_correspondences(self, K_seed, rays, pixels) -> None:
        """Seed ``fx,fy,cx,cy`` from `K_seed`; solve ``alpha`` by linear least squares
        from unit-ray/pixel correspondences."""
        self.fx, self.fy = float(K_seed[0, 0]), float(K_seed[1, 1])
        self.cx, self.cy = float(K_seed[0, 2]), float(K_seed[1, 2])
        rays = np.asarray(rays, dtype=np.float64)
        x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]
        mx = (pixels[:, 0] - self.cx) / self.fx
        my = (pixels[:, 1] - self.cy) / self.fy
        # unit rays (d = 1): alpha = (x - mx*z) / (mx*(1 - z)), solved linearly.
        A = np.concatenate([mx * (1.0 - z), my * (1.0 - z)])
        b = np.concatenate([x - mx * z, y - my * z])
        denom = float(A @ A)
        self.alpha = float(np.clip((A @ b) / denom, 1e-6, 1.0 - 1e-6)) if denom > 1e-12 else 0.5

    def to_dict(self) -> dict:
        """Serialize to ``{"model": "ucm", "fx": ..., ..., "alpha": ...}``."""
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "UCMModel":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(**{k: d[k] for k in cls.param_names})

    def __repr__(self) -> str:
        return "UCMModel(fx={:.3f}, fy={:.3f}, cx={:.3f}, cy={:.3f}, alpha={:.4f})".format(
            self.fx, self.fy, self.cx, self.cy, self.alpha)
