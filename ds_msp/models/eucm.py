"""Enhanced Unified Camera Model implementing the CameraModel contract."""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .eucm_math import eucm_project, eucm_project_jacobian, eucm_unproject


class EUCMModel:
    """Enhanced UCM (Khomutenko et al. 2016). Satisfies ``CameraModel``."""

    name: ClassVar[str] = "eucm"
    param_names: ClassVar[Tuple[str, ...]] = ("fx", "fy", "cx", "cy", "alpha", "beta")

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 alpha: float, beta: float) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.alpha = float(alpha)
        self.beta = float(beta)

    @classmethod
    def sample(cls) -> "EUCMModel":
        """Realistic instance for contract testing (the bundled calibration)."""
        return cls(711.57, 711.24, 949.18, 518.81, 0.6, 1.1)

    @property
    def params(self) -> np.ndarray:
        """Flat parameter vector ``[fx, fy, cx, cy, alpha, beta]``."""
        return np.array([self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta],
                        dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        """3x3 pinhole intrinsic matrix built from ``fx, fy, cx, cy``."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def distortion(self) -> np.ndarray:
        """Distortion tail ``[alpha, beta]`` (perspective blend, ellipse-radius weight)."""
        return np.array([self.alpha, self.beta], dtype=np.float64)

    def project(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project camera-frame points via an ellipse-radius-weighted UCM sphere.

        EUCM generalizes the Unified Camera Model (`UCMModel`) with a second
        parameter ``beta`` that reweights the radial term inside the sphere
        distance, ``d = sqrt(beta*(x^2+y^2) + z^2)``, before the same
        ``alpha``-blended perspective division. This extra degree of freedom
        (decoupling the radial and axial curvature of the projection surface)
        is what lets EUCM fit distortion profiles a single-sphere UCM cannot.

        Parameters
        ----------
        P : ndarray, shape (..., 3)
            Camera-frame points (meters), +Z forward.

        Returns
        -------
        uv : ndarray, shape (..., 2)
            Projected pixel coordinates ``(u, v)``, origin top-left.
        valid : ndarray, shape (...,)
            ``True`` iff the perspective-division denominator
            ``alpha*d + (1-alpha)*z`` is bounded away from zero. See
            ``ds_msp.models.eucm_math.eucm_project``.

        References
        ----------
        Khomutenko, B., Garcia, G., Martinet, P. "An Enhanced Unified Camera
        Model for Omnidirectional Cameras." IEEE RA-L 2016.

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import EUCMModel
        >>> m = EUCMModel.sample()
        >>> uv, valid = m.project(np.array([[0.0, 0.0, 1.0]]))
        >>> np.round(uv, 2)
        array([[949.18, 518.81]])
        """
        u, v, valid = eucm_project(np.asarray(P, dtype=np.float64),
                                   self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta)
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
            ``True`` iff the pixel lies within the reachable disc and the
            perspective-division denominator is bounded away from zero.
            Invalid rays are zeroed, never NaN. See
            ``ds_msp.models.eucm_math.eucm_unproject``.

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import EUCMModel
        >>> m = EUCMModel.sample()
        >>> rays, valid = m.unproject(np.array([[m.cx, m.cy]]))
        >>> np.round(rays, 4)
        array([[0., 0., 1.]])
        """
        return eucm_unproject(np.asarray(uv, dtype=np.float64),
                              self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta)

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
        J_param : ndarray, shape (..., 2, 6)
            ``d(u, v) / d(fx, fy, cx, cy, alpha, beta)``, columns in
            `param_names` order.
        valid : ndarray, shape (...,)
            Projectability mask, identical condition to `project`.

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import EUCMModel
        >>> m = EUCMModel.sample()
        >>> uv, J_point, J_param, valid = m.project_jacobian(np.array([[0.0, 0.0, 1.0]]))
        >>> J_point.shape, J_param.shape
        ((1, 2, 3), (1, 2, 6))
        """
        u, v, J_point, J_param, valid = eucm_project_jacobian(
            np.asarray(P, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta)
        return np.stack([u, v], axis=-1), J_point, J_param, valid

    @classmethod
    def from_params(cls, p: np.ndarray) -> "EUCMModel":
        """Build from a flat ``[fx, fy, cx, cy, alpha, beta]`` vector."""
        return cls(*np.asarray(p, dtype=np.float64).ravel())

    @classmethod
    def param_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        """Optimizer bounds: ``alpha in (0, 1)``, ``beta in (0, 10]``."""
        lb = np.array([1.0, 1.0, -1e5, -1e5, 1e-6, 1e-3], dtype=np.float64)
        ub = np.array([1e5, 1e5, 1e5, 1e5, 1.0 - 1e-6, 10.0], dtype=np.float64)
        return lb, ub

    def initialize_from_correspondences(self, K_seed, rays, pixels) -> None:
        """Seed ``fx,fy,cx,cy`` from `K_seed`; fix ``beta=1`` and solve ``alpha`` linearly
        (the UCM closed form; ``beta=1`` reduces EUCM to UCM)."""
        self.fx, self.fy = float(K_seed[0, 0]), float(K_seed[1, 1])
        self.cx, self.cy = float(K_seed[0, 2]), float(K_seed[1, 2])
        self.beta = 1.0
        rays = np.asarray(rays, dtype=np.float64)
        x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]
        mx = (pixels[:, 0] - self.cx) / self.fx
        my = (pixels[:, 1] - self.cy) / self.fy
        # beta=1, unit rays (d=1): alpha = (x - mx*z)/(mx*(1 - z)), linear LS.
        A = np.concatenate([mx * (1.0 - z), my * (1.0 - z)])
        b = np.concatenate([x - mx * z, y - my * z])
        denom = float(A @ A)
        self.alpha = float(np.clip((A @ b) / denom, 1e-6, 1.0 - 1e-6)) if denom > 1e-12 else 0.5

    def to_dict(self) -> dict:
        """Serialize to ``{"model": "eucm", "fx": ..., ..., "beta": ...}``."""
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EUCMModel":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(**{k: d[k] for k in cls.param_names})

    def __repr__(self) -> str:
        return ("EUCMModel(fx={:.3f}, fy={:.3f}, cx={:.3f}, cy={:.3f}, "
                "alpha={:.4f}, beta={:.4f})").format(
                    self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta)
