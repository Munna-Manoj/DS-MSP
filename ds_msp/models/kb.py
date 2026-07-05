"""Kannala-Brandt (equidistant fisheye) model — OpenCV cv2.fisheye compatible."""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .kb_math import kb_project, kb_project_jacobian, kb_unproject


class KannalaBrandtModel:
    """Kannala-Brandt / equidistant fisheye. Satisfies ``CameraModel``.

    ``K`` and ``distortion`` ([k1,k2,k3,k4]) plug directly into ``cv2.fisheye``.
    """

    name: ClassVar[str] = "kb"
    param_names: ClassVar[Tuple[str, ...]] = (
        "fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")

    def __init__(self, fx, fy, cx, cy, k1=0.0, k2=0.0, k3=0.0, k4=0.0) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.k3 = float(k3)
        self.k4 = float(k4)

    @classmethod
    def sample(cls) -> "KannalaBrandtModel":
        """Realistic instance for contract testing (a narrower-FOV KB lens)."""
        return cls(320.0, 321.0, 320.0, 240.0, 0.05, 0.01, -0.002, 0.0008)

    @property
    def params(self) -> np.ndarray:
        """Flat parameter vector ``[fx, fy, cx, cy, k1, k2, k3, k4]``."""
        return np.array([self.fx, self.fy, self.cx, self.cy,
                         self.k1, self.k2, self.k3, self.k4], dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        """3x3 pinhole intrinsic matrix built from ``fx, fy, cx, cy``."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def distortion(self) -> np.ndarray:
        """Odd-power angle-polynomial coefficients ``[k1, k2, k3, k4]`` (OpenCV order)."""
        return np.array([self.k1, self.k2, self.k3, self.k4], dtype=np.float64)

    def project(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project camera-frame points via the equidistant angle-polynomial map.

        Kannala-Brandt is distinct in operating on the incidence angle rather
        than a sphere: it computes ``theta = atan2(r, z)`` (the angle off the
        optical axis) and maps it through an odd-power polynomial
        ``theta_d = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 +
        k4*theta^9``, then scales the direction ``(x, y)/r`` by
        ``theta_d / r``. With ``k1..k4 = 0`` this reduces to the ideal
        equidistant fisheye ``r = f*theta``.

        Parameters
        ----------
        P : ndarray, shape (..., 3)
            Camera-frame points (meters), +Z forward.

        Returns
        -------
        uv : ndarray, shape (..., 2)
            Projected pixel coordinates ``(u, v)``, origin top-left.
        valid : ndarray, shape (...,)
            ``True`` iff both pixel coordinates are finite. See
            ``ds_msp.models.kb_math.kb_project``.

        References
        ----------
        Kannala, J., Brandt, S. S. "A Generic Camera Model and Calibration
        Method for Conventional, Wide-Angle, and Fish-Eye Lenses." IEEE
        TPAMI 2006. ``K`` and `distortion` plug directly into
        ``cv2.fisheye.projectPoints``.

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import KannalaBrandtModel
        >>> m = KannalaBrandtModel.sample()
        >>> uv, valid = m.project(np.array([[0.0, 0.0, 1.0]]))
        >>> np.round(uv, 2)
        array([[320., 240.]])
        """
        u, v, valid = kb_project(np.asarray(P, dtype=np.float64),
                                 self.fx, self.fy, self.cx, self.cy,
                                 self.k1, self.k2, self.k3, self.k4)
        return np.stack([u, v], axis=-1), valid

    def unproject(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unproject pixels to unit bearing rays via Newton-Raphson.

        Inverts ``theta_d(theta) = ru`` for ``theta`` with 10 fixed Newton
        iterations (no closed form exists for a degree-9 polynomial), then
        builds the ray from ``(sin(theta), cos(theta))`` and the pixel's
        angular direction. See ``ds_msp.models.kb_math.kb_unproject``.

        Parameters
        ----------
        uv : ndarray, shape (..., 2)
            Pixel coordinates ``(u, v)``, origin top-left.

        Returns
        -------
        rays : ndarray, shape (..., 3)
            Unit-norm camera-frame bearing vectors, +Z forward.
        valid : ndarray, shape (...,)
            ``True`` iff the Newton residual ``|theta_d(theta) - ru| < 1e-6``
            and ``theta <= pi``. Invalid rays are zeroed, never NaN.

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import KannalaBrandtModel
        >>> m = KannalaBrandtModel.sample()
        >>> rays, valid = m.unproject(np.array([[m.cx, m.cy]]))
        >>> np.round(rays, 4)
        array([[0., 0., 1.]])
        """
        return kb_unproject(np.asarray(uv, dtype=np.float64),
                            self.fx, self.fy, self.cx, self.cy,
                            self.k1, self.k2, self.k3, self.k4)

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
        J_param : ndarray, shape (..., 2, 8)
            ``d(u, v) / d(fx, fy, cx, cy, k1, k2, k3, k4)``, columns in
            `param_names` order (``d theta_d / d k_i = theta^(2i+1)``).
        valid : ndarray, shape (...,)
            Projectability mask, identical condition to `project`.

        References
        ----------
        Kannala, J., Brandt, S. S. IEEE TPAMI 2006 (closed-form Jacobian
        derived from the forward map; verified here by finite-difference
        check, relative error <= 1e-6, see ``pytest -m jac``).

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import KannalaBrandtModel
        >>> m = KannalaBrandtModel.sample()
        >>> uv, J_point, J_param, valid = m.project_jacobian(np.array([[0.0, 0.0, 1.0]]))
        >>> J_point.shape, J_param.shape
        ((1, 2, 3), (1, 2, 8))
        """
        u, v, J_point, J_param, valid = kb_project_jacobian(
            np.asarray(P, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy,
            self.k1, self.k2, self.k3, self.k4)
        return np.stack([u, v], axis=-1), J_point, J_param, valid

    @classmethod
    def from_params(cls, p: np.ndarray) -> "KannalaBrandtModel":
        """Build from a flat ``[fx, fy, cx, cy, k1, k2, k3, k4]`` vector."""
        return cls(*np.asarray(p, dtype=np.float64).ravel())

    @classmethod
    def param_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        """Optimizer bounds: each ``k_i in [-1, 1]``, focal/center wide-open."""
        lb = np.array([1.0, 1.0, -1e5, -1e5, -1.0, -1.0, -1.0, -1.0], dtype=np.float64)
        ub = np.array([1e5, 1e5, 1e5, 1e5, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        return lb, ub

    def initialize_from_correspondences(self, K_seed, rays, pixels) -> None:
        """Seed ``fx,fy,cx,cy`` from `K_seed`; solve ``k1..k4`` by linear least squares
        (``ru - theta`` is linear in the odd powers of ``theta``)."""
        self.fx, self.fy = float(K_seed[0, 0]), float(K_seed[1, 1])
        self.cx, self.cy = float(K_seed[0, 2]), float(K_seed[1, 2])
        rays = np.asarray(rays, dtype=np.float64)
        theta = np.arctan2(np.sqrt(rays[:, 0]**2 + rays[:, 1]**2), rays[:, 2])
        mx = (pixels[:, 0] - self.cx) / self.fx
        my = (pixels[:, 1] - self.cy) / self.fy
        ru = np.sqrt(mx*mx + my*my)
        # ru = theta + k1 th^3 + k2 th^5 + k3 th^7 + k4 th^9 -> linear in k.
        A = np.stack([theta**3, theta**5, theta**7, theta**9], axis=1)
        b = ru - theta
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.k1, self.k2, self.k3, self.k4 = (float(c) for c in coeffs)

    def to_dict(self) -> dict:
        """Serialize to ``{"model": "kb", "fx": ..., ..., "k4": ...}``."""
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "KannalaBrandtModel":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(**{k: d[k] for k in cls.param_names})

    def __repr__(self) -> str:
        return ("KannalaBrandtModel(fx={:.3f}, fy={:.3f}, cx={:.3f}, cy={:.3f}, "
                "k=[{:.5f}, {:.5f}, {:.5f}, {:.5f}])").format(
                    self.fx, self.fy, self.cx, self.cy,
                    self.k1, self.k2, self.k3, self.k4)
