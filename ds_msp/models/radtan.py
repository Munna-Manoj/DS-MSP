"""Pinhole + radial-tangential (Brown) model — OpenCV cv2.projectPoints compatible."""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .radtan_math import radtan_project, radtan_project_jacobian, radtan_unproject


class RadTanModel:
    """Pinhole with radial-tangential distortion. Satisfies ``CameraModel``.

    ``distortion`` returns ``[k1, k2, p1, p2, k3]`` (OpenCV order) for direct use
    with ``cv2.projectPoints`` / ``cv2.undistortPoints``. Narrow-FOV: only models
    rays with ``z > 0``.
    """

    name: ClassVar[str] = "radtan"
    param_names: ClassVar[Tuple[str, ...]] = (
        "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")

    def __init__(self, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0) -> None:
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.k1, self.k2, self.k3 = float(k1), float(k2), float(k3)
        self.p1, self.p2 = float(p1), float(p2)

    @classmethod
    def sample(cls) -> "RadTanModel":
        """Realistic instance for contract testing (a narrow-FOV pinhole lens)."""
        return cls(600.0, 602.0, 320.0, 240.0, -0.12, 0.05, 0.001, -0.0015, 0.008)

    @property
    def params(self) -> np.ndarray:
        """Flat parameter vector ``[fx, fy, cx, cy, k1, k2, p1, p2, k3]``."""
        return np.array([self.fx, self.fy, self.cx, self.cy,
                         self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        """3x3 pinhole intrinsic matrix built from ``fx, fy, cx, cy``."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def distortion(self) -> np.ndarray:
        """OpenCV distCoeffs order [k1, k2, p1, p2, k3]."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    def _coeffs(self):
        return (self.k1, self.k2, self.p1, self.p2, self.k3)

    def project(self, P):
        """Project camera-frame points via the classic Brown-Conrady model.

        The pinhole projects onto the normalized plane (``a=x/z, b=y/z``)
        and applies polynomial radial distortion (``k1, k2, k3`` on
        ``r2=a^2+b^2``) plus tangential distortion (``p1, p2``, modeling lens
        decentering). Unlike the other 7 models in this package, RadTan does
        **not** attempt to represent fisheye FOV: it is only valid for
        ``z > 0`` and cannot represent rays at or beyond 90° off-axis.

        Parameters
        ----------
        P : ndarray, shape (..., 3)
            Camera-frame points (meters), +Z forward.

        Returns
        -------
        uv : ndarray, shape (..., 2)
            Projected pixel coordinates ``(u, v)``, origin top-left.
        valid : ndarray, shape (...,)
            ``True`` iff ``z > 1e-9``. Points with ``z <= 0`` are masked and
            their pixels zeroed. See ``ds_msp.models.radtan_math.radtan_project``.

        References
        ----------
        Brown, D. C. "Decentering Distortion of Lenses." Photogrammetric
        Engineering, 1966 (tangential terms); OpenCV
        ``distCoeffs = [k1, k2, p1, p2, k3]`` convention, see
        ``cv2.projectPoints``.

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import RadTanModel
        >>> m = RadTanModel.sample()
        >>> uv, valid = m.project(np.array([[0.0, 0.0, 1.0]]))
        >>> np.round(uv, 2)
        array([[320., 240.]])
        """
        u, v, valid = radtan_project(np.asarray(P, dtype=np.float64),
                                     self.fx, self.fy, self.cx, self.cy, *self._coeffs())
        return np.stack([u, v], axis=-1), valid

    def unproject(self, uv):
        """Unproject pixels to unit bearing rays via fixed-point distortion inversion.

        Runs 20 fixed-point iterations of the standard OpenCV-style inverse
        (no closed form exists for the Brown-Conrady distortion), then
        re-distorts the recovered normalized coordinates to check
        consistency with the input.

        Parameters
        ----------
        uv : ndarray, shape (..., 2)
            Pixel coordinates ``(u, v)``, origin top-left.

        Returns
        -------
        rays : ndarray, shape (..., 3)
            Unit-norm camera-frame bearing vectors, +Z forward.
        valid : ndarray, shape (...,)
            ``True`` iff re-distorting the recovered point reproduces the
            input to within ``1e-6``. Invalid rays are zeroed, never NaN.

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import RadTanModel
        >>> m = RadTanModel.sample()
        >>> rays, valid = m.unproject(np.array([[m.cx, m.cy]]))
        >>> np.round(rays, 4)
        array([[0., 0., 1.]])
        """
        return radtan_unproject(np.asarray(uv, dtype=np.float64),
                                self.fx, self.fy, self.cx, self.cy, *self._coeffs())

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
        J_param : ndarray, shape (..., 2, 9)
            ``d(u, v) / d(fx, fy, cx, cy, k1, k2, p1, p2, k3)``, columns in
            `param_names` order.
        valid : ndarray, shape (...,)
            Projectability mask, identical condition to `project`.

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import RadTanModel
        >>> m = RadTanModel.sample()
        >>> uv, J_point, J_param, valid = m.project_jacobian(np.array([[0.0, 0.0, 1.0]]))
        >>> J_point.shape, J_param.shape
        ((1, 2, 3), (1, 2, 9))
        """
        u, v, J_point, J_param, valid = radtan_project_jacobian(
            np.asarray(P, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, *self._coeffs())
        return np.stack([u, v], axis=-1), J_point, J_param, valid

    @classmethod
    def from_params(cls, p):
        """Build from a flat ``[fx, fy, cx, cy, k1, k2, p1, p2, k3]`` vector."""
        return cls(*np.asarray(p, dtype=np.float64).ravel())

    @classmethod
    def param_bounds(cls):
        """Optimizer bounds: radial/tangential terms modestly bounded, focal/center wide-open."""
        lb = np.array([1.0, 1.0, -1e5, -1e5, -2.0, -2.0, -1.0, -1.0, -2.0], dtype=np.float64)
        ub = np.array([1e5, 1e5, 1e5, 1e5, 2.0, 2.0, 1.0, 1.0, 2.0], dtype=np.float64)
        return lb, ub

    def initialize_from_correspondences(self, K_seed, rays, pixels) -> None:
        """Seed ``fx,fy,cx,cy`` from `K_seed`; zero all distortion coefficients."""
        self.fx, self.fy = float(K_seed[0, 0]), float(K_seed[1, 1])
        self.cx, self.cy = float(K_seed[0, 2]), float(K_seed[1, 2])
        self.k1 = self.k2 = self.k3 = self.p1 = self.p2 = 0.0

    def to_dict(self) -> dict:
        """Serialize to ``{"model": "radtan", "fx": ..., ..., "k3": ...}``."""
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RadTanModel":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(**{k: d[k] for k in cls.param_names})

    def __repr__(self) -> str:
        return ("RadTanModel(fx={:.3f}, fy={:.3f}, cx={:.3f}, cy={:.3f}, "
                "k1={:.5f}, k2={:.5f}, p1={:.5f}, p2={:.5f}, k3={:.5f})").format(
                    self.fx, self.fy, self.cx, self.cy,
                    self.k1, self.k2, self.p1, self.p2, self.k3)
