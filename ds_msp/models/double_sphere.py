"""
Double Sphere model implementing the CameraModel contract.

Thin value-object wrapper over the pure math in ``ds_math``. Depends only on
``ds_math`` (numpy) and ``core.contracts`` — no OpenCV, no services.
"""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .ds_math import ds_project, ds_project_jacobian, ds_unproject


class DoubleSphereModel:
    """Double Sphere camera (Usenko et al. 2018). Satisfies ``CameraModel``."""

    name: ClassVar[str] = "ds"
    param_names: ClassVar[Tuple[str, ...]] = ("fx", "fy", "cx", "cy", "xi", "alpha")

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 xi: float, alpha: float) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.xi = float(xi)
        self.alpha = float(alpha)

    @classmethod
    def sample(cls) -> "DoubleSphereModel":
        """Realistic instance for contract testing (the bundled calibration)."""
        return cls(711.57, 711.24, 949.18, 518.81, 0.183, 0.809)

    # -- parameter access -------------------------------------------------
    @property
    def params(self) -> np.ndarray:
        """Flat parameter vector ``[fx, fy, cx, cy, xi, alpha]``, see `param_names`."""
        return np.array([self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha],
                        dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        """3x3 pinhole intrinsic matrix built from ``fx, fy, cx, cy``."""
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def distortion(self) -> np.ndarray:
        """Distortion tail ``[xi, alpha]`` (sphere offset, perspective blend)."""
        return np.array([self.xi, self.alpha], dtype=np.float64)

    # -- core math --------------------------------------------------------
    def project(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project camera-frame points to pixels via the two-sphere composition.

        The Double Sphere model is what makes this model's family distinct: a
        point is first re-centered onto a unit sphere shifted by ``xi`` along
        -Z (``z1 = z + xi*d1``), then perspective-divided from a second point
        blended between that shifted sphere and the pinhole plane by
        ``alpha``. Composing two spheres (rather than one, as in `UCMModel`)
        is what lets this closed-form model reach fields of view beyond 180°
        (Usenko et al. 2018, Sec. 3).

        Parameters
        ----------
        P : ndarray, shape (..., 3)
            Camera-frame points (meters), +Z forward.

        Returns
        -------
        uv : ndarray, shape (..., 2)
            Projected pixel coordinates ``(u, v)``, origin top-left.
        valid : ndarray, shape (...,)
            Projectability mask. ``True`` iff the point lies in the tilted
            half-space ``z > -w2 * d1`` (*not* the naive ``z > 0``) and the
            projection denominator is bounded away from zero — see
            ``ds_msp.models.ds_math.ds_project`` and
            [Projection validity and FOV](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/explain/projection_validity_and_fov.md).
            Invalid entries are masked, never NaN.

        References
        ----------
        Usenko, V., Demmel, N., Cremers, D. "The Double Sphere Camera Model."
        3DV 2018 (projection; see Eq. 43-45 for the validity half-space).

        Examples
        --------
        A point *behind* the pinhole plane (``z < 0``) can still be valid —
        this is the wide-FOV behavior the two-sphere composition exists for:

        >>> import numpy as np
        >>> from ds_msp.models import DoubleSphereModel
        >>> m = DoubleSphereModel.sample()
        >>> uv, valid = m.project(np.array([[1.0, 0.0, -0.3]]))
        >>> bool(valid[0])
        True
        """
        u, v, valid = ds_project(np.asarray(P, dtype=np.float64),
                                 self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha)
        return np.stack([u, v], axis=-1), valid

    def unproject(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unproject pixels to unit bearing rays (closed form).

        Inverts the two-sphere composition analytically (no iteration): first
        recovers the point on the shifted sphere from the normalized pixel,
        then un-shifts by ``xi``. See ``ds_msp.models.ds_math.ds_unproject``
        for the exact algebra.

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
            (radius set by ``alpha``) and every intermediate square root is
            real and the perspective denominator is bounded away from zero.
            Invalid rays are zeroed, never NaN.

        References
        ----------
        Usenko, V., Demmel, N., Cremers, D. "The Double Sphere Camera Model." 3DV 2018
        (closed-form unprojection).

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import DoubleSphereModel
        >>> m = DoubleSphereModel.sample()
        >>> rays, valid = m.unproject(np.array([[m.cx, m.cy]]))
        >>> np.round(rays, 4)
        array([[0., 0., 1.]])
        """
        return ds_unproject(np.asarray(uv, dtype=np.float64),
                            self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha)

    def project_jacobian(
        self, P: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            ``d(u, v) / d(fx, fy, cx, cy, xi, alpha)``, columns in
            `param_names` order.
        valid : ndarray, shape (...,)
            Projectability mask, identical condition to `project`.

        References
        ----------
        Usenko, V., Demmel, N., Cremers, D. "The Double Sphere Camera Model." 3DV 2018
        (closed-form Jacobian derived from the projection above; verified here by
        finite-difference check, relative error <= 1e-6, see ``pytest -m jac``).

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import DoubleSphereModel
        >>> m = DoubleSphereModel.sample()
        >>> uv, J_point, J_param, valid = m.project_jacobian(np.array([[0.0, 0.0, 1.0]]))
        >>> J_point.shape, J_param.shape
        ((1, 2, 3), (1, 2, 6))
        """
        u, v, J_point, J_param, valid = ds_project_jacobian(
            np.asarray(P, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha)
        return np.stack([u, v], axis=-1), J_point, J_param, valid

    # -- construction / bounds -------------------------------------------
    @classmethod
    def from_params(cls, p: np.ndarray) -> "DoubleSphereModel":
        """Build from a flat ``[fx, fy, cx, cy, xi, alpha]`` vector."""
        return cls(*np.asarray(p, dtype=np.float64).ravel())

    @classmethod
    def param_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        """Optimizer bounds: ``xi in [-1, 1]``, ``alpha in (0, 1)``, focal/center wide-open."""
        lb = np.array([1.0, 1.0, -1e5, -1e5, -1.0, 1e-6], dtype=np.float64)
        ub = np.array([1e5, 1e5, 1e5, 1e5, 1.0, 1.0 - 1e-6], dtype=np.float64)
        return lb, ub

    # -- conversion hook --------------------------------------------------
    def initialize_from_correspondences(
        self, K_seed: np.ndarray, rays: np.ndarray, pixels: np.ndarray
    ) -> None:
        """Seed ``fx,fy,cx,cy`` from `K_seed`; seed ``xi=0`` and solve ``alpha`` linearly
        (reduces to the UCM closed form since ``xi=0`` collapses both spheres into one)."""
        # Inherit pinhole intrinsics from the source.
        self.fx, self.fy = float(K_seed[0, 0]), float(K_seed[1, 1])
        self.cx, self.cy = float(K_seed[0, 2]), float(K_seed[1, 2])
        # Seed distortion: xi = 0 reduces DS to UCM; for unit rays (d1 = 1),
        #   mx = x / (alpha*(1 - z) + z)  =>  alpha = (x - mx*z) / (mx*(1 - z)).
        # Solve linearly over both axes.
        rays = np.asarray(rays, dtype=np.float64)
        x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]
        mx = (pixels[:, 0] - self.cx) / self.fx
        my = (pixels[:, 1] - self.cy) / self.fy
        A = np.concatenate([mx * (1.0 - z), my * (1.0 - z)])
        b = np.concatenate([x - mx * z, y - my * z])
        denom = float(A @ A)
        self.xi = 0.0
        self.alpha = float(np.clip((A @ b) / denom, 1e-6, 1.0 - 1e-6)) if denom > 1e-12 else 0.5

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        """Serialize to ``{"model": "ds", "fx": ..., ..., "alpha": ...}``."""
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DoubleSphereModel":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(**{k: d[k] for k in cls.param_names})

    def __repr__(self) -> str:
        return ("DoubleSphereModel(fx={:.3f}, fy={:.3f}, cx={:.3f}, cy={:.3f}, "
                "xi={:.4f}, alpha={:.4f})").format(
                    self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha)
