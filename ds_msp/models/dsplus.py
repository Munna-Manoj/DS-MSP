# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025-2026 Munna-Manoj. DS+ (Double Sphere Plus) camera model, from
# DS-MSP (https://github.com/Munna-Manoj/DS-MSP). NONCOMMERCIAL use only, with
# attribution — see LICENSE-NONCOMMERCIAL.txt and LICENSING.md. The rest of DS-MSP is MIT.
"""DS+ camera model (UCM core + division radial + 2-axis tilt) implementing the
CameraModel contract.

DS+ is the Double Sphere model with ``xi`` dropped (the proven near-null direction
for the target lens) and two extra closed-form-invertible stages added: a Fitzgibbon
division-model radial layer (``lambda1, lambda2``) and a 2-axis Scheimpflug tilt
homography (``tau_x, tau_y``). See ``dsplus_math`` for the full derivation.
"""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .dsplus_math import (
    dsplus_project,
    dsplus_project_jacobian,
    dsplus_unproject,
)


class DSPlusModel:
    """DS+ (UCM core + division radial + 2-axis tilt). Satisfies ``CameraModel``."""

    name: ClassVar[str] = "dsplus"
    param_names: ClassVar[Tuple[str, ...]] = (
        "fx", "fy", "cx", "cy", "alpha", "lambda1", "lambda2", "tau_x", "tau_y")

    def __init__(self, fx, fy, cx, cy, alpha=0.5,
                 lambda1=0.0, lambda2=0.0, tau_x=0.0, tau_y=0.0) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.alpha = float(alpha)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.tau_x = float(tau_x)
        self.tau_y = float(tau_y)

    @classmethod
    def sample(cls) -> "DSPlusModel":
        """Realistic instance for contract testing (the bundled calibration)."""
        return cls(711.57, 711.24, 949.18, 518.81, 0.62, -0.10, 0.02, 0.001, -0.001)

    @property
    def params(self) -> np.ndarray:
        """Flat parameter vector in `param_names` order."""
        return np.array([self.fx, self.fy, self.cx, self.cy, self.alpha,
                         self.lambda1, self.lambda2, self.tau_x, self.tau_y],
                        dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        """3x3 pinhole intrinsic matrix built from ``fx, fy, cx, cy``."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def distortion(self) -> np.ndarray:
        """Distortion tail ``[alpha, lambda1, lambda2, tau_x, tau_y]``."""
        return np.array([self.alpha, self.lambda1, self.lambda2,
                         self.tau_x, self.tau_y], dtype=np.float64)

    def project(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project camera-frame points through 4 closed-form-invertible stages.

        DS+ replaces Double Sphere's ``xi`` shift (the proven near-null
        direction for the target lens, see the module docstring) with a UCM
        sphere core followed by two extra closed-form stages: a Fitzgibbon
        division-model radial layer and a 2-axis Scheimpflug tilt homography,
        so the composed map stays fully closed-form invertible end to end::

            pixel = K . H_tau . D_lambda . S_alpha(bearing)

        Parameters
        ----------
        P : ndarray, shape (..., 3)
            Camera-frame points (meters), +Z forward.

        Returns
        -------
        uv : ndarray, shape (..., 2)
            Projected pixel coordinates ``(u, v)``, origin top-left.
        valid : ndarray, shape (...,)
            ``True`` iff all 4 stages are individually valid: the UCM
            half-space/denominator condition, a positive division-model
            radial factor, and a nonzero tilt-homography denominator. See
            ``ds_msp.models.dsplus_math.dsplus_project``.

        References
        ----------
        UCM core: Geyer, C., Daniilidis, K. "A Unifying Theory for Central
        Panoramic Systems." ECCV 2000; Mei, C., Rives, P. ICRA 2007.
        Division radial layer: Fitzgibbon, A. "Simultaneous linear estimation
        of multiple view geometry and lens distortion." CVPR 2001.
        Tilt homography: cf. OpenCV ``CALIB_TILTED_MODEL``. Staged
        composition and Jacobian are this repo's own extension — see
        ``ds_msp/models/dsplus_math.py`` and
        [ADR-0005](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/process/architecture/decisions/ADR-0005-dsplus-eucmplus.md).

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import DSPlusModel
        >>> m = DSPlusModel.sample()
        >>> uv, valid = m.project(np.array([[0.0, 0.0, 1.0]]))
        >>> np.round(uv, 2)
        array([[949.18, 518.81]])
        """
        u, v, valid = dsplus_project(
            np.asarray(P, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, self.alpha,
            self.lambda1, self.lambda2, self.tau_x, self.tau_y)
        return np.stack([u, v], axis=-1), valid

    def unproject(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unproject pixels to unit bearing rays (closed form, no iteration).

        Inverts the 4-stage chain in reverse (``K^-1``, tilt inverse, then a
        quadratic/quartic radical to invert the division-model radial layer,
        then the UCM closed form). See
        ``ds_msp.models.dsplus_math.dsplus_unproject`` and
        ``ds_msp.models.dsplus_math._invert_division``.

        Parameters
        ----------
        uv : ndarray, shape (..., 2)
            Pixel coordinates ``(u, v)``, origin top-left.

        Returns
        -------
        rays : ndarray, shape (..., 3)
            Unit-norm camera-frame bearing vectors, +Z forward.
        valid : ndarray, shape (...,)
            ``True`` iff the tilt inverse, the division-radial root, and the
            UCM sphere inverse are all well-defined. Invalid rays are
            zeroed, never NaN.

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import DSPlusModel
        >>> m = DSPlusModel.sample()
        >>> rays, valid = m.unproject(np.array([[m.cx, m.cy]]))
        >>> np.round(rays, 4)
        array([[0., 0., 1.]])
        """
        return dsplus_unproject(
            np.asarray(uv, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, self.alpha,
            self.lambda1, self.lambda2, self.tau_x, self.tau_y)

    def project_jacobian(self, P):
        """Project with analytic derivatives via the chain rule through all 4 stages.

        Parameters
        ----------
        P : ndarray, shape (..., 3)
            Camera-frame points (meters), +Z forward.

        Returns
        -------
        uv : ndarray, shape (..., 2)
            Projected pixel coordinates, identical to `project`.
        J_point : ndarray, shape (..., 2, 3)
            ``d(u, v) / d(x, y, z)``, computed as ``K @ J_H @ J_D @ J_S``.
        J_param : ndarray, shape (..., 2, 9)
            ``d(u, v) / d(fx, fy, cx, cy, alpha, lambda1, lambda2, tau_x, tau_y)``,
            columns in `param_names` order.
        valid : ndarray, shape (...,)
            Projectability mask, identical condition to `project`.

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import DSPlusModel
        >>> m = DSPlusModel.sample()
        >>> uv, J_point, J_param, valid = m.project_jacobian(np.array([[0.0, 0.0, 1.0]]))
        >>> J_point.shape, J_param.shape
        ((1, 2, 3), (1, 2, 9))
        """
        u, v, J_point, J_param, valid = dsplus_project_jacobian(
            np.asarray(P, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, self.alpha,
            self.lambda1, self.lambda2, self.tau_x, self.tau_y)
        return np.stack([u, v], axis=-1), J_point, J_param, valid

    @classmethod
    def from_params(cls, p: np.ndarray) -> "DSPlusModel":
        """Build from a flat vector in `param_names` order."""
        return cls(*np.asarray(p, dtype=np.float64).ravel())

    @classmethod
    def param_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        """Optimizer bounds: ``alpha in (0, 1)``, tilt/radial terms modestly bounded."""
        lb = np.array([1.0, 1.0, -1e5, -1e5, 1e-6, -2.0, -2.0, -0.5, -0.5],
                      dtype=np.float64)
        ub = np.array([1e5, 1e5, 1e5, 1e5, 1.0 - 1e-6, 2.0, 2.0, 0.5, 0.5],
                      dtype=np.float64)
        return lb, ub

    def initialize_from_correspondences(self, K_seed, rays, pixels) -> None:
        """Seed ``fx,fy,cx,cy`` from `K_seed`; solve ``alpha`` linearly (same UCM
        closed form as `UCMModel`); zero the radial/tilt terms."""
        self.fx, self.fy = float(K_seed[0, 0]), float(K_seed[1, 1])
        self.cx, self.cy = float(K_seed[0, 2]), float(K_seed[1, 2])
        rays = np.asarray(rays, dtype=np.float64)
        x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]
        mx = (pixels[:, 0] - self.cx) / self.fx
        my = (pixels[:, 1] - self.cy) / self.fy
        # Same UCM linear solve as ucm.py: alpha = (x - mx*z)/(mx*(1-z)).
        A = np.concatenate([mx * (1.0 - z), my * (1.0 - z)])
        b = np.concatenate([x - mx * z, y - my * z])
        denom = float(A @ A)
        self.alpha = float(np.clip((A @ b) / denom, 1e-6, 1.0 - 1e-6)) if denom > 1e-12 else 0.5
        self.lambda1 = 0.0
        self.lambda2 = 0.0
        self.tau_x = 0.0
        self.tau_y = 0.0

    def to_dict(self) -> dict:
        """Serialize to ``{"model": "dsplus", "fx": ..., ..., "tau_y": ...}``."""
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DSPlusModel":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(**{k: d[k] for k in cls.param_names})

    def __repr__(self) -> str:
        return ("DSPlusModel(fx={:.3f}, fy={:.3f}, cx={:.3f}, cy={:.3f}, "
                "alpha={:.4f}, lambda1={:.5f}, lambda2={:.5f}, "
                "tau_x={:.5f}, tau_y={:.5f})").format(
                    self.fx, self.fy, self.cx, self.cy, self.alpha,
                    self.lambda1, self.lambda2, self.tau_x, self.tau_y)
