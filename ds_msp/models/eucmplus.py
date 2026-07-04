# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025-2026 Munna-Manoj. EUCM+ (Extended UCM Plus) camera model, from
# DS-MSP (https://github.com/Munna-Manoj/DS-MSP). NONCOMMERCIAL use only, with
# attribution — see LICENSE-NONCOMMERCIAL.txt and LICENSING.md. The rest of DS-MSP is MIT.
"""EUCM+ camera model (EUCM core + division radial + 2-axis tilt) implementing
the CameraModel contract.

EUCM+ is the truly-closed-form (sqrt-only) sibling of :class:`DSPlusModel`. It
swaps DS+'s UCM core for the Enhanced UCM core (adding ``beta``) and keeps a
single Fitzgibbon division term (``lambda1``) so that the entire unprojection is
solvable with square roots alone — no cube root, no polynomial root finder, no
Newton iteration. The 2-axis Scheimpflug tilt (``tau_x, tau_y``) stays linear in
the inverse. See ``eucmplus_math`` for the staged math and analytic Jacobians.
"""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .eucmplus_math import (
    eucmplus_project,
    eucmplus_project_jacobian,
    eucmplus_unproject,
)


class EUCMPlusModel:
    """EUCM+ (EUCM core + division radial + 2-axis tilt). Satisfies ``CameraModel``."""

    name: ClassVar[str] = "eucmplus"
    param_names: ClassVar[Tuple[str, ...]] = (
        "fx", "fy", "cx", "cy", "alpha", "beta", "lambda1", "tau_x", "tau_y")

    def __init__(self, fx, fy, cx, cy, alpha=0.5, beta=1.0,
                 lambda1=0.0, tau_x=0.0, tau_y=0.0) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lambda1 = float(lambda1)
        self.tau_x = float(tau_x)
        self.tau_y = float(tau_y)

    @classmethod
    def sample(cls) -> "EUCMPlusModel":
        """Realistic instance for contract testing (the bundled calibration)."""
        return cls(711.57, 711.24, 949.18, 518.81, 0.62, 1.10, -0.10, 0.001, -0.001)

    @property
    def params(self) -> np.ndarray:
        """Flat parameter vector in `param_names` order."""
        return np.array([self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta,
                         self.lambda1, self.tau_x, self.tau_y], dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        """3x3 pinhole intrinsic matrix built from ``fx, fy, cx, cy``."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def distortion(self) -> np.ndarray:
        """Distortion tail ``[alpha, beta, lambda1, tau_x, tau_y]``."""
        return np.array([self.alpha, self.beta, self.lambda1,
                         self.tau_x, self.tau_y], dtype=np.float64)

    def project(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project camera-frame points through 4 sqrt-only-invertible stages.

        EUCM+ swaps DS+'s UCM core for the Enhanced UCM core (EUCM, adding
        the ellipse-radius weight ``beta``) and keeps a single Fitzgibbon
        division term, so the *entire* inverse chain is solvable with square
        roots alone — no cube root, no polynomial root finder, no Newton
        iteration (unlike `DSPlusModel`, whose 2-term division layer needs a
        quartic radical)::

            pixel = K . H_tau . D_lambda . S_(alpha,beta)(bearing)

        Parameters
        ----------
        P : ndarray, shape (..., 3)
            Camera-frame points (meters), +Z forward.

        Returns
        -------
        uv : ndarray, shape (..., 2)
            Projected pixel coordinates ``(u, v)``, origin top-left.
        valid : ndarray, shape (...,)
            ``True`` iff all 4 stages are individually valid: the EUCM
            denominator, a nonzero division-model radial factor, and a
            nonzero tilt-homography denominator. See
            ``ds_msp.models.eucmplus_math.eucmplus_project``.

        References
        ----------
        EUCM core: Khomutenko, B., Garcia, G., Martinet, P. "An Enhanced
        Unified Camera Model for Omnidirectional Cameras." IEEE RA-L 2016.
        Division radial layer: Fitzgibbon, A. CVPR 2001. Tilt homography: cf.
        OpenCV ``CALIB_TILTED_MODEL``. Staged composition and Jacobian are
        this repo's own extension — see ``ds_msp/models/eucmplus_math.py``
        and
        [ADR-0005](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/process/architecture/decisions/ADR-0005-dsplus-eucmplus.md).

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import EUCMPlusModel
        >>> m = EUCMPlusModel.sample()
        >>> uv, valid = m.project(np.array([[0.0, 0.0, 1.0]]))
        >>> np.round(uv, 2)
        array([[949.18, 518.81]])
        """
        u, v, valid = eucmplus_project(
            np.asarray(P, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta,
            self.lambda1, self.tau_x, self.tau_y)
        return np.stack([u, v], axis=-1), valid

    def unproject(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unproject pixels to unit bearing rays (closed form, sqrt-only).

        Inverts the tilt homography linearly, the 1-term division layer with
        a single quadratic ``sqrt``, and the EUCM sphere with its own
        sqrt-only closed form — no iteration anywhere in the chain. See
        ``ds_msp.models.eucmplus_math.eucmplus_unproject``.

        Parameters
        ----------
        uv : ndarray, shape (..., 2)
            Pixel coordinates ``(u, v)``, origin top-left.

        Returns
        -------
        rays : ndarray, shape (..., 3)
            Unit-norm camera-frame bearing vectors, +Z forward.
        valid : ndarray, shape (...,)
            ``True`` iff the tilt inverse, the division-radial discriminant,
            and the EUCM sphere inverse are all well-defined. Invalid rays
            are zeroed, never NaN.

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import EUCMPlusModel
        >>> m = EUCMPlusModel.sample()
        >>> rays, valid = m.unproject(np.array([[m.cx, m.cy]]))
        >>> np.round(rays, 4)
        array([[0., 0., 1.]])
        """
        return eucmplus_unproject(
            np.asarray(uv, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta,
            self.lambda1, self.tau_x, self.tau_y)

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
            ``d(u, v) / d(fx, fy, cx, cy, alpha, beta, lambda1, tau_x, tau_y)``,
            columns in `param_names` order.
        valid : ndarray, shape (...,)
            Projectability mask, identical condition to `project`.

        Examples
        --------
        >>> import numpy as np
        >>> from ds_msp.models import EUCMPlusModel
        >>> m = EUCMPlusModel.sample()
        >>> uv, J_point, J_param, valid = m.project_jacobian(np.array([[0.0, 0.0, 1.0]]))
        >>> J_point.shape, J_param.shape
        ((1, 2, 3), (1, 2, 9))
        """
        u, v, J_point, J_param, valid = eucmplus_project_jacobian(
            np.asarray(P, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta,
            self.lambda1, self.tau_x, self.tau_y)
        return np.stack([u, v], axis=-1), J_point, J_param, valid

    @classmethod
    def from_params(cls, p: np.ndarray) -> "EUCMPlusModel":
        """Build from a flat vector in `param_names` order."""
        return cls(*np.asarray(p, dtype=np.float64).ravel())

    @classmethod
    def param_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        """Optimizer bounds: ``alpha in (0, 1]``, ``beta in (0, 4]``."""
        lb = np.array([1.0, 1.0, -1e5, -1e5, 1e-6, 1e-3, -2.0, -0.5, -0.5],
                      dtype=np.float64)
        ub = np.array([1e5, 1e5, 1e5, 1e5, 1.0, 4.0, 2.0, 0.5, 0.5],
                      dtype=np.float64)
        return lb, ub

    def initialize_from_correspondences(self, K_seed, rays, pixels) -> None:
        """Seed ``fx,fy,cx,cy`` from `K_seed`; fix ``beta=1`` and solve ``alpha`` linearly
        (the UCM closed form; ``beta=1`` reduces EUCM to UCM); zero the radial/tilt terms."""
        self.fx, self.fy = float(K_seed[0, 0]), float(K_seed[1, 1])
        self.cx, self.cy = float(K_seed[0, 2]), float(K_seed[1, 2])
        self.beta = 1.0
        rays = np.asarray(rays, dtype=np.float64)
        x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]
        mx = (pixels[:, 0] - self.cx) / self.fx
        my = (pixels[:, 1] - self.cy) / self.fy
        # beta=1, unit rays (d=1): alpha = (x - mx*z)/(mx*(1 - z)), linear LS
        # (same UCM linear solve as ucm.py / eucm.py).
        A = np.concatenate([mx * (1.0 - z), my * (1.0 - z)])
        b = np.concatenate([x - mx * z, y - my * z])
        denom = float(A @ A)
        self.alpha = float(np.clip((A @ b) / denom, 1e-6, 1.0 - 1e-6)) if denom > 1e-12 else 0.5
        self.lambda1 = 0.0
        self.tau_x = 0.0
        self.tau_y = 0.0

    def to_dict(self) -> dict:
        """Serialize to ``{"model": "eucmplus", "fx": ..., ..., "tau_y": ...}``."""
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EUCMPlusModel":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(**{k: d[k] for k in cls.param_names})

    def __repr__(self) -> str:
        return ("EUCMPlusModel(fx={:.3f}, fy={:.3f}, cx={:.3f}, cy={:.3f}, "
                "alpha={:.4f}, beta={:.4f}, lambda1={:.5f}, "
                "tau_x={:.5f}, tau_y={:.5f})").format(
                    self.fx, self.fy, self.cx, self.cy, self.alpha, self.beta,
                    self.lambda1, self.tau_x, self.tau_y)
