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
        return cls(711.57, 711.24, 949.18, 518.81, 0.62, -0.10, 0.02, 0.001, -0.001)

    @property
    def params(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.cx, self.cy, self.alpha,
                         self.lambda1, self.lambda2, self.tau_x, self.tau_y],
                        dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def distortion(self) -> np.ndarray:
        return np.array([self.alpha, self.lambda1, self.lambda2,
                         self.tau_x, self.tau_y], dtype=np.float64)

    def project(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u, v, valid = dsplus_project(
            np.asarray(P, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, self.alpha,
            self.lambda1, self.lambda2, self.tau_x, self.tau_y)
        return np.stack([u, v], axis=-1), valid

    def unproject(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return dsplus_unproject(
            np.asarray(uv, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, self.alpha,
            self.lambda1, self.lambda2, self.tau_x, self.tau_y)

    def project_jacobian(self, P):
        u, v, J_point, J_param, valid = dsplus_project_jacobian(
            np.asarray(P, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, self.alpha,
            self.lambda1, self.lambda2, self.tau_x, self.tau_y)
        return np.stack([u, v], axis=-1), J_point, J_param, valid

    @classmethod
    def from_params(cls, p: np.ndarray) -> "DSPlusModel":
        return cls(*np.asarray(p, dtype=np.float64).ravel())

    @classmethod
    def param_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.array([1.0, 1.0, -1e5, -1e5, 1e-6, -2.0, -2.0, -0.5, -0.5],
                      dtype=np.float64)
        ub = np.array([1e5, 1e5, 1e5, 1e5, 1.0 - 1e-6, 2.0, 2.0, 0.5, 0.5],
                      dtype=np.float64)
        return lb, ub

    def initialize_from_correspondences(self, K_seed, rays, pixels) -> None:
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
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DSPlusModel":
        return cls(**{k: d[k] for k in cls.param_names})

    def __repr__(self) -> str:
        return ("DSPlusModel(fx={:.3f}, fy={:.3f}, cx={:.3f}, cy={:.3f}, "
                "alpha={:.4f}, lambda1={:.5f}, lambda2={:.5f}, "
                "tau_x={:.5f}, tau_y={:.5f})").format(
                    self.fx, self.fy, self.cx, self.cy, self.alpha,
                    self.lambda1, self.lambda2, self.tau_x, self.tau_y)
