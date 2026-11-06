"""
Test-support utilities shipped with the package (à la ``numpy.testing``).

Provides:
- :class:`FakeModel`: a trivial perfect-pinhole camera that satisfies the
  :class:`~ds_msp.core.contracts.CameraModel` Protocol. Services and the
  converter are tested against it, proving they depend only on the contract and
  run in the **absence of any real (fisheye) camera model**.
- Finite-difference Jacobian helpers and sampling utilities reused by the
  model-agnostic contract test suite, and available to downstream users who want
  to validate their own model implementations.

Pure NumPy — no OpenCV/SciPy — so it imports cleanly anywhere.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from .core.contracts import CameraModel


class FakeModel:
    """Reference pinhole model: ``u = fx·x/z + cx``, ``v = fy·y/z + cy``.

    Implements the full :class:`CameraModel` contract with closed-form analytic
    Jacobians. Used as a dependency-free stand-in so the rest of the library can
    be developed and tested before any fisheye model exists.
    """

    name = "fake_pinhole"
    param_names = ("fx", "fy", "cx", "cy")

    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

    # -- factory used by the contract suite -------------------------------
    @classmethod
    def sample(cls) -> "FakeModel":
        """A realistic instance for contract testing."""
        return cls(fx=600.0, fy=605.0, cx=320.0, cy=240.0)

    # -- parameter access -------------------------------------------------
    @property
    def params(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.cx, self.cy], dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def distortion(self) -> np.ndarray:
        return np.empty(0, dtype=np.float64)

    # -- core math --------------------------------------------------------
    def project(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        P = np.asarray(P, dtype=np.float64)
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        valid = z > 1e-9
        zsafe = np.where(valid, z, 1.0)
        u = self.fx * x / zsafe + self.cx
        v = self.fy * y / zsafe + self.cy
        uv = np.stack([u, v], axis=-1)
        uv = np.where(valid[..., None], uv, 0.0)
        return uv, valid

    def unproject(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        uv = np.asarray(uv, dtype=np.float64)
        mx = (uv[..., 0] - self.cx) / self.fx
        my = (uv[..., 1] - self.cy) / self.fy
        rays = np.stack([mx, my, np.ones_like(mx)], axis=-1)
        rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        valid = np.ones(uv.shape[:-1], dtype=bool)
        return rays, valid

    # -- analytic Jacobian ------------------------------------------------
    def project_jacobian(
        self, P: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        P = np.asarray(P, dtype=np.float64)
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        valid = z > 1e-9
        zsafe = np.where(valid, z, 1.0)
        inv = 1.0 / zsafe
        u = self.fx * x * inv + self.cx
        v = self.fy * y * inv + self.cy
        uv = np.stack([u, v], axis=-1)

        J_point = np.zeros(P.shape[:-1] + (2, 3), dtype=np.float64)
        J_point[..., 0, 0] = self.fx * inv
        J_point[..., 0, 2] = -self.fx * x * inv * inv
        J_point[..., 1, 1] = self.fy * inv
        J_point[..., 1, 2] = -self.fy * y * inv * inv

        J_param = np.zeros(P.shape[:-1] + (2, 4), dtype=np.float64)
        J_param[..., 0, 0] = x * inv   # du/dfx
        J_param[..., 0, 2] = 1.0       # du/dcx
        J_param[..., 1, 1] = y * inv   # dv/dfy
        J_param[..., 1, 3] = 1.0       # dv/dcy
        return uv, J_point, J_param, valid

    # -- construction / bounds -------------------------------------------
    @classmethod
    def from_params(cls, p: np.ndarray) -> "FakeModel":
        p = np.asarray(p, dtype=np.float64).ravel()
        return cls(*p)

    @classmethod
    def param_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.array([1.0, 1.0, -1e5, -1e5], dtype=np.float64)
        ub = np.array([1e5, 1e5, 1e5, 1e5], dtype=np.float64)
        return lb, ub

    # -- conversion hook --------------------------------------------------
    def initialize_from_correspondences(
        self, K_seed: np.ndarray, rays: np.ndarray, pixels: np.ndarray
    ) -> None:
        # Linear least-squares seed: u = fx*(x/z) + cx, per axis.
        xn = rays[:, 0] / rays[:, 2]
        yn = rays[:, 1] / rays[:, 2]
        A_x = np.stack([xn, np.ones_like(xn)], axis=1)
        A_y = np.stack([yn, np.ones_like(yn)], axis=1)
        (self.fx, self.cx) = np.linalg.lstsq(A_x, pixels[:, 0], rcond=None)[0]
        (self.fy, self.cy) = np.linalg.lstsq(A_y, pixels[:, 1], rcond=None)[0]

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        d = {"model": self.name}
        d.update({k: float(v) for k, v in zip(self.param_names, self.params)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FakeModel":
        return cls(**{k: d[k] for k in cls.param_names})


# ---------------------------------------------------------------------------
# Reusable contract helpers (used by the model-agnostic test suite and by
# downstream users validating their own CameraModel implementations).
# ---------------------------------------------------------------------------

def sample_forward_points(n: int = 48, seed: int = 0,
                          z_range: Tuple[float, float] = (1.0, 5.0),
                          xy: float = 0.8) -> np.ndarray:
    """Deterministic camera-frame points in front of the camera (``z > 0``)."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(z_range[0], z_range[1], n)
    x = rng.uniform(-xy, xy, n) * z
    y = rng.uniform(-xy, xy, n) * z
    return np.stack([x, y, z], axis=-1)


def finite_difference_point_jacobian(model: CameraModel, P: np.ndarray,
                                     eps: float = 1e-6) -> np.ndarray:
    """Numerical ``d(u,v)/d(x,y,z)`` via central differences, shape ``(N,2,3)``."""
    P = np.asarray(P, dtype=np.float64)
    J = np.zeros(P.shape[:-1] + (2, 3), dtype=np.float64)
    for k in range(3):
        dp = np.zeros(3)
        dp[k] = eps
        up, _ = model.project(P + dp)
        um, _ = model.project(P - dp)
        J[..., k] = (up - um) / (2 * eps)
    return J


def finite_difference_param_jacobian(model: CameraModel, P: np.ndarray,
                                     eps: float = 1e-6) -> np.ndarray:
    """Numerical ``d(u,v)/d(params)`` via central differences, shape ``(N,2,P)``.

    Uses a per-parameter RELATIVE step (scaled by ``|p_k|``) so it stays accurate
    for parameters spanning many magnitudes (e.g. a KB coefficient ~1e-4
    alongside a focal ~1e3), where a fixed absolute step would be meaningless.
    """
    p = np.asarray(model.params, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    J = np.zeros(P.shape[:-1] + (2, p.size), dtype=np.float64)
    cls = type(model)
    for k in range(p.size):
        h = eps * max(abs(p[k]), 1.0)
        pp = p.copy()
        pp[k] += h
        pm = p.copy()
        pm[k] -= h
        up, _ = cls.from_params(pp).project(P)
        um, _ = cls.from_params(pm).project(P)
        J[..., k] = (up - um) / (2 * h)
    return J


def _relerr(a: np.ndarray, b: np.ndarray) -> float:
    """Frobenius relative error ``‖a − b‖ / max(‖a‖, eps)``."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    den = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / max(den, 1e-12))


def gradcheck_project(model: CameraModel, P: np.ndarray = None, *,
                      rel_tol: float = 1e-6, h: float = 1e-4) -> dict:
    """Strict analytic-vs-finite-difference check of ``project_jacobian``.

    Unlike the cheap 5e-3 smoke check, this uses **Richardson extrapolation** of central
    differences — combining steps ``h`` and ``h/2`` as ``(4·D(h/2) − D(h))/3`` cancels the
    ``O(h²)`` truncation term, leaving ``O(h⁴))`` so the residual reflects the *analytic*
    Jacobian's correctness, not the differencing scheme. Returns relative Frobenius errors
    for ``J_point`` and ``J_param`` over valid points, plus an ``ok`` verdict.

    This is the package's standing "differentiability contract": every model's hand-derived
    Jacobian must agree with finite differences to ``rel_tol``.
    """
    if P is None:
        P = sample_forward_points()
    _, J_point_an, J_param_an, valid = model.project_jacobian(P)

    # Richardson: f' ≈ (4·D(h/2) − D(h)) / 3 from central differences.
    Jp = (4.0 * finite_difference_point_jacobian(model, P, eps=h / 2)
          - finite_difference_point_jacobian(model, P, eps=h)) / 3.0
    Jpar = (4.0 * finite_difference_param_jacobian(model, P, eps=h / 2)
            - finite_difference_param_jacobian(model, P, eps=h)) / 3.0

    m = np.asarray(valid, bool)
    point_err = _relerr(J_point_an[m], Jp[m])
    param_err = _relerr(J_param_an[m], Jpar[m])
    return {
        "point_rel_err": point_err,
        "param_rel_err": param_err,
        "rel_tol": rel_tol,
        "ok": point_err < rel_tol and param_err < rel_tol,
    }


def gradcheck_retraction(*, rel_tol: float = 1e-6, h: float = 1e-4,
                         seed: int = 0, n: int = 24) -> dict:
    """Strict check of the SO(3) right Jacobian used by the manifold re-basing solver.

    ``J_r(w)`` satisfies ``Exp(w+δ) ≈ Exp(w)·Exp(J_r(w)·δ)`` for small ``δ`` — i.e. column
    ``k`` of ``J_r(w)`` is ``∂/∂δ_k Log(Exp(w)ᵀ·Exp(w+δ))`` at ``δ=0``. Compares that against
    Richardson-extrapolated finite differences over random tangents (avoiding ``‖w‖≈π``).
    Makes the manifold Jacobian — the foundation of the re-basing LM — a first-class contract.
    """
    from .core.lie import so3_exp, so3_log, so3_right_jacobian

    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        w = rng.uniform(-2.0, 2.0, 3)            # ‖w‖ < ~3.46, comfortably off the π shell
        Jr_an = so3_right_jacobian(w)
        Rw_T = so3_exp(w).T

        def g(delta):
            return so3_log(Rw_T @ so3_exp(w + delta))

        Jr_fd = np.zeros((3, 3))
        for k in range(3):
            e = np.zeros(3)
            e[k] = 1.0
            d1 = (g(e * (h / 2)) - g(-e * (h / 2))) / h          # central, step h/2
            d2 = (g(e * h) - g(-e * h)) / (2 * h)                # central, step h
            Jr_fd[:, k] = (4.0 * d1 - d2) / 3.0                  # Richardson
        worst = max(worst, _relerr(Jr_an, Jr_fd))
    return {"rel_err": worst, "rel_tol": rel_tol, "ok": worst < rel_tol}


#: Sample factories the contract suite always runs against. Later phases append
#: ``(UCMModel.sample)``, ``(KannalaBrandtModel.sample)``, etc.
REFERENCE_MODELS: List[Tuple[str, Callable[[], CameraModel]]] = [
    ("fake_pinhole", FakeModel.sample),
]
