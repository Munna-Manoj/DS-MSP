"""
Automatic target-model selection for conversion.

``convert`` finds the best parameters for a *given* target model. But a target
model can be unable to represent a lens *at any parameters* — a representability
limit, not an optimiser failure. The deciding quantity is the number of
independent **shape** degrees of freedom each model exposes in its radial profile
r(theta). Expand the normalised profile as ``r/r'(0) = theta + c3 theta^3 +
c5 theta^5 + ...``; the map (shape params) -> (c3, c5, c7, ...) has image
dimension equal to the shape-parameter count: UCM 1, Double Sphere 2, EUCM 2,
Kannala-Brandt 4. A Kannala-Brandt source fixes four independent
coefficients (k1..k4); a model with fewer shape dof generically cannot reach that
point, so its *global* optimum carries a strictly positive residual (tens of
pixels for a strongly-distorted lens). This is geometry of the model family, not
conditioning or local minima — DS still ties KB sub-pixel on ordinary fisheyes,
whose higher-order distortion terms happen to be consistent with the leading
term (i.e. they lie on the DS image).

``convert_best`` makes "find an optimal solution in any scenario" concrete: it
tries a capability-ordered ladder of target models and returns the **simplest one
that meets a reprojection-RMS tolerance** (Occam's razor — prefer the fewest
parameters that suffice). If none meet the tolerance it returns the best
available and says so. The default ladder ends in Kannala-Brandt, the most
capable shipped model (four shape dof, a polynomial in the incidence angle).

This module is allowed to depend on concrete models (it provides a default
ladder); ``convert`` itself stays model-free.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Type

from ..core.contracts import CameraModel
from .convert import convert


def default_ladder() -> Tuple[Type[CameraModel], ...]:
    """Capability-ordered target models, simplest (fewest params) first."""
    from ..models import DoubleSphereModel, EUCMModel, KannalaBrandtModel, UCMModel
    # UCM(5) < DS(6) = EUCM(6) < KB(8). DS before EUCM keeps the namesake
    # sphere model preferred when a 2-extra-param sphere fit suffices.
    return (UCMModel, DoubleSphereModel, EUCMModel, KannalaBrandtModel)


def convert_best(source: CameraModel, *, width: int, height: int,
                 candidates: Optional[Sequence[Type[CameraModel]]] = None,
                 target_rms: float = 3.0, n_samples: int = 2000,
                 n_restarts: int = 6, max_fov_deg: Optional[float] = None,
                 ) -> Tuple[Optional[CameraModel], dict,
                            List[Tuple[Optional[CameraModel], dict]]]:
    """Convert ``source`` to the simplest target model meeting ``target_rms``.

    Parameters
    ----------
    source : CameraModel
        The calibrated source model to convert.
    width, height : int
        Image size for sampling and reporting.
    candidates : sequence of model classes, optional
        Capability-ordered ladder to try. Defaults to :func:`default_ladder`.
    target_rms : float
        Reprojection-RMS tolerance in pixels. The first candidate at or under
        this value is returned.
    n_samples, n_restarts, max_fov_deg
        Forwarded to :func:`convert`.

    Returns
    -------
    (model, report, all_results)
        ``model``/``report`` for the selected target; ``all_results`` is the list
        of ``(model, report)`` for every candidate attempted, in order. The
        chosen ``report`` carries ``selected``/``selected_reason`` keys.
    """
    cands = tuple(candidates) if candidates is not None else default_ladder()
    results: List[Tuple[Optional[CameraModel], dict]] = []
    for cls in cands:
        try:
            model, report = convert(source, cls, width=width, height=height,
                                    n_samples=n_samples, n_restarts=n_restarts,
                                    max_fov_deg=max_fov_deg)
        except Exception as exc:  # a model that cannot be fitted at all is skipped
            results.append((None, {"target_model": cls.name, "error": str(exc),
                                    "rms_px": float("inf")}))
            continue
        results.append((model, report))
        if report["rms_px"] <= target_rms:
            report["selected"] = True
            report["selected_reason"] = (
                f"simplest model with RMS {report['rms_px']:.3f}px <= {target_rms}px")
            return model, report, results

    # Nothing met the tolerance: return the best available, flagged honestly.
    best_model, best_report = min(
        results, key=lambda mr: mr[1].get("rms_px", float("inf")))
    best_report["selected"] = True
    best_report["selected_reason"] = (
        f"target {target_rms}px not met by any candidate; best available "
        f"is {best_report['target_model']} at RMS {best_report['rms_px']:.3f}px")
    return best_model, best_report, results
