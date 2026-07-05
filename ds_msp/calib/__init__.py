"""Generic calibration (bundle adjustment) for any CameraModel.

``calibrate`` and ``AprilGridTarget`` are dependency-light. ``detect_aprilgrid``
needs the optional ``aprilgrid`` backend (``pip install ds_msp[calib]``) and is
imported lazily so this package stays importable without it.

``load_camera`` re-exports :func:`ds_msp.io.kalibr.load_kalibr` under a name that doesn't
require knowing the output is Kalibr-formatted — ``ds-msp-calibrate``'s output is meant to be
loaded straight back into a ready :class:`~ds_msp.core.contracts.CameraModel` instance, not
inspected as a file format.
"""

from .bundle import calibrate
from .stereo import estimate_relative_pose, relative_pose_error
from .targets import AprilGridTarget

__all__ = ["calibrate", "AprilGridTarget", "detect_aprilgrid", "load_camera",
           "estimate_relative_pose", "relative_pose_error"]


def __getattr__(name: str):
    # Lazy: only pull in the OpenCV+aprilgrid detection adapter on demand.
    if name == "detect_aprilgrid":
        from .detect import detect_aprilgrid
        return detect_aprilgrid
    if name == "load_camera":
        from ..io.kalibr import load_kalibr
        return load_kalibr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
