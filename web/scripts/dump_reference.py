"""Dump project/unproject reference values from ds_msp for the TS port to match.

Writes /tmp/ds_ref.json: for each model, a list of {P, u, v, valid, ray} where
ray is the library's unproject of (u,v). The TS port must reproduce these.
"""
import json
import math
import numpy as np
from ds_msp.models import (
    DoubleSphereModel, UCMModel, EUCMModel,
    KannalaBrandtModel, RadTanModel, OCamModel,
    DSPlusModel, EUCMPlusModel,
)

# Same defaults as web/src/lib/cameras.ts
MODELS = {
    "ds":       DoubleSphereModel(fx=180, fy=180, cx=320, cy=320, xi=0.3, alpha=0.6),
    "ucm":      UCMModel(fx=180, fy=180, cx=320, cy=320, alpha=0.6),
    "eucm":     EUCMModel(fx=180, fy=180, cx=320, cy=320, alpha=0.6, beta=1.0),
    "kb":       KannalaBrandtModel(fx=180, fy=180, cx=320, cy=320, k1=0, k2=0, k3=0, k4=0),
    "radtan":   RadTanModel(fx=320, fy=320, cx=320, cy=320, k1=0, k2=0, p1=0, p2=0, k3=0),
    "ocam":     OCamModel(cx=320, cy=320, c=1, d=0, e=0, a0=-230, a1=0, a2=0.0016, a3=0, a4=0),
    "dsplus":   DSPlusModel(fx=180, fy=180, cx=320, cy=320, alpha=0.6,
                            lambda1=0.0, lambda2=0.0, tau_x=0.0, tau_y=0.0),
    "eucmplus": EUCMPlusModel(fx=180, fy=180, cx=320, cy=320, alpha=0.6, beta=1.0,
                              lambda1=0.0, tau_x=0.0, tau_y=0.0),
}

# directions spanning front hemisphere and wide angles
def dirs():
    out = []
    for th_deg in (5, 25, 45, 70, 89, 100, 125):
        th = math.radians(th_deg)
        for ph_deg in (0, 40, 90, 155, 215, 290):
            ph = math.radians(ph_deg)
            out.append([math.sin(th) * math.cos(ph),
                        math.sin(th) * math.sin(ph),
                        math.cos(th)])
    return out

ref = {}
for name, model in MODELS.items():
    pts = np.array(dirs(), dtype=float)
    px, valid = model.project(pts)
    recs = []
    for i, P in enumerate(pts):
        u, v = float(px[i, 0]), float(px[i, 1])
        ok = bool(valid[i])
        ray = None
        if ok and math.isfinite(u) and math.isfinite(v):
            r, rv = model.unproject(np.array([[u, v]], dtype=float))
            if bool(rv[0]):
                ray = [float(r[0, 0]), float(r[0, 1]), float(r[0, 2])]
        recs.append({"P": [float(P[0]), float(P[1]), float(P[2])],
                     "u": u, "v": v, "valid": ok, "ray": ray})
    ref[name] = recs

with open("/tmp/ds_ref.json", "w") as f:
    json.dump(ref, f)
print("wrote /tmp/ds_ref.json:", {k: len(v) for k, v in ref.items()})
