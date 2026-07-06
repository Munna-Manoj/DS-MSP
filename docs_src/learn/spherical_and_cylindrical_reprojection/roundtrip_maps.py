"""Chapter — sphere/cylinder/pinhole are the same bijection, in code.

Composing sphere -> cylinder -> sphere (and sphere -> pinhole -> sphere) with the
closed-form conversion maps from the chapter's table should return every pixel to
where it started, to float64 round-off. This is that composition, self-contained
(no fixtures, no camera model -- just the pixel<->ray formulas themselves).
"""

import numpy as np

F, CX, CY = 320.0, 600.0, 350.0        # panorama focal (px/rad) and centre, shared by sphere & cylinder
FP, CXP, CYP = 500.0, 600.0, 350.0     # pinhole focal (px) and centre -- independent intrinsics


def sphere_pix_to_ray(u, v):
    lam, psi = (u - CX) / F, (CY - v) / F
    cps = np.cos(psi)
    return np.stack([cps * np.sin(lam), -np.sin(psi), cps * np.cos(lam)], axis=-1)


def ray_to_sphere_pix(d):
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    lam, psi = np.arctan2(x, z), np.arctan2(-y, np.hypot(x, z))
    return np.stack([CX + F * lam, CY - F * psi], axis=-1)


def ray_to_cylinder_pix(d):
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return np.stack([CX + F * np.arctan2(x, z), CY + F * y / np.hypot(x, z)], axis=-1)


def cylinder_pix_to_ray(u, v):
    lam, h = (u - CX) / F, (CY - v) / F   # h == tan(elevation)
    return np.stack([np.sin(lam), -h, np.cos(lam)], axis=-1)


def ray_to_pinhole_pix(d):
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return np.stack([CXP + FP * x / z, CYP + FP * y / z], axis=-1)


def pinhole_pix_to_ray(u, v):
    return np.stack([(u - CXP) / FP, (v - CYP) / FP, np.ones_like(u)], axis=-1)


def main() -> None:
    u, v = np.meshgrid(np.linspace(0, 1200, 200), np.linspace(0, 700, 120))
    ray = sphere_pix_to_ray(u, v)

    # sphere -> cylinder -> sphere
    cyl_pix = ray_to_cylinder_pix(ray)
    ray_back = cylinder_pix_to_ray(cyl_pix[..., 0], cyl_pix[..., 1])
    sphere_back = ray_to_sphere_pix(ray_back)
    resid_cyl = np.hypot(sphere_back[..., 0] - u, sphere_back[..., 1] - v)
    print(f"sphere -> cylinder -> sphere: max residual = {resid_cyl.max():.2e} px")

    # sphere -> pinhole -> sphere (front hemisphere only: pinhole dies at z<=0)
    front = ray[..., 2] > 0
    pin_pix = ray_to_pinhole_pix(ray[front])
    ray_back_p = pinhole_pix_to_ray(pin_pix[..., 0], pin_pix[..., 1])
    sphere_back_p = ray_to_sphere_pix(ray_back_p)
    resid_pin = np.hypot(sphere_back_p[..., 0] - u[front], sphere_back_p[..., 1] - v[front])
    print(f"sphere -> pinhole  -> sphere: max residual = {resid_pin.max():.2e} px   (front hemisphere only)")


if __name__ == "__main__":
    main()
