# mmf.py
import numpy as np
import cv2

def maxwell_garnett_two_phase(eps_m, eps_i: float, f: np.ndarray) -> np.ndarray:
    f = np.clip(f, 0.0, 1.0).astype(np.float32)
    eps_m = np.asarray(eps_m, dtype=np.float32)

    denom = (eps_i + 2.0 * eps_m)
    denom = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
    alpha = (eps_i - eps_m) / denom

    num = 1.0 + 2.0 * f * alpha
    den = 1.0 - 1.0 * f * alpha
    den = np.where(np.abs(den) > 1e-8, den, 1e-8)
    return eps_m * (num / den)

def eps_eff_map_from_rgb(
    rgb01: np.ndarray,
    eps_host: float = 2.0,
    eps_channels=(3.0, 4.0, 5.0),
    clip=(1.0, 20.0),
) -> np.ndarray:
    if rgb01.ndim != 3 or rgb01.shape[2] != 3:
        raise ValueError("Expected rgb01 shape (H,W,3)")

    I = rgb01.astype(np.float32)
    s = I.sum(axis=2, keepdims=True) + 1e-8
    f = I / s

    epsA, epsB, epsC = map(float, eps_channels)
    e1 = maxwell_garnett_two_phase(eps_host, epsA, f[..., 0])
    e2 = maxwell_garnett_two_phase(e1, epsB, f[..., 1])
    e3 = maxwell_garnett_two_phase(e2, epsC, f[..., 2])
    return np.clip(e3, clip[0], clip[1]).astype(np.float32)

def eps_feature_stack(
    rgb01: np.ndarray,
    eps_clip=(1.0, 20.0),
    eps_host=2.0,
    eps_channels=(3.0,4.0,5.0),
    var_ksize=9,
) -> np.ndarray:
    eps = eps_eff_map_from_rgb(rgb01, eps_host=eps_host, eps_channels=eps_channels, clip=eps_clip)

    # normalize eps to 0..1
    eps01 = (eps - eps_clip[0]) / (eps_clip[1] - eps_clip[0] + 1e-8)
    eps01 = np.clip(eps01, 0.0, 1.0).astype(np.float32)

    # local variance via E[x^2]-E[x]^2 using box filter
    k = int(var_ksize)
    if k % 2 == 0:
        k += 1
    mean = cv2.blur(eps01, (k, k))
    mean2 = cv2.blur(eps01 * eps01, (k, k))
    var = np.clip(mean2 - mean * mean, 0.0, 1.0).astype(np.float32)

    # gradient magnitude via Sobel
    gx = cv2.Sobel(eps01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eps01, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    # normalize grad robustly
    gmax = float(np.percentile(grad, 99.5)) + 1e-8
    grad01 = np.clip(grad / gmax, 0.0, 1.0).astype(np.float32)

    return np.stack([eps01, var, grad01], axis=2)  
