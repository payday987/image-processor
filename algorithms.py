# -*- coding: utf-8 -*-
"""
数字图像处理算法集合
- 交互路径走 OpenCV 快路径（流畅）
- 每个算法保留手写教学版（_slow 后缀），便于课程报告对照原理
输入输出统一为 HWC uint8 ndarray，BGR 顺序。
"""
import numpy as np
import cv2


# ============== 工具 ==============
def _to_float(img):
    return img.astype(np.float64)


def _clip(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def _gray(img):
    if img.ndim == 2:
        return img.astype(np.float64)
    return 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]


# ============== 1. 点处理 ==============
def to_gray(img):
    return _clip(_gray(img))


def adjust_brightness(img, beta=0):
    return _clip(_to_float(img) + beta)


def adjust_contrast(img, alpha=1.0):
    return _clip(_to_float(img) * alpha + 128 * (1 - alpha))


def invert(img):
    """反色/负片：out = 255 - img"""
    return 255 - img


def threshold_binary(img, th=128):
    g = _gray(img)
    return np.where(g >= th, 255, 0).astype(np.uint8)


def gamma_transform(img, gamma=1.0):
    n = _to_float(img) / 255.0
    return _clip(255.0 * np.power(n, gamma))


def log_transform(img, c=None):
    n = _to_float(img)
    if c is None:
        c = 255.0 / np.log(1 + 255)
    return _clip(c * np.log(1 + n))


def linear_gray(img, a=1.0, b=0.0):
    return _clip(_to_float(img) * a + b)


def stretch_gray(img, low=20, high=235):
    n = _to_float(img)
    out = (n - low) * 255.0 / (high - low)
    return _clip(out)


# ============== 2. 直方图 ==============
def hist_gray(img):
    g = _gray(img).astype(np.uint8)
    return np.bincount(g.ravel(), minlength=256)


def hist_rgb(img):
    return {
        'B': np.bincount(img[..., 0].ravel(), minlength=256),
        'G': np.bincount(img[..., 1].ravel(), minlength=256),
        'R': np.bincount(img[..., 2].ravel(), minlength=256),
    }


def hist_equalize(img):
    if img.ndim == 2:
        g = img
    else:
        g = _gray(img)
    h = hist_gray(g.astype(np.uint8))
    cdf = h.cumsum()
    nz = cdf[h > 0]
    cdf_min = nz.min() if nz.size else 0
    total = g.size
    mapped = (cdf - cdf_min) * 255.0 / max(1, (total - cdf_min))
    out = mapped[g.astype(np.uint8)]
    if img.ndim == 2:
        return _clip(out)
    old_l = _gray(img)[..., None]
    return _clip(_to_float(img) + (out[..., None] - old_l))


# ============== 3. 空间滤波（快路径 cv2） ==============
def mean_filter(img, k=3):
    return cv2.blur(img, (k, k))


def gaussian_filter(img, k=5, sigma=1.0):
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), sigma)


def median_filter(img, k=3):
    if k % 2 == 0:
        k += 1
    return cv2.medianBlur(img, k)


# ---- 手写教学版（慢，保留供报告对照） ----
def mean_filter_slow(img, k=3):
    p = k // 2
    pad = np.pad(img.astype(np.float64), ((p, p), (p, p), (0, 0)), mode='reflect')
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = pad[i:i + k, j:j + k].reshape(-1, img.shape[2]).mean(axis=0)
    return _clip(out)


# ============== 4. 边缘检测 ==============
def sobel_edge(img):
    g = _gray(img).astype(np.uint8)
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    return _clip(np.sqrt(sx ** 2 + sy ** 2))


def prewitt_edge(img):
    g = _gray(img)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    ky = kx.T
    return _clip(np.abs(_conv2d(g, kx)) + np.abs(_conv2d(g, ky)))


def roberts_edge(img):
    g = _gray(img)
    k1 = np.array([[1, 0], [0, -1]], dtype=np.float64)
    k2 = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    p1 = np.zeros_like(g); p2 = np.zeros_like(g)
    for i in range(g.shape[0] - 1):
        for j in range(g.shape[1] - 1):
            p1[i, j] = (g[i:i + 2, j:j + 2] * k1).sum()
            p2[i, j] = (g[i:i + 2, j:j + 2] * k2).sum()
    return _clip(np.sqrt(p1 ** 2 + p2 ** 2))


def laplace_edge(img):
    g = _gray(img).astype(np.uint8)
    return _clip(np.abs(cv2.Laplacian(g, cv2.CV_64F, ksize=3)))


def canny_edge(img, low=50, high=150):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.Canny(g, low, high)


def _conv2d(img, kernel):
    p = kernel.shape[0] // 2
    pad = np.pad(img, p, mode='reflect')
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = (pad[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel).sum()
    return out


# ============== 5. 锐化 ==============
def laplace_sharpen(img):
    g = _gray(img).astype(np.uint8)
    lap = cv2.Laplacian(g, cv2.CV_64F, ksize=3)
    sharp = g - lap
    if img.ndim == 2:
        return _clip(sharp)
    old_l = _gray(img)
    return _clip(_to_float(img) + (sharp - old_l)[..., None])


# ============== 6. 形态学（新增） ==============
def _kernel(k):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))


def erode(img, k=3, iter=1):
    return cv2.erode(img, _kernel(k), iterations=iter)


def dilate(img, k=3, iter=1):
    return cv2.dilate(img, _kernel(k), iterations=iter)


def morph_open(img, k=3, iter=1):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, _kernel(k), iterations=iter)


def morph_close(img, k=3, iter=1):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, _kernel(k), iterations=iter)


# ============== 7. 伪彩色（新增） ==============
COLORMAPS = {
    'Jet': cv2.COLORMAP_JET,
    'Hot': cv2.COLORMAP_HOT,
    'Cool': cv2.COLORMAP_COOL,
    'Spring': cv2.COLORMAP_SPRING,
    'Winter': cv2.COLORMAP_WINTER,
    'Rainbow': cv2.COLORMAP_RAINBOW,
}


def pseudocolor(img, cmap_name='Jet'):
    g = _gray(img).astype(np.uint8)
    return cv2.applyColorMap(g, COLORMAPS.get(cmap_name, cv2.COLORMAP_JET))


# ============== 8. 几何变换 ==============
def flip_h(img): return img[:, ::-1].copy()
def flip_v(img): return img[::-1].copy()
def rotate90(img): return np.rot90(img, k=1).copy()
def rotate180(img): return np.rot90(img, k=2).copy()
def rotate270(img): return np.rot90(img, k=3).copy()


# ============== 9. 质量评价（新增） ==============
def psnr(img1, img2):
    if img1.shape != img2.shape:
        return float('nan')
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)
