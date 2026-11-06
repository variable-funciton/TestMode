"""
wavelet_core.py
2次元離散ウェーブレット変換による画像特徴量の抽出と分類（UIに依存しない計算部分）

特徴量:
    レベル j = 1..J, 方向 d ∈ {H, V, D} の詳細係数 c_{j,d} について
        E_{j,d} = (1/N_{j,d}) Σ c_{j,d}[m,n]^2      （1係数あたりのエネルギー）
    を並べたベクトル（3J 次元）。必要に応じて log や相対エネルギーに変換する。
分類:
    クラス A, B の代表特徴量（プロトタイプ）との距離が近い方に分類（最近傍プロトタイプ法）。
"""
from __future__ import annotations

import numpy as np

try:
    import pywt  # PyWavelets
    HAS_PYWT = True
except ImportError:  # PyWavelets が無い場合は Haar のみ numpy で実装
    pywt = None
    HAS_PYWT = False

DIRS = ("H", "V", "D")  # 水平・垂直・対角の詳細成分


# ---------------------------------------------------------------- 2次元DWT
def _haar_dwt2(x: np.ndarray):
    """正規直交 Haar による1段の2次元DWT（pywt.dwt2(x,'haar') と同じ規約）"""
    h, w = x.shape
    x = x[: h - h % 2, : w - w % 2]
    a, b = x[0::2, 0::2], x[0::2, 1::2]
    c, d = x[1::2, 0::2], x[1::2, 1::2]
    cA = (a + b + c + d) / 2
    cH = (a + b - c - d) / 2   # 上下の差 → 水平方向のエッジ
    cV = (a - b + c - d) / 2   # 左右の差 → 垂直方向のエッジ
    cD = (a - b - c + d) / 2   # 対角
    return cA, (cH, cV, cD)


def wavedec2(img: np.ndarray, wavelet: str = "haar", level: int = 3):
    """多重解像度分解。戻り値は pywt と同じ [cA_J, (cH_J,cV_J,cD_J), ..., (cH_1,cV_1,cD_1)]"""
    img = np.asarray(img, dtype=float)
    if HAS_PYWT:
        return pywt.wavedec2(img, wavelet=wavelet, level=level, mode="periodization")
    if wavelet != "haar":
        raise ValueError("PyWavelets が無いため 'haar' のみ使用できます (pip install PyWavelets)")
    details = []
    cA = img
    for _ in range(level):
        cA, d = _haar_dwt2(cA)
        details.append(d)
    return [cA] + details[::-1]


def swt2(img: np.ndarray, wavelet: str = "haar", level: int = 3):
    """
    定常（間引きなし）ウェーブレット変換。戻り値の形式は wavedec2 と同じ
    [cA_J, (cH_J,cV_J,cD_J), ..., (cH_1,cV_1,cD_1)] だが、各係数は元画像と同じサイズ。
    巡回シフトに対してサブバンドのエネルギーが厳密に不変になる。
    """
    img = np.asarray(img, dtype=float)
    if HAS_PYWT:
        return pywt.swt2(img, wavelet, level=level, trim_approx=True, norm=True)
    if wavelet != "haar":
        raise ValueError("PyWavelets が無いため 'haar' のみ使用できます (pip install PyWavelets)")
    lo = lambda a, s, ax: (a + np.roll(a, -s, axis=ax)) / 2
    hi = lambda a, s, ax: (a - np.roll(a, -s, axis=ax)) / 2
    details, cA = [], img
    for j in range(level):
        s = 2 ** j  # à trous: フィルタを 2^j だけ間隔をあけて適用
        L, Hh = lo(cA, s, 0), hi(cA, s, 0)
        details.append((lo(Hh, s, 1), hi(L, s, 1), hi(Hh, s, 1)))
        cA = lo(L, s, 1)
    return [cA] + details[::-1]


def decompose(img, wavelet="haar", level=3, transform="dwt"):
    """transform = "dwt"（間引きあり）または "swt"（定常・シフト不変）"""
    return swt2(img, wavelet, level) if transform == "swt" else wavedec2(img, wavelet, level)


def available_wavelets():
    if HAS_PYWT:
        return ["haar", "db2", "db4", "sym4", "coif1", "bior2.2"]
    return ["haar"]


def max_level(shape, wavelet: str = "haar") -> int:
    n = min(shape)
    if HAS_PYWT:
        return pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    return int(np.floor(np.log2(n)))


# ---------------------------------------------------------------- 特徴量
def band_energies(coeffs):
    """{(level, dir): エネルギー} を返す。level=1 が最も細かいスケール"""
    J = len(coeffs) - 1
    out = {}
    for k, triple in enumerate(coeffs[1:]):
        j = J - k
        for d, c in zip(DIRS, triple):
            out[(j, d)] = float(np.mean(c ** 2))
    return out


def feature_labels(level: int):
    return [f"L{j}-{d}" for j in range(1, level + 1) for d in DIRS]


def features(img, wavelet="haar", level=3, mode="log"):
    """
    mode:
      "raw"      E_{j,d} そのもの
      "log"      log(E_{j,d})            ← スケール差が大きいので通常はこれ
      "relative" E_{j,d} / Σ E           ← コントラスト（定数倍）に不変
    """
    E = band_energies(wavedec2(img, wavelet, level))
    v = np.array([E[(j, d)] for j in range(1, level + 1) for d in DIRS])
    if mode == "log":
        return np.log(v + 1e-12)
    if mode == "relative":
        return v / (v.sum() + 1e-12)
    return v


# ---------------------------------------------------------------- 分類
def distance(u, v, metric="euclid"):
    if metric == "cosine":
        return 1.0 - float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    return float(np.linalg.norm(u - v))


def classify(x_feat, protos, metric="euclid"):
    """protos = [proto_A, proto_B] → (予測ラベル 0/1, [d_A, d_B])"""
    ds = [distance(x_feat, p, metric) for p in protos]
    return int(np.argmin(ds)), ds


# ---------------------------------------------------------------- 画像生成・劣化
def normalize01(img):
    img = np.asarray(img, dtype=float)
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)


def _smooth_noise(n, sigma, rng):
    """ガウスフィルタをかけた白色雑音（FFTで周期的に平滑化）"""
    x = rng.standard_normal((n, n))
    f = np.fft.fftfreq(n)
    K = np.exp(-2 * (np.pi * sigma) ** 2 * (f[:, None] ** 2 + f[None, :] ** 2))
    return normalize01(np.real(np.fft.ifft2(np.fft.fft2(x) * K)))


SAMPLE_PAIRS = {
    "縦縞 vs 横縞": ("縦縞", "横縞"),
    "粗いテクスチャ vs 細かいテクスチャ": ("粗いテクスチャ", "細かいテクスチャ"),
    "円 vs 正方形": ("円", "正方形"),
    "市松模様 vs 斜め縞": ("市松模様", "斜め縞"),
}


def make_sample(name: str, n: int = 128, seed: int = 0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    if name == "縦縞":
        return 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 8 * x))
    if name == "横縞":
        return 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 8 * y))
    if name == "斜め縞":
        return 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 8 * (x + y)))
    if name == "市松模様":
        return ((np.floor(x * 8) + np.floor(y * 8)) % 2).astype(float)
    if name == "粗いテクスチャ":
        return _smooth_noise(n, sigma=6.0, rng=rng)
    if name == "細かいテクスチャ":
        return _smooth_noise(n, sigma=1.0, rng=rng)
    if name == "円":
        return ((x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.3 ** 2).astype(float)
    if name == "正方形":
        return ((np.abs(x - 0.5) < 0.27) & (np.abs(y - 0.5) < 0.27)).astype(float)
    raise ValueError(name)


def perturb(img, noise=0.1, max_shift=0, contrast=(1.0, 1.0), rng=None):
    """テスト画像の生成: 平行移動（周期的）→ コントラスト倍率 → ガウス雑音"""
    rng = np.random.default_rng() if rng is None else rng
    out = np.asarray(img, dtype=float)
    if max_shift > 0:
        sy, sx = rng.integers(-max_shift, max_shift + 1, size=2)
        out = np.roll(out, (sy, sx), axis=(0, 1))
    out = out * rng.uniform(*contrast)
    return out + noise * rng.standard_normal(out.shape)


# ---------------------------------------------------------------- シミュレーション
def train_prototypes(imgA, imgB, wavelet="haar", level=3, mode="log",
                     noise=0.1, max_shift=0, contrast=(1.0, 1.0), n_train=20, rng=None):
    """学習: 各クラスの原画像に同じ劣化を加えた n_train 枚の特徴量の平均をプロトタイプとする"""
    rng = np.random.default_rng() if rng is None else rng
    protos = []
    for base in (imgA, imgB):
        F = [features(perturb(base, noise, max_shift, contrast, rng), wavelet, level, mode)
             for _ in range(n_train)]
        protos.append(np.mean(F, axis=0))
    return protos


def simulate(imgA, imgB, wavelet="haar", level=3, mode="log", metric="euclid",
             noise=0.1, max_shift=0, contrast=(1.0, 1.0), n_trials=100, n_train=20, seed=0):
    """
    学習用サンプルでプロトタイプを作ったのち、クラスA・Bそれぞれ n_trials 枚のテスト画像で
      (1) ウェーブレット特徴量による最近傍プロトタイプ分類
      (2) 比較用：画素値そのものの距離（原画像とのテンプレートマッチング）
    の正解率と混同行列を返す。
    """
    rng = np.random.default_rng(seed)
    protos = train_prototypes(imgA, imgB, wavelet, level, mode,
                              noise, max_shift, contrast, n_train, rng)
    tmpl = [np.asarray(imgA, float).ravel(), np.asarray(imgB, float).ravel()]
    conf_w = np.zeros((2, 2), int)
    conf_p = np.zeros((2, 2), int)
    feats, labels = [], []
    for true, base in enumerate((imgA, imgB)):
        for _ in range(n_trials):
            t = perturb(base, noise, max_shift, contrast, rng)
            f = features(t, wavelet, level, mode)
            pred_w, _ = classify(f, protos, metric)
            pred_p = int(np.argmin([np.linalg.norm(t.ravel() - m) for m in tmpl]))
            conf_w[true, pred_w] += 1
            conf_p[true, pred_p] += 1
            feats.append(f)
            labels.append(true)
    total = 2 * n_trials
    return {
        "acc_wavelet": np.trace(conf_w) / total,
        "acc_pixel": np.trace(conf_p) / total,
        "conf_wavelet": conf_w,
        "conf_pixel": conf_p,
        "features": np.array(feats),
        "labels": np.array(labels),
        "protos": np.array(protos),
    }
