"""
tea_core.py
2枚の写真（例：類似したペットボトル茶）のウェーブレット類似度（UIに依存しない計算部分）

類似度の指標
  (1) エネルギー特徴の類似度  S_E ∈ (0,1]
        各チャンネル・各サブバンドの対数エネルギー log E_{j,d} の差の二乗平均平方根 δ を求め
        S_E = exp(-mean_ch δ)。位置ずれに強い（「模様の粗さ・向き・色の強さ」の比較）。
        定常ウェーブレット変換(SWT)を使うと巡回シフトに対して厳密に不変。
  (2) 係数の相関による構造類似度  S_C ∈ [-1,1]
        同じサブバンドどうしの係数の相関係数を、エネルギーで重み付き平均。
        位置まで含めた「見た目の一致」の比較（位置合わせが重要）。
  (3) 近似成分の相関  S_A ∈ [-1,1]
        最も粗いスケールの近似係数どうしの相関。大まかな色・明暗の配置の比較。
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import wavelet_core as wc

CHANNEL_MODES = {
    "gray": ["Y"],
    "ycbcr": ["Y", "Cb", "Cr"],
    "rgb": ["R", "G", "B"],
}


# ---------------------------------------------------------------- 前処理
def prepare(pil_img: Image.Image, size=(128, 256), crop=(0, 0, 0, 0)):
    """切り抜き（上・下・左・右の割合[%]）→ リサイズ → RGB配列 (H,W,3), 値は0..1"""
    img = pil_img.convert("RGB")
    W, H = img.size
    t, b, l, r = crop
    box = (int(W * l / 100), int(H * t / 100), int(W * (1 - r / 100)), int(H * (1 - b / 100)))
    img = img.crop(box).resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=float) / 255.0


NORMALIZE = {"y": "輝度(Y)のみ標準化", "all": "全チャンネル標準化", "none": "補正なし"}


def split_channels(rgb: np.ndarray, mode="ycbcr", normalize="y"):
    """
    チャンネル分解と照明補正。
      normalize="y"   : 第1チャンネル(Y)だけ平均0・標準偏差1に（明るさの差を補正しつつ色の強さは残す）
      normalize="all" : 全チャンネルを標準化（色の強さの情報も消える点に注意）
      normalize="none": 補正なし
    """
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if mode == "gray":
        chs = [0.299 * R + 0.587 * G + 0.114 * B]
    elif mode == "ycbcr":
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        chs = [Y, 0.564 * (B - Y), 0.713 * (R - Y)]
    else:
        chs = [R, G, B]
    std = lambda c: (c - c.mean()) / (c.std() + 1e-8)
    if normalize == "all":
        chs = [std(c) for c in chs]
    elif normalize == "y":
        chs = [std(chs[0])] + list(chs[1:])
    return chs


def phase_align(ref: np.ndarray, mov: np.ndarray):
    """位相限定相関法で mov を ref に平行移動で合わせる（周期的シフト）。戻り値: (合わせた画像, (dy,dx))"""
    g = lambda a: a.mean(axis=2) if a.ndim == 3 else a
    F1, F2 = np.fft.fft2(g(ref)), np.fft.fft2(g(mov))
    R = F1 * np.conj(F2)
    r = np.real(np.fft.ifft2(R / (np.abs(R) + 1e-12)))
    dy, dx = np.unravel_index(np.argmax(r), r.shape)
    H, W = r.shape
    dy = dy - H if dy > H // 2 else dy
    dx = dx - W if dx > W // 2 else dx
    return np.roll(mov, (dy, dx), axis=(0, 1)), (int(dy), int(dx))


# ---------------------------------------------------------------- 類似度
def _corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 1e-12 else 0.0


def compare(rgb1, rgb2, wavelet="haar", level=3, mode="ycbcr", normalize="y", transform="swt"):
    """2枚のRGB画像の各種類似度と、サブバンドごとの内訳を返す"""
    ch_names = CHANNEL_MODES[mode]
    C1 = split_channels(rgb1, mode, normalize)
    C2 = split_channels(rgb2, mode, normalize)
    rows = []          # サブバンドごとの内訳
    delta_list, approx_list = [], []
    num, den = 0.0, 0.0
    coeffs_all = []
    for name, x1, x2 in zip(ch_names, C1, C2):
        k1 = wc.decompose(x1, wavelet, level, transform)
        k2 = wc.decompose(x2, wavelet, level, transform)
        coeffs_all.append((k1, k2))
        E1, E2 = wc.band_energies(k1), wc.band_energies(k2)
        keys = list(E1)
        l1 = np.log(np.array([E1[k] for k in keys]) + 1e-8)
        l2 = np.log(np.array([E2[k] for k in keys]) + 1e-8)
        delta_list.append(float(np.sqrt(np.mean((l1 - l2) ** 2))))
        approx_list.append(_corr(k1[0], k2[0]))
        J = len(k1) - 1
        for i, (t1, t2) in enumerate(zip(k1[1:], k2[1:])):
            j = J - i
            for d, c1, c2 in zip(wc.DIRS, t1, t2):
                rho = _corr(c1, c2)
                w = float(np.sum(c1 ** 2) + np.sum(c2 ** 2))
                num += w * rho
                den += w
                rows.append({"チャンネル": name, "レベル": j, "方向": d, "相関": rho,
                             "E(画像1)": E1[(j, d)], "E(画像2)": E2[(j, d)]})
    S_E = float(np.exp(-np.mean(delta_list)))
    S_C = num / den if den > 0 else 0.0
    S_A = float(np.mean(approx_list))
    return {"S_E": S_E, "S_C": S_C, "S_A": S_A, "rows": rows,
            "per_channel_E": dict(zip(ch_names, [float(np.exp(-d)) for d in delta_list])),
            "per_channel_A": dict(zip(ch_names, approx_list)),
            "coeffs": coeffs_all, "channels": ch_names}


def total_score(res, wE=1.0, wC=1.0, wA=1.0):
    """0〜100 の総合スコア（相関系は負の値を0に切り詰める）"""
    parts = [(wE, res["S_E"]), (wC, max(res["S_C"], 0.0)), (wA, max(res["S_A"], 0.0))]
    W = sum(w for w, _ in parts)
    return 100.0 * sum(w * s for w, s in parts) / W if W > 0 else 0.0


def difference_map(res, level_j: int, shape):
    """レベル j の詳細係数の振幅差 Σ_{ch,d} (|c1|-|c2|)^2 を画像サイズに拡大した「違いの地図」"""
    acc = None
    for k1, k2 in res["coeffs"]:
        J = len(k1) - 1
        t1, t2 = k1[1 + J - level_j], k2[1 + J - level_j]
        m = sum((np.abs(a) - np.abs(b)) ** 2 for a, b in zip(t1, t2))
        acc = m if acc is None else acc + m
    img = Image.fromarray(wc.normalize01(acc).astype(np.float32), mode="F")
    return np.asarray(img.resize((shape[1], shape[0]), Image.BILINEAR))


# ---------------------------------------------------------------- デモ用の合成ボトル
def demo_bottle(label_rgb=(40, 120, 60), pattern="leaf", offset=(0, 0), size=(256, 512)):
    """写真が無くても試せるよう、ペットボトル風の画像を描く"""
    W, H = size
    ox, oy = offset
    img = Image.new("RGB", size, (235, 235, 230))
    d = ImageDraw.Draw(img)
    # ボトル本体・肩・キャップ
    d.rounded_rectangle((60 + ox, 130 + oy, 196 + ox, 490 + oy), radius=28, fill=(205, 215, 190))
    d.polygon([(60 + ox, 150 + oy), (100 + ox, 70 + oy), (156 + ox, 70 + oy), (196 + ox, 150 + oy)],
              fill=(205, 215, 190))
    d.rectangle((102 + ox, 30 + oy, 154 + ox, 72 + oy), fill=(30, 90, 50))
    # ラベル
    d.rectangle((60 + ox, 230 + oy, 196 + ox, 400 + oy), fill=label_rgb)
    d.rectangle((60 + ox, 250 + oy, 196 + ox, 262 + oy), fill=(230, 220, 170))
    for k in range(4):  # 文字列のような模様（縦書き）
        x = 150 + ox - 22 * k
        for y in range(280, 380, 14):
            d.rectangle((x, y + oy, x + 12, y + 9 + oy), fill=(250, 250, 245))
    if pattern == "leaf":
        d.ellipse((72 + ox, 300 + oy, 110 + ox, 360 + oy), fill=(150, 200, 110))
    else:
        for k in range(5):
            d.line((66 + ox, 290 + oy + 15 * k, 112 + ox, 300 + oy + 15 * k), fill=(200, 160, 90), width=4)
    return img.filter(ImageFilter.GaussianBlur(1.0))
