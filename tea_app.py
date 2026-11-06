"""
tea_app.py  ―  ウェーブレットによる2枚の写真（類似したペットボトル茶など）の類似度判定
起動:  streamlit run tea_app.py
必要:  pip install streamlit PyWavelets numpy pandas matplotlib pillow
同じフォルダに wavelet_core.py, tea_core.py を置くこと
"""
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import tea_core as tc
import wavelet_core as wc

st.set_page_config(page_title="ウェーブレット類似度判定", layout="wide")
st.title("ウェーブレットによる2枚の写真の類似度判定")
if not wc.HAS_PYWT:
    st.warning("PyWavelets が見つからないため Haar のみ使用します（`pip install PyWavelets`）。")

SIZE = (128, 256)   # (幅, 高さ)  縦長のボトル向け。2の冪にしておくと SWT が使いやすい

DEMOS = {
    "緑ラベル・葉の模様": dict(label_rgb=(40, 120, 60), pattern="leaf"),
    "緑ラベル・葉の模様（位置ずれ）": dict(label_rgb=(40, 120, 60), pattern="leaf", offset=(12, 20)),
    "緑ラベル・線の模様": dict(label_rgb=(40, 120, 60), pattern="stripe"),
    "茶ラベル・葉の模様": dict(label_rgb=(130, 90, 40), pattern="leaf"),
    "茶ラベル・線の模様（位置ずれ）": dict(label_rgb=(130, 90, 40), pattern="stripe", offset=(12, 20)),
}

# ================================================================ 入力
with st.sidebar:
    st.header("① 写真")
    source = st.radio("入力", ["デモ画像", "写真をアップロード"])
    if source == "デモ画像":
        d1 = st.selectbox("画像1", list(DEMOS), index=0)
        d2 = st.selectbox("画像2", list(DEMOS), index=2)
        pil1, pil2 = tc.demo_bottle(**DEMOS[d1]), tc.demo_bottle(**DEMOS[d2])
    else:
        u1 = st.file_uploader("写真1", type=["png", "jpg", "jpeg", "webp"])
        u2 = st.file_uploader("写真2", type=["png", "jpg", "jpeg", "webp"])
        if u1 is None or u2 is None:
            st.info("比較する2枚の写真をアップロードしてください。"
                    "同じ背景・同じ距離・正面から撮ると比較しやすくなります。")
            st.stop()
        pil1 = Image.open(io.BytesIO(u1.getvalue()))
        pil2 = Image.open(io.BytesIO(u2.getvalue()))

    st.subheader("切り抜き（両方に同じ割合で適用）[%]")
    ct = st.slider("上", 0, 45, 0)
    cb = st.slider("下", 0, 45, 0)
    cl = st.slider("左", 0, 45, 0)
    cr = st.slider("右", 0, 45, 0)

    st.header("② 前処理")
    mode = st.radio("色の扱い", list(tc.CHANNEL_MODES), index=1,
                    format_func={"gray": "グレースケール", "ycbcr": "YCbCr（輝度+色差）",
                                 "rgb": "RGB"}.get)
    normalize = st.radio("照明補正", list(tc.NORMALIZE), format_func=tc.NORMALIZE.get)
    align = st.checkbox("位置合わせ（位相限定相関）", value=True)

    st.header("③ ウェーブレット")
    transform = st.radio("変換", ["swt", "dwt"],
                         format_func={"swt": "定常WT（SWT, シフト不変）",
                                      "dwt": "離散WT（DWT, 間引きあり）"}.get)
    wavelet = st.selectbox("ウェーブレット", wc.available_wavelets())
    level = st.slider("分解レベル J", 1, 5, 3)

    st.header("④ 総合スコアの重み")
    wE = st.slider("S_E（エネルギー特徴）", 0.0, 1.0, 1.0, 0.1)
    wC = st.slider("S_C（係数の相関）", 0.0, 1.0, 1.0, 0.1)
    wA = st.slider("S_A（近似成分＝色・明暗の配置）", 0.0, 1.0, 1.0, 0.1)
    thr = st.slider("「同じ商品」と判定する閾値", 0, 100, 85)

crop = (ct, cb, cl, cr)
rgb1 = tc.prepare(pil1, SIZE, crop)
rgb2_raw = tc.prepare(pil2, SIZE, crop)
if align:
    rgb2, shift = tc.phase_align(rgb1, rgb2_raw)
else:
    rgb2, shift = rgb2_raw, (0, 0)

res = tc.compare(rgb1, rgb2, wavelet, level, mode, normalize, transform)
score = tc.total_score(res, wE, wC, wA)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["類似度", "サブバンドの内訳", "違いの地図", "頑健性の実験", "数理メモ"])

# ---------------------------------------------------------------- 1. 類似度
with tab1:
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.image(rgb1, caption="画像1（前処理後）", width=220)
    c2.image(np.clip(rgb2, 0, 1), caption=f"画像2（位置合わせ後, ずれ dy={shift[0]}, dx={shift[1]}）",
             width=220)
    with c3:
        st.metric("総合類似度スコア", f"{score:.1f} / 100")
        if score >= thr:
            st.success(f"閾値 {thr} 以上 → 同じ商品の可能性が高い")
        else:
            st.error(f"閾値 {thr} 未満 → 別の商品の可能性が高い")
        m1, m2, m3 = st.columns(3)
        m1.metric("S_E エネルギー", f"{res['S_E']:.3f}")
        m2.metric("S_C 係数相関", f"{res['S_C']:.3f}")
        m3.metric("S_A 近似成分", f"{res['S_A']:.3f}")
        st.caption("S_E: 模様の細かさ・向き・色の強さ（位置に依存しない）　"
                   "S_C: 同じ位置に同じ模様があるか　S_A: 大まかな色・明暗の配置")
        st.markdown("**チャンネル別**")
        st.dataframe(pd.DataFrame({"S_E": res["per_channel_E"], "S_A": res["per_channel_A"]})
                     .style.format("{:.3f}"))

# ---------------------------------------------------------------- 2. 内訳
rows = pd.DataFrame(res["rows"])
with tab2:
    st.subheader("サブバンドごとの係数の相関")
    st.caption("値が小さい（色が薄い）ところが、2枚で違いの大きいスケール・方向です。")
    for ch in res["channels"]:
        t = rows[rows["チャンネル"] == ch].pivot(index="レベル", columns="方向", values="相関")
        t = t[list(wc.DIRS)]
        st.markdown(f"**{ch}**")
        st.dataframe(t.style.format("{:.3f}").background_gradient(cmap="Greens", vmin=-1, vmax=1))
    st.subheader("対数エネルギー log E の比較")
    ch_sel = st.selectbox("チャンネル", res["channels"])
    sub = rows[rows["チャンネル"] == ch_sel].copy()
    sub["band"] = "L" + sub["レベル"].astype(str) + "-" + sub["方向"]
    sub = sub.sort_values(["レベル", "方向"]).set_index("band")
    st.bar_chart(np.log(sub[["E(画像1)", "E(画像2)"]] + 1e-8), stack=False)

# ---------------------------------------------------------------- 3. 違いの地図
with tab3:
    st.subheader("どこが違うか（詳細係数の振幅差）")
    j = st.slider("見るレベル（1 が最も細かい）", 1, level, 1)
    alpha = st.slider("重ねる濃さ", 0.0, 1.0, 0.6, 0.05)
    dm = tc.difference_map(res, j, rgb1.shape[:2])
    fig, axes = plt.subplots(1, 3, figsize=(9, 6))
    axes[0].imshow(rgb1); axes[0].set_title("image 1")
    axes[1].imshow(np.clip(rgb2, 0, 1)); axes[1].set_title("image 2 (aligned)")
    axes[2].imshow(rgb1); axes[2].imshow(dm, cmap="inferno", alpha=alpha)
    axes[2].set_title(f"difference, level {j}")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("ラベル部分だけに差が出ているなら、切り抜きでラベルに絞ると判定が鋭くなります。")

# ---------------------------------------------------------------- 4. 頑健性
with tab4:
    st.subheader("同じ写真を劣化させたとき、類似度はどう変わるか")
    st.caption("画像1を平行移動・明るさ変化・雑音で劣化させたものと、元の画像1を比べます"
               "（本来は「同一」と判定されるべき組）。")
    kind = st.radio("劣化の種類", ["平行移動", "明るさ", "雑音"], horizontal=True)
    use_align = st.checkbox("この実験でも位置合わせを行う", value=False)

    @st.cache_data(show_spinner="計算中…")
    def robustness(rgb1, kind, use_align, wavelet, level, mode, normalize):
        rng = np.random.default_rng(0)
        if kind == "平行移動":
            xs = list(range(0, 33, 2))
            make = lambda v: np.roll(rgb1, (v, v // 2), axis=(0, 1))
        elif kind == "明るさ":
            xs = [round(v, 2) for v in np.linspace(0.4, 1.6, 13)]
            make = lambda v: np.clip(rgb1 * v, 0, 1)
        else:
            xs = [round(v, 2) for v in np.linspace(0, 0.3, 13)]
            make = lambda v: rgb1 + v * rng.standard_normal(rgb1.shape)
        out = []
        for v in xs:
            img = make(v)
            if use_align:
                img = tc.phase_align(rgb1, img)[0]
            for tr in ("dwt", "swt"):
                r = tc.compare(rgb1, img, wavelet, level, mode, normalize, tr)
                out.append({"x": v, "変換": tr.upper(), "S_E": r["S_E"], "S_C": r["S_C"],
                            "S_A": r["S_A"]})
        return pd.DataFrame(out)

    df = robustness(rgb1, kind, use_align, wavelet, level, mode, normalize)
    xlabel = {"平行移動": "ずれ [px]", "明るさ": "明るさの倍率", "雑音": "雑音の標準偏差"}[kind]
    for key in ("S_E", "S_C", "S_A"):
        st.markdown(f"**{key}**　（横軸: {xlabel}）")
        st.line_chart(df.pivot(index="x", columns="変換", values=key))
    st.caption("平行移動では DWT の S_E が揺れ、SWT の S_E は一定。S_C・S_A は位置合わせなしでは急落する。")

# ---------------------------------------------------------------- 5. 数理メモ
with tab5:
    st.subheader("類似度の定義")
    st.markdown("画像をチャンネル $\\ell$（Y, Cb, Cr など）に分け、各チャンネルを $J$ 段に分解して"
                "詳細係数 $c^{(\\ell)}_{j,d}$（$d\\in\\{H,V,D\\}$）と近似係数 $a^{(\\ell)}_J$ を得る。")
    st.markdown("**(1) エネルギー特徴の類似度**")
    st.latex(r"E^{(\ell)}_{j,d}=\frac{1}{N}\sum_{m,n}\bigl|c^{(\ell)}_{j,d}[m,n]\bigr|^2,\quad"
             r"\delta_\ell=\Bigl(\frac{1}{3J}\sum_{j,d}\bigl(\log E^{(\ell)}_{j,d}-\log\tilde E^{(\ell)}_{j,d}\bigr)^2\Bigr)^{1/2},"
             r"\quad S_E=\exp\Bigl(-\frac{1}{L}\sum_\ell\delta_\ell\Bigr)")
    st.markdown("**(2) 係数の相関（エネルギー重み付き）**")
    st.latex(r"S_C=\frac{\sum_{\ell,j,d} w_{\ell,j,d}\,\rho\bigl(c^{(\ell)}_{j,d},\tilde c^{(\ell)}_{j,d}\bigr)}"
             r"{\sum_{\ell,j,d} w_{\ell,j,d}},\qquad w=\|c\|^2+\|\tilde c\|^2")
    st.markdown("**(3) 近似成分の相関**")
    st.latex(r"S_A=\frac1L\sum_\ell\rho\bigl(a^{(\ell)}_J,\tilde a^{(\ell)}_J\bigr)")
    st.markdown("**総合スコア**")
    st.latex(r"S=100\cdot\frac{w_E S_E+w_C\max(S_C,0)+w_A\max(S_A,0)}{w_E+w_C+w_A}")
    st.markdown(
        "- **DWT と SWT**：DWT は各段で 2 つに 1 つ間引くため、$2^j$ の倍数でない平行移動で"
        "係数が変わり、エネルギーも揺れる。SWT（à trous 法）は間引かずフィルタを $2^{j-1}$ 間隔に"
        "広げるので、巡回シフト $\\tau$ に対して $c_{j,d}[\\cdot-\\tau]$ となり $E_{j,d}$ は厳密に不変。\n"
        "- **位置合わせ**：位相限定相関 $r=\\mathcal F^{-1}\\!\\left[\\frac{F_1\\overline{F_2}}"
        "{|F_1\\overline{F_2}|}\\right]$ のピーク位置が平行移動量を与える（フーリエ変換のシフト定理）。\n"
        "- **照明補正**：輝度 Y を平均0・分散1にすると明るさ・コントラストの差は消える。"
        "色差まで標準化すると「色の濃さ」の違いも消えるので、茶葉の色の違いを見たいときは避ける。\n"
        "- **グレースケールの限界**：緑と茶のラベルは輝度がほぼ同じになることがあり、"
        "色差チャンネルを使わないと区別できない（デモ画像で確認できる）。"
    )
