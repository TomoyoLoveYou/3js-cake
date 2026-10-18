# -*- coding: utf-8 -*-
"""
render_to_pixel.py —— 把 Three.js 渲染图转成像素图（轮廓优先）

为什么不直接用「缩小 + 最近邻 / 众数投票」：
2~3px 的描边缩小到 48 格后只剩 20% 左右的覆盖率，众数投票会把它判给背景色，
于是轮廓碎成虚线、相邻同色材质（奶油 / 盘子）塌成一块白，颜色信息一起丢掉。

本脚本的三步：
  1. 色板取自渲染图本身 —— 对真实颜色做 ΔE 贪心聚类。Toon 渲染的明暗是离散档位，
     聚出来的就是美术实际用的那几簇色，不做主观调色。
  2. 描边单独抽成 mask —— 反向外扩描边在渲染图里本来就是连续的一条线，
     直接采样它比事后猜边缘准确；再骨架化成 1px 中心线，避免粗描边糊成一坨。
  3. 每格按面积占比决定：骨架线穿过 -> 描边；否则取占比最大的颜色。
     最后只做最小限度的连通性修补，不添加原图没有的东西。

用法：
    python tools/render_to_pixel.py --grid 48 64 96
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "video-assets"
SRC_DEFAULT = ASSETS / "render-reset-view.png"


# ----------------------------------------------------------------- 色彩空间
_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
_WP = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = (c @ _M.T) / _WP
    d = 6.0 / 29.0
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4.0 / 29.0)
    return np.stack([116.0 * f[..., 1] - 16.0,
                     500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])], axis=-1)


def rgb2hex(c) -> str:
    return "%02X%02X%02X" % tuple(int(round(float(v))) for v in c)


# ----------------------------------------------------------------- 色板
def build_palette(rgb, eye, max_colors=32, merge_de=5.0, min_share=0.0012):
    """对真实出现的颜色做 ΔE 贪心聚类，簇心回吸到真实像素（避免平均色发灰）"""
    px = rgb[eye].reshape(-1, 3)
    packed = (px[:, 0].astype(np.int64) << 16) | (px[:, 1].astype(np.int64) << 8) | px[:, 2]
    uniq, counts = np.unique(packed, return_counts=True)
    cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.uint8)
    lab = srgb_to_lab(cols.astype(np.float32))
    order = np.argsort(-counts)

    centers, weights, members = [], [], []
    for idx in order:
        p, w = lab[idx], float(counts[idx])
        if centers:
            d = np.linalg.norm(np.stack(centers) - p, axis=1)
            j = int(d.argmin())
            if d[j] < merge_de or len(centers) >= max_colors:
                tot = weights[j] + w
                centers[j] = (centers[j] * weights[j] + p * w) / tot
                weights[j] = tot
                members[j].append(int(idx))
                continue
        centers.append(p.copy()); weights.append(w); members.append([int(idx)])

    total = counts.sum()
    keep = sorted([i for i, w in enumerate(weights) if w / total >= min_share],
                  key=lambda i: -weights[i])
    out_rgb, out_cnt = [], []
    for i in keep:
        m = members[i]
        out_rgb.append(cols[m[int(np.argmax(counts[m]))]])
        out_cnt.append(weights[i])
    colors = np.array(out_rgb, dtype=np.uint8)
    return colors, np.array(out_cnt), srgb_to_lab(colors.astype(np.float32))


def rgb_to_labels(lab, pal_lab, chunk=200_000):
    h, w, _ = lab.shape
    flat = lab.reshape(-1, 3)
    out = np.empty(flat.shape[0], dtype=np.int16)
    step = max(1, chunk // max(1, pal_lab.shape[0]))
    for i in range(0, flat.shape[0], step):
        d = ((flat[i:i + step, None, :] - pal_lab[None, :, :]) ** 2).sum(-1)
        out[i:i + step] = d.argmin(1)
    return out.reshape(h, w)


def auto_name(c) -> str:
    import colorsys
    r, g, b = [float(v) / 255 for v in c]
    mx, mn = max(r, g, b), min(r, g, b)
    v, s = mx, 0.0 if mx == 0 else (mx - mn) / mx
    if s < 0.12:
        return "白" if v > 0.92 else "浅灰" if v > 0.7 else "灰" if v > 0.45 else "深灰"
    h, _, _ = colorsys.rgb_to_hsv(r, g, b)
    deg = h * 360
    names = [(15, "红"), (40, "橙"), (65, "黄"), (100, "黄绿"), (150, "绿"),
             (195, "青"), (255, "蓝"), (295, "紫"), (335, "品红"), (361, "红")]
    hue = next(n for t, n in names if deg < t)
    tone = "深" if v < 0.6 else ("亮" if v > 0.97 else "中")
    return f"{hue}{tone}"


# ----------------------------------------------------------------- 形态学
def trim_white(img, thr=250):
    a = np.asarray(img.convert("RGB"))
    nw = ~((a > thr).all(axis=2))
    rs, cs = np.where(nw.any(1))[0], np.where(nw.any(0))[0]
    if not len(rs) or not len(cs):
        return img.convert("RGB")
    return img.convert("RGB").crop((int(cs.min()), int(rs.min()),
                                    int(cs.max()) + 1, int(rs.max()) + 1))


def largest_component_filled(mask):
    """取最大 4 连通域并填内部空洞（去掉背景噪点 / 彩纸碎屑）"""
    h, w = mask.shape
    lab = np.full((h, w), -1, dtype=np.int32)
    cur = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if lab[y0, x0] != -1:
            continue
        q = deque([(y0, x0)]); lab[y0, x0] = cur
        while q:
            y, x = q.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and lab[yy, xx] == -1:
                    lab[yy, xx] = cur; q.append((yy, xx))
        cur += 1
    if cur == 0:
        return mask
    sizes = np.bincount(lab[lab >= 0])
    subj = lab == int(sizes.argmax())

    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if not subj[y, x] and not outside[y, x]:
                outside[y, x] = True; q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not subj[y, x] and not outside[y, x]:
                outside[y, x] = True; q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and not subj[yy, xx] and not outside[yy, xx]:
                outside[yy, xx] = True; q.append((yy, xx))
    return ~outside


def label_components(g, value):
    """8 连通分量，返回 list[list[(i, j)]]"""
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for i in range(h):
        for j in range(w):
            if g[i, j] != value or seen[i, j]:
                continue
            comp, q = [], deque([(i, j)])
            seen[i, j] = True
            while q:
                y, x = q.popleft()
                comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if not (dy or dx):
                            continue
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and g[yy, xx] == value and not seen[yy, xx]:
                            seen[yy, xx] = True
                            q.append((yy, xx))
            comps.append(comp)
    return comps


def zhang_suen(mask, max_iter=64):
    """Zhang-Suen 细化：粗描边 -> 1px 中心线"""
    img = mask.astype(np.uint8).copy()
    pad = np.pad(img, 1)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = pad
            P2, P3, P4 = p[0:-2, 1:-1], p[0:-2, 2:], p[1:-1, 2:]
            P5, P6, P7 = p[2:, 2:], p[2:, 1:-1], p[2:, 0:-2]
            P8, P9 = p[1:-1, 0:-2], p[0:-2, 0:-2]
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = sum(((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8) for k in range(8))
            if step == 0:
                c1 = (P2 * P4 * P6) == 0
                c2 = (P4 * P6 * P8) == 0
            else:
                c1 = (P2 * P4 * P8) == 0
                c2 = (P2 * P6 * P8) == 0
            rm = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
            if rm.any():
                img[rm] = 0
                pad[1:-1, 1:-1] = img
                changed = True
        if not changed:
            break
    return img.astype(bool)


# ----------------------------------------------------------------- 主流程
def convert(src: Path, out_dir: Path, grid: int, scale: int, margin: float,
            min_share: float, bg_flat: bool, dump: bool = False):
    rgb = np.asarray(trim_white(Image.open(src)))
    h, w, _ = rgb.shape
    lab = srgb_to_lab(rgb.astype(np.float32))

    # --- 背景：四角中位数 + ΔE 容忍（背景是径向渐变，整体归一）
    corners = np.concatenate([rgb[:6, :6].reshape(-1, 3), rgb[:6, -6:].reshape(-1, 3),
                              rgb[-6:, :6].reshape(-1, 3), rgb[-6:, -6:].reshape(-1, 3)])
    bg_rgb = np.median(corners, axis=0)
    bg_lab = srgb_to_lab(bg_rgb[None, :].astype(np.float32))[0]
    subj = largest_component_filled(np.linalg.norm(lab - bg_lab, axis=-1) >= 14.0)

    # --- 方形裁切（按主体外接框，留白 margin）
    ys, xs = np.nonzero(subj)
    side = int(round(max(ys.max() + 1 - ys.min(), xs.max() + 1 - xs.min()) * (1 + 2 * margin)))
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    sx0, sy0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    canvas = np.empty((side, side, 3), np.uint8)
    canvas[:, :] = bg_rgb.astype(np.uint8)
    ox0, oy0 = max(0, sx0), max(0, sy0)
    ox1, oy1 = min(w, sx0 + side), min(h, sy0 + side)
    canvas[oy0 - sy0:oy1 - sy0, ox0 - sx0:ox1 - sx0] = rgb[oy0:oy1, ox0:ox1]
    crop_subj = np.zeros((side, side), bool)
    crop_subj[oy0 - sy0:oy1 - sy0, ox0 - sx0:ox1 - sx0] = subj[oy0:oy1, ox0:ox1]

    # --- 色板 + 逐像素归类
    colors, counts, pal_lab = build_palette(canvas, crop_subj, min_share=min_share * 0.3)
    labels = rgb_to_labels(srgb_to_lab(canvas.astype(np.float32)), pal_lab)
    labels[~crop_subj] = -1

    # --- 描边色：主体内占比最大的「低彩度中灰」（反向外扩描边）
    outline_idx, best = -1, -1.0
    for i, l in enumerate(pal_lab):
        chroma = float(np.hypot(l[1], l[2]))
        if 32 < l[0] < 64 and chroma < 18 and counts[i] > best:
            best, outline_idx = counts[i], i

    outline_mask = (labels == outline_idx) if outline_idx >= 0 else np.zeros_like(labels, bool)
    skel = zhang_suen(outline_mask).ravel()

    # --- 每格统计
    n_lab = len(colors)
    cell = side / grid
    ci = (np.arange(side) * grid // side)
    cell_id = (ci[:, None] * grid + ci[None, :]).ravel()
    lab_flat = labels.ravel()
    ok = lab_flat >= 0
    all_counts = np.bincount(cell_id[ok] * n_lab + lab_flat[ok],
                             minlength=grid * grid * n_lab).reshape(grid * grid, n_lab)
    skel_px = np.bincount(cell_id[skel], minlength=grid * grid)[:grid * grid].astype(np.float64)
    total_px = np.full(grid * grid, float((side / grid) ** 2))

    body = all_counts.copy()
    if outline_idx >= 0:
        body[:, outline_idx] = 0
    dom = body.argmax(1)
    body_share = body.sum(1) / total_px
    bg_share = np.clip(1.0 - body_share - (all_counts[:, outline_idx] / total_px
                                           if outline_idx >= 0 else 0.0), 0.0, 1.0)

    OUT, BG = -2, -1

    def n8(i, j):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    ii, jj = i + di, j + dj
                    if 0 <= ii < grid and 0 <= jj < grid:
                        yield ii, jj

    def n4(i, j):
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < grid and 0 <= jj < grid:
                yield ii, jj

    # 描边带往往比一个格子宽（掠射角处尤其厚）。要让轮廓只占 1 格，
    # 先给「没有自身主体色」的格子找一个最近的可靠主体色，用它回填。
    trust = (body_share >= 0.30) & (skel_px == 0)
    near = dom.copy()
    near_ok = trust.copy()
    frontier = [divmod(int(k), grid) for k in np.nonzero(trust)[0]]
    for _ in range(4):
        nxt = []
        for i, j in frontier:
            for ii, jj in n8(i, j):
                if not near_ok[ii * grid + jj]:
                    near[ii * grid + jj] = near[i * grid + j]
                    near_ok[ii * grid + jj] = True
                    nxt.append((ii, jj))
        frontier = nxt
        if not frontier:
            break

    g = np.full((grid, grid), BG, np.int32)
    for k in range(grid * grid):
        i, j = divmod(k, grid)
        # 骨架线真正穿过这一格 -> 描边（1 格宽）
        if skel_px[k] >= 1 and bg_share[k] < 0.9:
            g[i, j] = OUT
        elif bg_share[k] > 0.5:
            g[i, j] = BG
        elif body_share[k] >= 0.30:
            g[i, j] = int(dom[k])
        elif near_ok[k]:
            g[i, j] = int(near[k])          # 粗描边带的剩余格子 -> 就近主体色
        else:
            g[i, j] = BG

    # --- 只做一件事：删掉不挨任何线的孤立描边点（细化毛刺 / 压缩噪点）
    dropped = 0
    for _ in range(3):
        changed = 0
        for i in range(grid):
            for j in range(grid):
                if g[i, j] != OUT:
                    continue
                if any(g[ii, jj] == OUT for ii, jj in n8(i, j)):
                    continue
                # 孤立描边点：四周既没别的线，也没形成环 -> 回退成主体色
                if body_share[i * grid + j] > 0.4:
                    g[i, j] = int(dom[i * grid + j])
                elif sum(1 for ii, jj in n4(i, j) if g[ii, jj] == BG) >= 2:
                    g[i, j] = BG
                else:
                    g[i, j] = int(dom[i * grid + j])
                changed += 1
                dropped += 1
        if not changed:
            break
    print(f"[{grid}] 去掉孤立描边点 {dropped} 个")

    # 去掉孤立的中性色噪点（阴影/AA 抖出来的白点灰点）；
    # 有彩度的单点保留 —— 那些多半是糖针，属有效细节。
    def chroma_of(k):
        l = pal_lab[k]
        return float(np.hypot(l[1], l[2]))

    cleaned = 0
    for _ in range(2):
        changed = 0
        for i in range(grid):
            for j in range(grid):
                v = g[i, j]
                if v in (OUT, BG) or chroma_of(v) >= 8:
                    continue
                nb = [g[ii, jj] for ii, jj in n8(i, j)]
                if nb and len(set(nb)) == 1 and nb[0] not in (OUT, BG) and nb[0] != v:
                    g[i, j] = nb[0]
                    changed += 1
                    cleaned += 1
        if not changed:
            break
    print(f"[{grid}] 清理中性噪点 {cleaned} 个")

    # 抗锯齿抖出来的碎点：这一格主色占比很低，且 8 邻域几乎同色 -> 并进去
    dom_cov = body.max(1) / total_px
    absorbed = 0
    for _ in range(2):
        changed = 0
        for i in range(grid):
            for j in range(grid):
                v = g[i, j]
                if v in (OUT, BG) or dom_cov[i * grid + j] >= 0.30:
                    continue
                nb = [g[ii, jj] for ii, jj in n8(i, j) if g[ii, jj] != v]
                if len(nb) >= 7 and len(set(nb)) == 1 and nb[0] not in (OUT, BG):
                    g[i, j] = nb[0]
                    changed += 1
                    absorbed += 1
        if not changed:
            break
    print(f"[{grid}] 吸收碎点 {absorbed} 个")

    # --- 出图
    pal_out = np.vstack([colors, bg_rgb[None, :].astype(np.uint8)])
    idx = np.where(g == BG, n_lab, g)
    if outline_idx >= 0:
        idx = np.where(g == OUT, outline_idx, idx)
    else:
        idx = np.where(g == OUT, int(np.argmin(pal_lab[:, 0])), idx)
    tab = np.array(pal_out, dtype=np.uint8)
    pix = tab[np.clip(idx, 0, len(tab) - 1)].astype(np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    small = Image.fromarray(pix, "RGB")
    small.save(out_dir / f"cake@{grid}.png")
    small.resize((grid * scale, grid * scale), Image.NEAREST).save(out_dir / f"cake@{grid}x{scale}.png")

    used = {}
    for v in idx.ravel():
        used[int(v)] = used.get(int(v), 0) + 1
    entries = []
    for v, n in sorted(used.items(), key=lambda t: -t[1]):
        c = tab[v]
        if v == n_lab:
            role = "背景"
        elif v == outline_idx:
            role = "描边"
        else:
            role = "主体"
        entries.append({"name": auto_name(c), "hex": rgb2hex(c), "count": n, "role": role})

    with open(out_dir / f"cake@{grid}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"source": src.name, "grid": grid, "scale": scale,
                   "crop": {"x": sx0, "y": sy0, "side": side},
                   "palette": entries,
                   "cells": [[int(idx[i, j]) for j in range(grid)] for i in range(grid)]},
                  f, ensure_ascii=False, indent=1)

    print(f"[{grid}] crop {side}px | {len(entries)} 色 | 网格 {grid}×{grid}")
    for e in entries:
        print(f"      {e['role']:<2} #{e['hex']}  {e['name']:<5} x{e['count']}")

    if dump:
        Image.fromarray(np.where(crop_subj[..., None], canvas, 0).astype(np.uint8)).save(
            out_dir / f"_crop@{grid}.png")
        Image.fromarray((outline_mask * 255).astype(np.uint8)).save(out_dir / f"_outline@{grid}.png")
        Image.fromarray((skel.reshape(side, side) * 255).astype(np.uint8)).save(
            out_dir / f"_skel@{grid}.png")
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT)
    ap.add_argument("--out", type=Path, default=ASSETS / "pixel")
    ap.add_argument("--grid", type=int, nargs="+", default=[48])
    ap.add_argument("--scale", type=int, default=16)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--min-share", type=float, default=0.004)

    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    for g in a.grid:
        convert(a.src, a.out, g, a.scale, a.margin, a.min_share, True, a.dump)


if __name__ == "__main__":
    main()
