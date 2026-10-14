# -*- coding: utf-8 -*-
"""
把 render-reset-view.png（复位视角纯净渲染）转换成：
  1. pixel-cake.png      —— 48×48 像素图（放大 16 倍，最近邻）
  2. pixel-cake@48.png   —— 48×48 原始像素图
  3. perler-cake.png     —— 拼豆图纸（豆珠网格 + 色号表）
  4. pattern.json        —— 网格数据 / 色号 / 每色数量（供视频动画使用）

用法： python make_pixel_assets.py
"""
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent / "video-assets"
SRC = ROOT / "render-reset-view.png"
GRID = 48
SCALE = 16  # 像素图放大倍数

# ---------------------------------------------------------------- 色板
# (名称, HEX) —— 按渲染图实测颜色校正：
#   MeshToonMaterial 的四个梯度档位（1.0/0.949/0.831/0.706）× 材质色；
#   高档与次档视觉接近已合并；未出现的颜色不会进色号表。
PALETTE = [
    ("奶白",      "FFFDFA"),  # 盘子 / 奶油受光面 / 白糖针
    ("奶油暗部",  "B4B1AF"),  # 奶油背光面
    ("海绵粉",    "F2A8C0"),  # 蛋糕胚主色（0.949 档）
    ("海绵粉·中", "D495A9"),  # 0.831 档
    ("海绵粉·深", "B58091"),  # 0.706 档
    ("果酱红",    "F24865"),
    ("果酱·中",  "D44059"),
    ("果酱深",    "B4364B"),
    ("草莓红",    "F23A57"),
    ("草莓·中",  "D4334A"),
    ("草莓深",    "B52B3F"),
    ("叶子绿",    "70BA46"),
    ("叶子·中",  "62A33D"),
    ("叶子深",    "538B35"),
    ("蜡烛粉",    "F288AB"),
    ("蜡烛·中",  "D47795"),
    ("蜡烛深",    "B4657F"),
    ("描边灰紫",  "8F7580"),  # 背面外扩描边（不受光）
    ("火苗橙",    "FF8A3D"),  # 火焰外层（不受光）
    ("火苗黄",    "FFE08A"),  # 火焰内芯（不受光）
    ("糖针黄",    "F2C761"),
    ("糖针蓝",    "87CDF4"),
    ("梅子·浅",  "5C4070"),  # 背景（顶部受径向高光提亮）
    ("梅子·中",  "473058"),
    ("梅子·深",  "3C2C49"),  # 背景（底部）
    ("盘影",      "31243C"),  # 盘下投影
]
PAL = np.array([tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for _, h in PALETTE], dtype=np.int32)
PAL_RGB = [tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for _, h in PALETTE]
BG_IDX = {i for i, (n, _) in enumerate(PALETTE) if n.startswith("梅子") or n == "盘影"}
SPECKLE_IDX = {i for i, (n, _) in enumerate(PALETTE) if n.startswith("糖针") or n == "奶白"}


def hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c)


# ---------------------------------------------------------------- 1. 裁切蛋糕
img = Image.open(SRC).convert("RGB")
arr = np.asarray(img).astype(np.int32)
# 无头截图可能比视口大，边缘会露出浏览器默认白边 —— 先裁掉
nonwhite = ~((arr > 250).all(axis=2))
rs = np.where(nonwhite.any(axis=1))[0]
cs = np.where(nonwhite.any(axis=0))[0]
if len(rs) and len(cs):
    arr = arr[rs.min():rs.max() + 1, cs.min():cs.max() + 1]
H, W, _ = arr.shape
print(f"[src] {W}x{H}")

bright = arr.max(axis=2)
mask = bright > 132  # 梅子背景最亮约 120，蛋糕/蜡烛/糖针都远高于此

col_c = mask.sum(axis=0)
row_c = mask.sum(axis=1)
c_thr = max(40, col_c.max() * 0.10)
r_thr = max(40, row_c.max() * 0.10)
cols = np.where(col_c > c_thr)[0]
rows = np.where(row_c > r_thr)[0]
x0, x1, y0, y1 = int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())
print(f"[bbox] x {x0}..{x1}  y {y0}..{y1}  ({x1 - x0}x{y1 - y0})")

cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
side = int(max(x1 - x0, y1 - y0) * 1.06)
sx0 = int(round(max(0, min(cx - side / 2, W - side))))
sy0 = int(round(max(0, min(cy - side / 2, H - side))))
crop = arr[sy0:sy0 + side, sx0:sx0 + side]
print(f"[crop] {side}x{side} @ ({sx0},{sy0})")

# ---------------------------------------------------------------- 2. 量化到色板 + 单元格众数
flat = crop.reshape(-1, 3).astype(np.float32)
a2 = (flat ** 2).sum(1, keepdims=True)
b2 = (PAL.astype(np.float32) ** 2).sum(1)[None, :]
dist = a2 + b2 - 2.0 * (flat @ PAL.astype(np.float32).T)
qidx = dist.argmin(1).astype(np.int32).reshape(side, side)

edges = [round(i * side / GRID) for i in range(GRID + 1)]
grid = np.zeros((GRID, GRID), dtype=np.int32)
for i in range(GRID):
    for j in range(GRID):
        block = qidx[edges[i]:edges[i + 1], edges[j]:edges[j + 1]]
        vals, counts = np.unique(block, return_counts=True)
        grid[i, j] = int(vals[counts.argmax()])


# ---------------------------------------------------------------- 3. 去噪（清掉飘进来的彩纸碎屑）
# 规则：糖针色的小簇（≤3 格）且邻接背景 ≥5 格 → 并入背景色；
#       落在奶油上的糖针邻接是奶油，不会误删。
def nbr_coords(i, j):
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ii, jj = i + di, j + dj
            if 0 <= ii < GRID and 0 <= jj < GRID:
                yield ii, jj


removed = 0
visited = set()
for i in range(GRID):
    for j in range(GRID):
        if (i, j) in visited or int(grid[i, j]) not in SPECKLE_IDX:
            continue
        color = int(grid[i, j])
        stack, seen = [(i, j)], {(i, j)}
        while stack:  # 同色小簇
            ci, cj = stack.pop()
            for ni, nj in nbr_coords(ci, cj):
                if (ni, nj) not in seen and int(grid[ni, nj]) == color:
                    seen.add((ni, nj))
                    stack.append((ni, nj))
        visited |= seen
        if len(seen) > 3:
            continue
        bg = Counter(int(grid[ni, nj]) for ci, cj in seen for ni, nj in nbr_coords(ci, cj)
                     if int(grid[ni, nj]) in BG_IDX)
        if sum(bg.values()) >= 5:
            fill = bg.most_common(1)[0][0]
            for ci, cj in seen:
                grid[ci, cj] = fill
            removed += len(seen)
print(f"[clean] absorbed {removed} confetti cells")

counts = Counter(int(v) for v in grid.ravel())
used = [(i, PALETTE[i][0], PALETTE[i][1], counts.get(i, 0)) for i in range(len(PALETTE)) if counts.get(i, 0) > 0]
used.sort(key=lambda t: -t[3])
print(f"[palette] used {len(used)} colors, total {GRID * GRID} beads")
for _, n, h, c in used:
    print(f"    {n:<8} #{h}  x{c}")

# ---------------------------------------------------------------- 4. 像素图
raw = Image.new("RGB", (GRID, GRID))
raw.putdata([PAL_RGB[int(grid[i, j])] for i in range(GRID) for j in range(GRID)])
raw.save(ROOT / "pixel-cake@48.png")
raw.resize((GRID * SCALE, GRID * SCALE), Image.NEAREST).save(ROOT / "pixel-cake.png")

# ---------------------------------------------------------------- 5. 拼豆图纸
F_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
F_REG = "C:/Windows/Fonts/msyh.ttc"
for f in (F_BOLD, F_REG):
    if not os.path.exists(f):
        raise FileNotFoundError(f"缺少中文字体: {f}")

CW, CH = 1700, 1300
PAPER = hex2rgb("FBF8F3")
INK = hex2rgb("4A4257")
SUB = hex2rgb("8A8098")
GRID_L = hex2rgb("E7E1EE")
GRID_S = hex2rgb("B3AAC2")
FRAME = hex2rgb("6E6480")
HEAD_BG = hex2rgb("F5EFF8")
HEAD_LN = hex2rgb("E0D6EA")

M = 56
G = 920
cell = G / GRID
gx, gy = M + 40, 300
board_pad_x, board_pad_y = 30, 34

chart = Image.new("RGB", (CW, CH), PAPER)
d = ImageDraw.Draw(chart)

f_title = ImageFont.truetype(F_BOLD, 44)
f_sub = ImageFont.truetype(F_REG, 22)
f_sec = ImageFont.truetype(F_BOLD, 28)
f_name = ImageFont.truetype(F_BOLD, 21)
f_hex = ImageFont.truetype(F_REG, 17)
f_cnt = ImageFont.truetype(F_BOLD, 21)
f_num = ImageFont.truetype(F_REG, 15)
f_foot = ImageFont.truetype(F_REG, 19)

# ---- 头部
d.rounded_rectangle([M, 44, CW - M, 208], radius=18, fill=HEAD_BG, outline=HEAD_LN, width=2)
d.text((M + 32, 74), "卡通草莓蛋糕 · 拼豆图纸", font=f_title, fill=INK)
d.text((M + 34, 140), "Three.js 渲染 → 48×48 像素图 → 拼豆图纸 ｜ 共 2304 颗", font=f_sub, fill=SUB)
# 头部右侧：三颗豆 + 说明
bx = CW - M - 300
for k, colr in enumerate([PAL_RGB[8], PAL_RGB[2], PAL_RGB[11]]):  # 草莓红/海绵粉/叶子绿
    cxp, cyp = bx + k * 44, 108
    d.ellipse([cxp - 17, cyp - 17, cxp + 17, cyp + 17], fill=colr, outline=shade(colr, .72), width=2)
    d.ellipse([cxp - 8, cyp - 8, cxp + 8, cyp + 8], fill=PAPER, outline=shade(colr, .85), width=1)
    d.ellipse([cxp - 3, cyp - 3, cxp + 3, cyp + 3], fill=shade(colr, .9))
d.text((bx + 152, 92), "适用 2.5mm 拼豆", font=f_name, fill=INK)
d.text((bx + 152, 122), "5mm 中豆请等比放大", font=f_hex, fill=SUB)

# ---- 图纸底板
d.rounded_rectangle([gx - board_pad_x + 6, gy - board_pad_y + 8,
                     gx + G + board_pad_x + 6, gy + G + board_pad_y + 8],
                    radius=14, fill=hex2rgb("E9E2F1"))
d.rounded_rectangle([gx - board_pad_x, gy - board_pad_y,
                     gx + G + board_pad_x, gy + G + board_pad_y],
                    radius=14, fill=(255, 255, 255), outline=hex2rgb("D9CFE4"), width=2)

# ---- 网格线
for i in range(GRID + 1):
    y = gy + i * cell
    strong = i % 5 == 0
    d.line([(gx, y), (gx + G, y)], fill=GRID_S if strong else GRID_L, width=2 if strong else 1)
for j in range(GRID + 1):
    x = gx + j * cell
    strong = j % 5 == 0
    d.line([(x, gy), (x, gy + G)], fill=GRID_S if strong else GRID_L, width=2 if strong else 1)

# ---- 坐标数字（每 5 格）
for v in range(5, GRID, 5):
    d.text((gx + v * cell, gy - board_pad_y / 2), str(v), font=f_num, fill=SUB, anchor="mm")
    d.text((gx - board_pad_x / 2, gy + v * cell), str(v), font=f_num, fill=SUB, anchor="mm")

# ---- 豆珠
br = cell * 0.40
for i in range(GRID):
    for j in range(GRID):
        c = PAL_RGB[int(grid[i, j])]
        cxp, cyp = gx + (j + .5) * cell, gy + (i + .5) * cell
        d.ellipse([cxp - br, cyp - br, cxp + br, cyp + br], fill=c, outline=shade(c, .72),
                  width=max(1, int(br * .14)))
        hr = br * .46
        d.ellipse([cxp - hr, cyp - hr, cxp + hr, cyp + hr], fill=(255, 255, 255),
                  outline=shade(c, .85), width=max(1, int(br * .08)))
        cr = br * .15
        d.ellipse([cxp - cr, cyp - cr, cxp + cr, cyp + cr], fill=shade(c, .9))
d.rounded_rectangle([gx - 2, gy - 2, gx + G + 2, gy + G + 2], radius=4, outline=FRAME, width=3)

# ---- 色号表
lx = gx + G + board_pad_x + 56
lw = CW - M - lx
ly = 300
d.text((lx, ly - 46), "色号表", font=f_sec, fill=INK)
d.text((lx + 96, ly - 40), f"{len(used)} 色 · 按数量排序", font=f_hex, fill=SUB)

row_h = min(37.0, (gy + G + board_pad_y - ly) / max(1, len(used)))
for k, (idx, name, hx, cnt) in enumerate(used):
    cy2 = ly + row_h * (k + .5)
    colr = PAL_RGB[idx]
    r = min(13.0, row_h * .36)
    d.ellipse([lx + 18 - r, cy2 - r, lx + 18 + r, cy2 + r], fill=colr, outline=shade(colr, .72), width=2)
    hr = r * .46
    d.ellipse([lx + 18 - hr, cy2 - hr, lx + 18 + hr, cy2 + hr], fill=(255, 255, 255),
              outline=shade(colr, .85), width=1)
    d.ellipse([lx + 18 - r * .15, cy2 - r * .15, lx + 18 + r * .15, cy2 + r * .15], fill=shade(colr, .9))
    d.text((lx + 44, cy2), name, font=f_name, fill=INK, anchor="lm")
    d.text((lx + 200, cy2), "#" + hx, font=f_hex, fill=SUB, anchor="lm")
    d.text((lx + lw, cy2), f"×{cnt}", font=f_cnt, fill=INK, anchor="rm")

# ---- 页脚
d.line([(M, CH - 66), (CW - M, CH - 66)], fill=HEAD_LN, width=2)
d.text((CW // 2, CH - 38),
       "色号由渲染图自动量化生成 ｜ 深色豆不足时可用相邻色号代替 ｜ 可映射 Mard / Perler / Hama 品牌色卡",
       font=f_foot, fill=SUB, anchor="mm")

chart.save(ROOT / "perler-cake.png")

# ---------------------------------------------------------------- 6. 网格数据（供视频动画）
data = {
    "source": SRC.name,
    "grid": GRID,
    "cell_px": round(cell, 3),
    "total": GRID * GRID,
    "crop": {"x": sx0, "y": sy0, "side": side},
    "palette": [{"index": i, "name": n, "hex": h, "count": counts.get(i, 0)}
                for i, (n, h) in enumerate(PALETTE)],
    "cells": [[int(grid[i, j]) for j in range(GRID)] for i in range(GRID)],
}
with open(ROOT / "pattern.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(data, f, ensure_ascii=False, indent=1)

print("[done] pixel-cake.png / pixel-cake@48.png / perler-cake.png / pattern.json")
