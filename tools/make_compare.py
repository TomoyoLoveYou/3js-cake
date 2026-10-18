# -*- coding: utf-8 -*-
"""常见工具 vs 新流程：同一组分辨率下的对比图

「常见工具」= 最典型的在线转换器做法：Lanczos 缩小到目标格数 + 中位切分量化到 16 色。
新流程 = tools/render_to_pixel.py 的默认参数输出。
所有面板按同一显示尺寸、整数倍最近邻放大，不做二次重采样。
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
A = REPO / "video-assets"
P = A / "pixel"
F_B = "C:/Windows/Fonts/msyhbd.ttc"
F_R = "C:/Windows/Fonts/msyh.ttc"

PAPER = (247, 244, 250)
INK = (58, 48, 68)
SUB = (132, 122, 146)
CARD = (255, 255, 255)
LINE = (226, 218, 236)
GRIDS = (48, 32, 24)
SAVE_NAIVE = (96, 48)   # 供视频/拼豆参考的「常见工具」版本
S = 288          # 面板边长（48/32/24 的整数倍）
NAIVE_COLORS = 16


def upscale(grid_png: Path, grid: int) -> Image.Image:
    k = S // grid
    assert S % grid == 0, (S, grid)
    return Image.open(grid_png).convert("RGB").resize((grid * k, grid * k), Image.NEAREST)


def naive(crop: Image.Image, grid: int) -> Image.Image:
    sm = crop.resize((grid, grid), Image.LANCZOS)
    q = sm.quantize(colors=NAIVE_COLORS, method=Image.Quantize.MEDIANCUT).convert("RGB")
    return q.resize((S, S), Image.NEAREST)


def main():
    src = Image.open(A / "render-reset-view.png").convert("RGB")
    meta = json.loads((P / "cake@96.json").read_text(encoding="utf-8"))
    c = meta["crop"]
    crop = src.crop((c["x"], c["y"], c["x"] + c["side"], c["y"] + c["side"]))

    x0, y0, gap = 220, 186, 24
    W = x0 + (len(GRIDS) + 1) * (S + gap) - gap + 56
    H = y0 + 2 * (S + 74) + 120
    sheet = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(sheet)
    f_title = ImageFont.truetype(F_B, 42)
    f_cap = ImageFont.truetype(F_B, 23)
    f_row = ImageFont.truetype(F_B, 26)
    f_note = ImageFont.truetype(F_R, 19)

    d.text((56, 34), "草莓蛋糕：常见工具 vs 轮廓优先流程", font=f_title, fill=INK)
    d.text((56, 92), "同一张渲染图、同一组格子数。常见工具 = Lanczos 缩小 + 16 色中位切分量化",
           font=f_note, fill=SUB)

    rows = [("常见工具", "naive"), ("新流程", "ours")]

    for r, (label, kind) in enumerate(rows):
        y = y0 + r * (S + 74)
        d.text((56, y + S / 2), label, font=f_row, fill=INK, anchor="lm")
        for k in range(len(GRIDS) + 1):
            x = x0 + k * (S + gap)
            d.rounded_rectangle([x - 9, y - 9, x + S + 9, y + S + 9], radius=14,
                                fill=CARD, outline=LINE, width=2)
            if k == 0:
                im = crop.resize((S, S), Image.LANCZOS)
            elif kind == "naive":
                im = naive(crop, GRIDS[k - 1])
            else:
                im = upscale(P / f"cake@{GRIDS[k-1]}.png", GRIDS[k - 1])
            sheet.paste(im, (x, y))

    for k, cap in enumerate(["原渲染"] + [f"{g}×{g}" for g in GRIDS]):
        x = x0 + k * (S + gap) + S / 2
        d.text((x, y0 - 30), cap, font=f_cap, fill=SUB, anchor="mm")

    d.text((56, H - 96),
           "48 格：常见工具把描边平均掉、草莓糊成灰块；新流程轮廓闭合、盘子还在。",
           font=f_note, fill=SUB)
    d.text((56, H - 66),
           "32 格：新流程还能分出两层和盘子，但描边已经占满格子；常见工具彻底没有轮廓，只剩粉色渐变。",
           font=f_note, fill=SUB)
    d.text((56, H - 36),
           "24 格：这张渲染图的线密度已经超过格子数，两种做法都不能用 —— 要到这个尺寸，直接照 32 格重画更快。",
           font=f_note, fill=SUB)

    # 单独存一份 96 / 48 的常见工具版本，便于和自有素材并排比较
    for g in SAVE_NAIVE:
        k = 768 // g
        img = naive(crop, g)
        img.resize((g, g), Image.NEAREST).save(P / f"naive@{g}.png")
        img.save(P / f"naive@{g}x{k}.png")

    out = P / "compare.png"
    sheet.save(out)
    print("saved", out, sheet.size)


if __name__ == "__main__":
    main()
