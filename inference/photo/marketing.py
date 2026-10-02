#!/usr/bin/env python3
"""AI 修图 · 上架物料素材生成（#224 A 类）。

输入 = **真实产线** `photo.retouch` 的 before/after 输出目录（`outputs/photo-retouch/<case>/`），
输出 = 渠道上架用效果图（去技术化：不出现模型/管线/内部路径/工单/人名）：

    A1 首图主图           ai-photo-main-01-1200x1200.jpg
    A2 前后对比图 ×N       ai-photo-ba-0N-1200x1200.jpg
    A3 服务四选图          ai-photo-service-01-1200x1200.png
    A4 流程与时效图        ai-photo-flow-01-1200x1200.png
    A5 合规隐私图          ai-photo-compliance-01-1200x1200.png
    A6 价格套餐图          ai-photo-price-01-1200x1200.png

    python3 inference/photo/marketing.py \
        --source /mnt/data/ai_workspace/outputs/photo-retouch \
        --output /mnt/data/ai_workspace/outputs/photo-retouch/marketing

价格/文案口径：`cland-op docs/operations/定价战术-v0.md` + `cland-pdm AI修图-对外上架物料-v1.md`。
样例素材仅 CC0/PD，图内标注「公有领域素材演示」。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

# ------------------------------------------------------------------ 常量 / 字体
W = 1200
FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_MED = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_BLACK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"

INK = (33, 37, 41)
MUTED = (108, 117, 125)
GREEN = (0, 138, 82)
DARK = (18, 26, 32)
GOLD = (198, 152, 62)
BG = (245, 246, 248)
WHITE = (255, 255, 255)


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def load_rgb(path: str | Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def cover(im: Image.Image, w: int, h: int) -> Image.Image:
    """等比缩放 + 居中裁剪填满 w×h。"""
    r = max(w / im.width, h / im.height)
    im2 = im.resize((max(1, round(im.width * r)), max(1, round(im.height * r))), Image.LANCZOS)
    x = (im2.width - w) // 2
    y = (im2.height - h) // 2
    return im2.crop((x, y, x + w, y + h))


def cover_bias(im: Image.Image, w: int, h: int, fy: float = 0.5, fx: float = 0.5) -> Image.Image:
    """cover，用 fy/fx(0..1) 控制裁剪焦点。"""
    r = max(w / im.width, h / im.height)
    im2 = im.resize((max(1, round(im.width * r)), max(1, round(im.height * r))), Image.LANCZOS)
    x = round((im2.width - w) * fx)
    y = round((im2.height - h) * fy)
    return im2.crop((x, y, x + w, y + h))


def contain(im: Image.Image, w: int, h: int) -> Image.Image:
    r = min(w / im.width, h / im.height)
    return im.resize((max(1, round(im.width * r)), max(1, round(im.height * r))), Image.LANCZOS)


def rounded(im: Image.Image, radius: int) -> Image.Image:
    im = im.convert("RGBA")
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, im.width - 1, im.height - 1), radius, fill=255)
    im.putalpha(mask)
    return im


def paste_center(canvas: Image.Image, im: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    canvas.alpha_composite(im, (x0 + (x1 - x0 - im.width) // 2, y0 + (y1 - y0 - im.height) // 2))


def drop_shadow(canvas: Image.Image, box, radius: int, blur: int = 14, alpha: int = 60) -> None:
    x0, y0, x1, y1 = box
    sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((x0 + 6, y0 + 10, x1 + 6, y1 + 10), radius, fill=(0, 0, 0, alpha))
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    canvas.alpha_composite(sh)


def text_shadow(d: ImageDraw.ImageDraw, xy, s: str, f, fill, anchor="la", off=3, shadow=(0, 0, 0, 160)) -> None:
    x, y = xy
    d.text((x + off, y + off), s, font=f, fill=shadow, anchor=anchor)
    d.text((x, y), s, font=f, fill=fill, anchor=anchor)


def pill(d: ImageDraw.ImageDraw, xy, s: str, f, fg, bg, pad=(22, 12)) -> None:
    x, y = xy
    bb = d.textbbox((0, 0), s, font=f)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.rounded_rectangle((x, y, x + w + pad[0] * 2, y + h + pad[1] * 2), (h + pad[1] * 2) // 2, fill=bg)
    d.text((x + pad[0], y + pad[1] - bb[1]), s, font=f, fill=fg)


def gradient_overlay(canvas: Image.Image, box, top=(0, 0, 0), bottom=None, top_a=210, bot_a=0) -> None:
    """在 box 区域加竖向不透明度渐变（用于压暗图片托字）。"""
    x0, y0, x1, y1 = box
    h = y1 - y0
    if bottom is None:
        bottom = top
    grad = Image.new("RGBA", (1, h), (0, 0, 0, 0))
    px = grad.load()
    for i in range(h):
        t = i / max(1, h - 1)
        c = tuple(int(top[k] + (bottom[k] - top[k]) * t) for k in range(3))
        px[0, i] = (*c, int(top_a + (bot_a - top_a) * t))
    grad = grad.resize((x1 - x0, h), Image.NEAREST)
    canvas.alpha_composite(grad, (x0, y0))


# ------------------------------------------------------------------ A1 首图主图
def gen_main(src: Path, out: Path, hero_case: str = "hist01_migrant_mother_colorize") -> dict:
    before = load_rgb(src / hero_case / "before.png")
    after = load_rgb(src / hero_case / "after.png")
    # 焦点偏上（人物面部），主图吸睛
    hero = cover_bias(after, W, W, fy=0.18, fx=0.42)
    canvas = hero.convert("RGBA")
    gradient_overlay(canvas, (0, 0, W, 430), (8, 12, 16), (8, 12, 16), 225, 0)
    gradient_overlay(canvas, (0, W - 360, W, W), (8, 12, 16), (8, 12, 16), 0, 235)
    d = ImageDraw.Draw(canvas)

    # 角标
    pill(d, (56, 54), "AI 技术制作", font(FONT_MED, 34), WHITE, (12, 18, 22, 190))

    # 主标题
    text_shadow(d, (W // 2, 118), "老照片修复 · 上色", font(FONT_BLACK, 96), WHITE, anchor="ma")
    text_shadow(d, (W // 2, 246), "模糊破损黑白照 → 清晰自然彩色", font(FONT_MED, 46), (228, 236, 240), anchor="ma")

    # 底部价格
    text_shadow(d, (W // 2, W - 250), "体验 ¥19 起 · 标准 ¥49 · 精品 ¥129", font(FONT_BOLD, 62), WHITE, anchor="ma")
    pill(d, (W // 2 - 300, W - 158), "新客首单 ¥9.9（限 1 张）", font(FONT_MED, 40), (60, 40, 0), (255, 214, 102))
    d.text((W // 2, W - 74), "效果示意图 · 公有领域素材演示 · 效果受原图质量影响",
           font=font(FONT_REG, 28), fill=(220, 226, 230), anchor="ma")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out, quality=92)
    return {"file": out.name, "role": "A1 首图主图", "source_case": hero_case,
            "source": "photo.retouch colorize 真实产出", "license": "素材 PD/CC0"}


# ------------------------------------------------------------------ A2 前后对比
def ba_card(before: Image.Image, after: Image.Image, title: str, subtitle: str,
            out: Path, focus=(0.5, 0.5), crop_h: float = 1.0) -> None:
    canvas = Image.new("RGBA", (W, W), BG + (255,))
    d = ImageDraw.Draw(canvas)

    # 顶部标题条
    d.rectangle((0, 0, W, 210), fill=DARK)
    d.rectangle((0, 210, W, 216), fill=GOLD)
    d.text((56, 42), title, font=font(FONT_BLACK, 66), fill=WHITE)
    d.text((58, 130), subtitle, font=font(FONT_MED, 38), fill=(196, 206, 214))
    d.text((W - 56, 62), "AI 技术制作", font=font(FONT_MED, 34), fill=(160, 172, 182), anchor="ra")

    # 两栏
    margin, top, gap = 44, 250, 28
    foot = 118
    pw = (W - margin * 2 - gap) // 2
    ph = W - top - foot
    box_b = (margin, top, margin + pw, top + ph)
    box_a = (margin + pw + gap, top, W - margin, top + ph)

    def panel(box, im, label, label_fg, tag):
        x0, y0, x1, y1 = box
        drop_shadow(canvas, box, 18)
        # 白底 + 内嵌图片（contain，保持真实比例）
        d.rounded_rectangle(box, 18, fill=WHITE)
        inner = cover(im, (x1 - x0) - 24, (y1 - y0) - 24)
        card = rounded(inner, 10)
        paste_center(canvas, card, (x0 + 12, y0 + 12, x1 - 12, y1 - 12))
        # 标签
        f = font(FONT_BOLD, 40)
        bb = d.textbbox((0, 0), label, font=f)
        lw, lh = bb[2] - bb[0], bb[3] - bb[1]
        d.rounded_rectangle((x0 + 20, y0 + 20, x0 + 20 + lw + 44, y0 + 20 + lh + 30), 12, fill=label_fg)
        d.text((x0 + 42, y0 + 34 - bb[1]), label, font=f, fill=WHITE)

    panel(box_b, crop_focus(before, focus, crop_h), "原图", (90, 98, 106), "before")
    panel(box_a, crop_focus(after, focus, crop_h), "修复后", GREEN, "after")

    # 底部
    d.text((56, W - 86), "公有领域素材演示", font=font(FONT_BOLD, 34), fill=(150, 108, 34))
    d.text((W - 56, W - 84), "效果受原图质量影响 · 不承诺 100% 还原", font=font(FONT_REG, 30),
           fill=MUTED, anchor="ra")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out, quality=92)


def crop_focus(im: Image.Image, focus=(0.5, 0.5), crop_h: float = 1.0) -> Image.Image:
    """按比例裁剪后再 contain；crop_h<1 取上部/中部带状（用于面部特写）。"""
    if crop_h >= 0.999:
        return im
    h = round(im.height * crop_h)
    y = round((im.height - h) * focus[1])
    x = round(im.width * 0.06)
    return im.crop((x, y, im.width - x, y + h))


# ------------------------------------------------------------------ 通用标题头
def page(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGBA", (W, W), BG + (255,))
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 0, W, 216), fill=DARK)
    d.rectangle((0, 216, W, 222), fill=GOLD)
    d.text((56, 44), title, font=font(FONT_BLACK, 64), fill=WHITE)
    d.text((58, 132), subtitle, font=font(FONT_MED, 36), fill=(196, 206, 214))
    d.text((W - 56, 64), "AI 技术制作", font=font(FONT_MED, 32), fill=(160, 172, 182), anchor="ra")
    return canvas, d


# ------------------------------------------------------------------ A3 服务四选图
def gen_service(src: Path, out: Path) -> None:
    canvas, d = page("四项服务 · 一站搞定", "修复 / 上色 / 高清增强 / 电商主图")
    items = [
        ("老照片修复", "去噪 · 去划痕 · 恢复清晰", "hist01_migrant_mother_repair", (90, 98, 106)),
        ("黑白照上色", "黑白 → 自然彩色", "hist02_lincoln_colorize", GREEN),
        ("高清增强", "小图放大 · 更清晰", "hist01_migrant_mother_upscale", (34, 96, 160)),
        ("电商主图", "去背 · 纯白底 · 自然阴影", "prod01_potpourri_jar", (150, 92, 30)),
    ]
    m, gap, top = 48, 30, 258
    tw = (W - m * 2 - gap) // 2
    th = (W - top - m - gap) // 2
    for i, (name, desc, case, color) in enumerate(items):
        cx = m + (i % 2) * (tw + gap)
        cy = top + (i // 2) * (th + gap)
        box = (cx, cy, cx + tw, cy + th)
        drop_shadow(canvas, box, 18)
        d.rounded_rectangle(box, 18, fill=WHITE)
        im = load_rgb(src / case / "after.png")
        thumb = rounded(cover(im, tw - 24, th - 132), 10)
        canvas.alpha_composite(thumb, (cx + 12, cy + 12))
        d.rounded_rectangle((cx + 12, cy + th - 120, cx + 12 + 160, cy + th - 120 + 14), 7, fill=color)
        d.text((cx + 30, cy + th - 92), name, font=font(FONT_BOLD, 46), fill=INK)
        d.text((cx + 30, cy + th - 38), desc, font=font(FONT_REG, 30), fill=MUTED)
    d.text((W // 2, W - 26), "公有领域素材演示", font=font(FONT_REG, 28), fill=(150, 108, 34), anchor="mm")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out)


# ------------------------------------------------------------------ A4 流程与时效
def gen_flow(out: Path) -> None:
    canvas, d = page("交付流程 · 快至 2 小时", "简单四步，坐等成图")
    steps = [
        ("1", "下单", "选档位（体验/标准/精品）"),
        ("2", "发照片", "拍照/扫描件私聊发我们"),
        ("3", "AI 制作", "确认需求后开始处理"),
        ("4", "回传成图", "标准 24 小时内交付"),
    ]
    m, gap, top = 60, 18, 300
    bw = (W - m * 2 - gap * 3) // 4
    bh = 380
    for i, (n, name, desc) in enumerate(steps):
        x = m + i * (bw + gap)
        box = (x, top, x + bw, top + bh)
        drop_shadow(canvas, box, 18)
        d.rounded_rectangle(box, 18, fill=WHITE)
        d.ellipse((x + bw // 2 - 52, top + 34, x + bw // 2 + 52, top + 138), fill=GREEN)
        d.text((x + bw // 2, top + 62), n, font=font(FONT_BLACK, 62), fill=WHITE, anchor="ma")
        d.text((x + bw // 2, top + 176), name, font=font(FONT_BOLD, 50), fill=INK, anchor="ma")
        # 描述换行
        f = font(FONT_REG, 30)
        words = desc
        lines = [words[j:j + 8] for j in range(0, len(words), 8)]
        for k, ln in enumerate(lines):
            d.text((x + bw // 2, top + 250 + k * 42), ln, font=f, fill=MUTED, anchor="ma")
        if i < 3:
            ax = x + bw + 2
            d.text((ax, top + bh // 2 - 30), "›", font=font(FONT_BLACK, 60), fill=(180, 188, 194), anchor="ma")
    # 时效条
    box = (m, top + bh + 60, W - m, W - 60)
    d.rounded_rectangle(box, 20, fill=DARK)
    d.text((box[0] + 50, box[1] + 46), "标准交付", font=font(FONT_BOLD, 46), fill=(176, 188, 198))
    d.text((box[0] + 50, box[1] + 112), "24 小时内", font=font(FONT_BLACK, 78), fill=WHITE)
    d.text((W - m - 46, box[1] + 46), "加急交付", font=font(FONT_BOLD, 46), fill=(176, 188, 198), anchor="ra")
    d.text((W - m - 46, box[1] + 112), "2 小时内", font=font(FONT_BLACK, 78), fill=(255, 214, 102), anchor="ra")
    d.text((W // 2, W - 26), "时效以工作日计 · 批量按量排期", font=font(FONT_REG, 28), fill=MUTED, anchor="mm")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out)


# ------------------------------------------------------------------ A5 合规隐私图
def gen_compliance(out: Path) -> None:
    canvas, d = page("合规与隐私承诺", "放心把回忆交给我们")
    rows = [
        ("AI 技术制作", "本服务由 AI 完成，非手工绘制，不夸大效果", GREEN),
        ("仅本次使用", "您的照片仅用于本次制作，不作他用", (34, 96, 160)),
        ("完成后删除", "按约定删除原图与成品，不外传、不公开", (150, 92, 30)),
        ("权利请确认", "请确保您拥有照片使用权利（本人/家族照片）", (120, 70, 150)),
        ("不接的委托", "换脸/人脸替换、侵权违法、凭空造细节", (176, 54, 54)),
        ("售后保障", "不满意免费重做 1 次", (176, 132, 30)),
    ]
    m, top, gap = 56, 268, 16
    rh = (W - top - 80 - gap * 5) // 6
    f_name = font(FONT_BOLD, 42)
    f_desc = font(FONT_REG, 32)
    for i, (name, desc, color) in enumerate(rows):
        y = top + i * (rh + gap)
        box = (m, y, W - m, y + rh)
        d.rounded_rectangle(box, 16, fill=WHITE)
        d.rounded_rectangle((m, y, m + 14, y + rh), 7, fill=color)
        d.text((m + 44, y + rh // 2), name, font=f_name, fill=INK, anchor="lm")
        d.text((m + 44 + 300, y + rh // 2), desc, font=f_desc, fill=MUTED, anchor="lm")
    d.text((W // 2, W - 40), "公有领域素材演示 · 实际以您提供的照片为准", font=font(FONT_REG, 28),
           fill=(150, 108, 34), anchor="mm")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out)


# ------------------------------------------------------------------ A6 价格套餐图
def gen_price(out: Path) -> None:
    canvas, d = page("价格套餐 · 明码标价", "老照片修复 / 上色 / 清晰化（按张）")
    tiers = [
        ("体验版", "19", "单张清晰化 / 简单修复", (90, 98, 106), False),
        ("标准版", "49", "修复 + 上色（常见需求）", GREEN, True),
        ("精品版", "129", "破损严重 / 高要求 / 大幅面", (150, 92, 30), False),
    ]
    m, gap, top = 52, 24, 300
    tw = (W - m * 2 - gap * 2) // 3
    th = 380
    from PIL import ImageDraw as _ID  # noqa
    for i, (name, price, desc, color, hot) in enumerate(tiers):
        x = m + i * (tw + gap)
        box = (x, top, x + tw, top + th)
        drop_shadow(canvas, box, 18)
        d.rounded_rectangle(box, 20, fill=color if hot else WHITE)
        fg = WHITE if hot else INK
        sub = (230, 236, 240) if hot else MUTED
        d.text((x + tw // 2, top + 40), name, font=font(FONT_BOLD, 46), fill=fg, anchor="ma")
        d.text((x + tw // 2, top + 108), "¥", font=font(FONT_BOLD, 54), fill=(255, 214, 102) if hot else GOLD,
               anchor="ma")
        d.text((x + tw // 2, top + 150), price, font=font(FONT_BLACK, 130), fill=fg, anchor="ma")
        f = font(FONT_REG, 30)
        lines = [desc[j:j + 8] for j in range(0, len(desc), 8)]
        for k, ln in enumerate(lines):
            d.text((x + tw // 2, top + 300 + k * 40), ln, font=f, fill=sub, anchor="ma")
        if hot:
            d.rounded_rectangle((x + tw // 2 - 78, top - 22, x + tw // 2 + 78, top + 18), 20, fill=GOLD)
            d.text((x + tw // 2, top - 16), "最受欢迎", font=font(FONT_BOLD, 30), fill=WHITE, anchor="ma")
    # 新客条
    y = top + th + 34
    d.rounded_rectangle((m, y, W - m, y + 96), 18, fill=(255, 236, 186))
    d.text((W // 2, y + 48), "新客首单 ¥9.9（限 1 张）· 批量 ≥10 张 8 折", font=font(FONT_BOLD, 40),
           fill=(120, 78, 0), anchor="mm")
    # 电商主图
    y2 = y + 130
    d.rounded_rectangle((m, y2, W - m, W - 70), 18, fill=DARK)
    d.text((m + 44, y2 + 40), "电商主图 / 批量精修", font=font(FONT_BOLD, 46), fill=WHITE)
    d.text((W - 60, y2 + 34), "¥59/张", font=font(FONT_BLACK, 60), fill=(255, 214, 102), anchor="ra")
    d.text((W - 60, y2 + 116), "5 张 ¥199 · 包月 ¥899–1999", font=font(FONT_REG, 30),
           fill=(196, 206, 214), anchor="ra")
    d.text((W // 2, W - 24), "价格以店铺实际标价为准 · 破损严重单独报价", font=font(FONT_REG, 28),
           fill=MUTED, anchor="mm")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out)


# ------------------------------------------------------------------ main
BA_GROUPS = [
    # (文件名, 标题, 副标题, case, 焦点 fy, 裁剪带高)
    ("ai-photo-ba-01-1200x1200.jpg", "老照片上色 · 前后对比", "黑白历史照片 → 自然彩色",
     "hist01_migrant_mother_colorize", 0.30, 0.62),
    ("ai-photo-ba-02-1200x1200.jpg", "老照片修复 · 前后对比", "去噪 / 去划痕 / 恢复清晰",
     "hist01_migrant_mother_repair", 0.30, 0.62),
    ("ai-photo-ba-03-1200x1200.jpg", "老照片上色 · 前后对比", "百年人像 → 自然肤色",
     "hist02_lincoln_colorize", 0.34, 0.60),
    ("ai-photo-ba-04-1200x1200.jpg", "破损修复 · 前后对比", "破损老照片 → 去划痕白斑 · 清晰还原",
     "hist03_burnley_repair", 0.42, 0.9),
    ("ai-photo-ba-05-1200x1200.jpg", "高清增强 · 前后对比", "低清小图 → 高清细节",
     "hist01_migrant_mother_upscale", 0.30, 0.72),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="AI 修图上架物料生成（#224 A 类）")
    ap.add_argument("--source", default="/mnt/data/ai_workspace/outputs/photo-retouch",
                    help="真实产线输出目录（含 <case>/before.png / after.png）")
    ap.add_argument("--output", default="/mnt/data/ai_workspace/outputs/photo-retouch/marketing",
                    help="物料输出目录")
    args = ap.parse_args()
    src = Path(args.source)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []

    # A1
    manifest.append(gen_main(src, out / "ai-photo-main-01-1200x1200.jpg"))
    # A2
    for fname, title, sub, case, fy, ch in BA_GROUPS:
        before = load_rgb(src / case / "before.png")
        after = load_rgb(src / case / "after.png")
        ba_card(before, after, title, sub, out / fname, focus=(0.5, fy), crop_h=ch)
        manifest.append({"file": fname, "role": "A2 修复前后对比图", "title": title,
                         "source_case": case, "source": "photo.retouch 真实产出", "license": "素材 PD/CC0"})
    # A3-A6
    gen_service(src, out / "ai-photo-service-01-1200x1200.png")
    gen_flow(out / "ai-photo-flow-01-1200x1200.png")
    gen_compliance(out / "ai-photo-compliance-01-1200x1200.png")
    gen_price(out / "ai-photo-price-01-1200x1200.png")
    for fn, role in [("ai-photo-service-01-1200x1200.png", "A3 服务四选图"),
                     ("ai-photo-flow-01-1200x1200.png", "A4 流程与时效图"),
                     ("ai-photo-compliance-01-1200x1200.png", "A5 合规隐私图"),
                     ("ai-photo-price-01-1200x1200.png", "A6 价格套餐图")]:
        manifest.append({"file": fn, "role": role, "source": "图文设计（价格引《定价战术 v0》）"})

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK -> {out}  ({len(manifest)} assets)")
    for m in manifest:
        print("  ", m["file"], m["role"])


if __name__ == "__main__":
    main()
