"""Баннер и аватар для канала: «клиническая картография» измерения.

Идея: баннер выглядит как распечатка с прибора, а не как рекламный постер. Почти
чёрный фон, тонкая сетка-миллиметровка, одна светящаяся линия — та самая ось,
которая растёт вместе с alpha. Подписи редкие и мелкие: название, версия, состав.

Цифры на графике не выдуманы: это замер Qwen2.5-1.5B (4 бита, слой 10) из
`studies/02-самоотчёт-о-боли-проценты` — проекция по ходу генерации на уровнях
alpha 0.0…0.5.

    python tools/make_banner.py            # docs/banner.png и docs/avatar.png
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SKILL_FONTS = Path.home() / ".dsh" / "skills" / "canvas-design" / "canvas-fonts"
WINDOWS_FONTS = Path("C:/Windows/Fonts")
OUT_BANNER = ROOT / "docs" / "banner.png"
OUT_AVATAR = ROOT / "docs" / "avatar.png"

# Палитра: почти чёрный, холодный серый, один акцент — индиго/лаванда
BG_TOP = (10, 15, 24)
BG_BOTTOM = (16, 23, 36)
GRID = (120, 140, 175, 16)
GRID_MAJOR = (120, 140, 175, 30)
TEXT = (232, 237, 245)
MUTED = (138, 152, 174)
ACCENT = (124, 132, 255)
ACCENT_SOFT = (168, 160, 255)
HAIR = (70, 82, 104)

# Реальные измерения: alpha 0.0…0.5 → проекция по ходу генерации
MEASURED = [(-0.69, 0.00), (1.55, 0.10), (5.58, 0.20), (9.62, 0.30),
            (13.66, 0.40), (17.46, 0.50)]


def _cyrillic_ok(path: Path, size: int) -> bool:
    """Отличить настоящую кириллицу от квадратика-заглушки."""
    try:
        font = ImageFont.truetype(str(path), size)
        tofu = font.getmask("\ue000")
        probe = (font.getmask("О"), font.getmask("б"), font.getmask("ы"))
        return all(bytes(m) != bytes(tofu) for m in probe)
    except Exception:
        return False


def pick(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for name in candidates:
        for base in (SKILL_FONTS, WINDOWS_FONTS):
            path = base / name
            if path.exists() and _cyrillic_ok(path, size):
                return ImageFont.truetype(str(path), size)
    return ImageFont.truetype(str(WINDOWS_FONTS / "segoeui.ttf"), size)


def spaced(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, tracking: float = 0.0,
           anchor_left: bool = True) -> float:
    """Нарисовать текст с трекингом (межбуквенным интервалом) и вернуть ширину."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += font.getlength(ch) + tracking
    return x - xy[0]


def text_width(font, text: str, tracking: float = 0.0) -> float:
    return sum(font.getlength(ch) for ch in text) + tracking * max(0, len(text) - 1)


def gradient(size: tuple[int, int]) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, BG_TOP)
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(1, h - 1)
        t = t ** 0.85
        colour = tuple(int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3))
        d.line([(0, y), (w, y)], fill=colour)
    return img


def draw_grid(draw: ImageDraw.ImageDraw, size: tuple[int, int], margin: int) -> None:
    w, h = size
    step = 32
    for x in range(margin, w - margin + 1, step):
        major = (x - margin) % (step * 4) == 0
        draw.line([(x, margin), (x, h - margin)], fill=GRID_MAJOR if major else GRID)
    for y in range(margin, h - margin + 1, step):
        major = (y - margin) % (step * 4) == 0
        draw.line([(margin, y), (w - margin, y)], fill=GRID_MAJOR if major else GRID)


PLOT_LEFT, PLOT_RIGHT = 620, 1224
PLOT_TOP, PLOT_BOTTOM = 150, 496


def draw_axis(draw: ImageDraw.ImageDraw, size: tuple[int, int], margin: int) -> None:
    """График: светящаяся линия замеров, заливка под ней, редкие подписи."""
    left, right, top, bottom = PLOT_LEFT, PLOT_RIGHT, PLOT_TOP, PLOT_BOTTOM
    lo, hi = -2.5, 19.0

    def y_of(v: float) -> float:
        return bottom - (v - lo) / (hi - lo) * (bottom - top)

    def x_of(i: int) -> float:
        return left + (right - left) * i / (len(MEASURED) - 1)

    for value in (0, 4, 8, 12, 16):
        y = y_of(value)
        draw.line([(left, y), (right, y)], fill=GRID_MAJOR if value == 0 else GRID)
    draw.line([(left, top - 18), (left, bottom + 10)], fill=HAIR)
    draw.line([(left, bottom), (right, bottom)], fill=HAIR)

    points = [(x_of(i), y_of(value)) for i, (value, _a) in enumerate(MEASURED)]

    # заливка под линией: полупрозрачный акцент
    fill_layer = Image.new("RGBA", size, (0, 0, 0, 0))
    fd = ImageDraw.Draw(fill_layer)
    fd.polygon(points + [(points[-1][0], bottom), (points[0][0], bottom)],
               fill=ACCENT + (26,))

    # светящаяся линия: размытие + чёткая линия сверху
    line_layer = Image.new("RGBA", size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(line_layer)
    ld.line(points, fill=ACCENT + (255,), width=3, joint="curve")

    _AXIS_LAYER.clear()
    _AXIS_LAYER.append(fill_layer)
    _AXIS_LAYER.append(line_layer.filter(ImageFilter.GaussianBlur(10)))
    _AXIS_LAYER.append(line_layer)

    # точки, подписи alpha и два значения
    for (x, y), (_value, alpha) in zip(points, MEASURED):
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=ACCENT_SOFT, outline=BG_TOP, width=2)
        draw.line([(x, bottom + 2), (x, bottom + 7)], fill=HAIR)
        label = f"{alpha:.1f}"
        draw.text((x - text_width(_FONTS["mono_small"], label) / 2, bottom + 14), label,
                  font=_FONTS["mono_small"], fill=MUTED)
    draw.text((points[-1][0], points[-1][1] - 16), "+17.46",
              font=_FONTS["mono_small"], fill=ACCENT_SOFT, anchor="rs")
    draw.text((points[0][0] + 4, points[0][1] + 16), "−0.69",
              font=_FONTS["mono_small"], fill=MUTED)
    draw.text((left - 14, y_of(0) - 4), "0", font=_FONTS["mono_small"], fill=MUTED,
              anchor="rs")


_AXIS_LAYER: list[Image.Image] = []
_FONTS: dict[str, ImageFont.FreeTypeFont] = {}


def build_banner() -> Path:
    size = (1280, 640)
    margin = 56
    img = gradient(size).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    draw_grid(draw, size, margin)

    _FONTS.clear()
    _FONTS.update({
        "title": pick(["BigShoulders-Bold.ttf", "WorkSans-Bold.ttf"], 84),
        "sub": pick(["InstrumentSans-Regular.ttf", "WorkSans-Regular.ttf"], 34),
        "body": pick(["InstrumentSans-Regular.ttf", "WorkSans-Regular.ttf"], 20),
        "mono": pick(["IBMPlexMono-Regular.ttf", "JetBrainsMono-Regular.ttf"], 16),
        "mono_small": pick(["IBMPlexMono-Regular.ttf", "JetBrainsMono-Regular.ttf"], 14),
    })

    draw_axis(draw, size, margin)
    for layer in _AXIS_LAYER:
        img.alpha_composite(layer)
    draw = ImageDraw.Draw(img, "RGBA")

    # ── левая колонка (до x = 560)
    x = margin
    spaced(draw, (x, 92), "PAIN AXIS", _FONTS["title"], TEXT, tracking=1.2)
    spaced(draw, (x, 196), "WORKBENCH", _FONTS["sub"], MUTED, tracking=11.0)
    draw.line([(x, 258), (x + 148, 258)], fill=HAIR)
    draw.text((x, 280), "оси, стиринг и измерения", font=_FONTS["body"], fill=TEXT)
    draw.text((x, 308), "открытых языковых моделей", font=_FONTS["body"], fill=TEXT)
    draw.text((x, 366), "локальные модели · HF · Ollama", font=_FONTS["mono"], fill=MUTED)

    # ── подпись графика справа сверху
    label = "проекция на ось боли по ходу генерации"
    f = _FONTS["mono_small"]
    draw.text((PLOT_RIGHT - text_width(f, label), 96), label, font=f, fill=MUTED)
    draw.text((PLOT_RIGHT - text_width(f, "alpha 0.0 → 0.5"), 116), "alpha 0.0 → 0.5",
              font=f, fill=HAIR)

    # ── нижняя строка метаданных, только в левой колонке
    meta = "v1.0.0   ·   22 ОСИ   ·   24 ПРЕСЕТА   ·   10 БЛОКОВ"
    spaced(draw, (margin, 560), meta, _FONTS["mono"], MUTED, tracking=1.1)

    # тонкая рамка
    draw.rectangle([42, 42, size[0] - 42, size[1] - 42], outline=HAIR, width=1)

    img.convert("RGB").save(OUT_BANNER, quality=95)
    return OUT_BANNER


def build_avatar() -> Path:
    """Аватар канала: тот же язык, но квадрат и одна линия."""
    size = (512, 512)
    img = gradient(size).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    margin = 40
    draw_grid(draw, size, margin)

    title = pick(["BigShoulders-Bold.ttf", "WorkSans-Bold.ttf"], 132)
    mono = pick(["IBMPlexMono-Regular.ttf", "JetBrainsMono-Regular.ttf"], 22)

    # линия-ось по диагонали
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    pts = [(90, 372), (170, 340), (250, 300), (330, 244), (410, 176)]
    ld.line(pts, fill=ACCENT + (255,), width=4, joint="curve")
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(10)))
    img.alpha_composite(layer)
    draw = ImageDraw.Draw(img, "RGBA")
    for x, y in pts:
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=ACCENT_SOFT, outline=BG_TOP, width=2)

    spaced(draw, (margin + 6, margin + 8), "PAIN", title, TEXT, tracking=1.0)
    spaced(draw, (margin + 6, margin + 128), "AXIS", title, MUTED, tracking=1.0)
    spaced(draw, (margin + 8, size[1] - margin - 30), "WORKBENCH 1.0",
           mono, MUTED, tracking=1.6)
    img.convert("RGB").save(OUT_AVATAR, quality=95)
    return OUT_AVATAR


def main() -> int:
    OUT_BANNER.parent.mkdir(parents=True, exist_ok=True)
    print("баннер:", build_banner())
    print("аватар:", build_avatar())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
