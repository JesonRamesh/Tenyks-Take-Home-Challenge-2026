"""Drawing primitives for the overlay: fonts, palette, rounded shapes, text compositing.

Presentation concerns only — nothing here reads or affects detection, tracking, zone,
dwell or staff logic. Kept separate from renderer.py so the renderer stays a description
of *what* is drawn and this holds *how* it is drawn.

Text goes through PIL rather than cv2.putText: the Hershey fonts cv2 ships are stroked
vector fonts with no hinting or anti-aliasing, which is what makes an overlay read as a
debug view. Converting the frame to PIL per text call would be wasteful, so text is
queued and composited in a single pass (TextLayer) after all cv2 drawing is done.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ordered by preference, covering macOS then common Linux/Colab locations. The renderer
# must not depend on a specific machine's fonts, so this falls back to PIL's bundled
# default rather than failing if none are present.
_FONT_CANDIDATES = (
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Helvetica.ttc"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
)

# Person-box palette: saturated but light enough to stay legible on a dim CCTV frame, and
# spaced around the hue circle so adjacent ids never read as the same colour. Cyan is
# deliberately absent — it belongs to the ROI outline, which must not be confusable with a
# person. Red/amber are absent too: those are the staff signal. BGR, as OpenCV expects.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (247, 195, 79),   # light blue
    (132, 199, 129),  # green
    (79, 213, 255),   # amber-yellow
    (200, 104, 186),  # purple
    (146, 98, 240),   # pink
    (101, 204, 156),  # lime
    (203, 134, 121),  # indigo
)
STAFF_COLOR = (47, 47, 211)     # strong red, reserved so staff never cycle into the palette
ROI_COLOR = (218, 198, 38)      # cyan, reserved for the zone outline
PANEL_BG = (28, 24, 20)         # near-black, tinted slightly warm so it reads as a panel
TEXT_COLOR = (245, 245, 245)
ACCENT = ROI_COLOR


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for regular, heavy in _FONT_CANDIDATES:
        path = Path(heavy if bold else regular)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def color_for(track_id: int) -> tuple[int, int, int]:
    """Stable colour per id, cycling once more ids are active than the palette holds."""
    return PALETTE[track_id % len(PALETTE)]


def rounded_rect(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 6,
    thickness: int = -1,
    alpha: float = 1.0,
) -> None:
    """Filled (thickness -1) or outlined rounded rectangle, anti-aliased, drawn in place.

    alpha < 1 blends against what is already there, which is how the panel and the zone
    fill stay readable without hiding the scene behind them.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    target = image if alpha >= 1.0 else image.copy()

    if thickness < 0:
        cv2.rectangle(target, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(target, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)
    else:
        cv2.line(target, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(target, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(target, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(target, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)

    for (cx, cy), start in (
        ((x1 + radius, y1 + radius), 180),
        ((x2 - radius, y1 + radius), 270),
        ((x2 - radius, y2 - radius), 0),
        ((x1 + radius, y2 - radius), 90),
    ):
        cv2.ellipse(target, (cx, cy), (radius, radius), start, 0, 90, color, thickness, cv2.LINE_AA)

    if alpha < 1.0:
        cv2.addWeighted(target, alpha, image, 1.0 - alpha, 0, dst=image)


def fill_polygon(image: np.ndarray, polygon: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    """Translucent polygon fill, so the zone reads as an area without hiding people in it."""
    layer = image.copy()
    cv2.fillPoly(layer, [polygon], color, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, image, 1.0 - alpha, 0, dst=image)


class TextLayer:
    """Queues text and composites it in one PIL pass.

    A frame round-trips BGR->RGB->BGR once per frame instead of once per string, which
    matters at 30 fps over a full slice.
    """

    def __init__(self) -> None:
        self._items: list[tuple[tuple[int, int], str, ImageFont.FreeTypeFont, tuple[int, int, int], str]] = []

    def add(
        self,
        position: tuple[int, int],
        text: str,
        font: ImageFont.FreeTypeFont,
        color: tuple[int, int, int],
        anchor: str = "la",
    ) -> None:
        self._items.append((position, text, font, color, anchor))

    def flush(self, frame: np.ndarray) -> np.ndarray:
        if not self._items:
            return frame
        canvas = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(canvas)
        for position, text, font, color, anchor in self._items:
            draw.text(position, text, font=font, fill=(color[2], color[1], color[0]), anchor=anchor)
        self._items.clear()
        return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def text_size(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top
