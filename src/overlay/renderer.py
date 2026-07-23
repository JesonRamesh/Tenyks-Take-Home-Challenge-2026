"""Overlay video renderer: draw the pipeline's per-frame result on the source video.

    python -m src.overlay.renderer --video digital_kiosk.mp4 --config configs/cam1.yaml
    python -m src.overlay.renderer --video digital_kiosk.mp4 --config configs/cam1_v2.yaml \
        --out-dir outputs/v2_final/final_sliceb --preview-frame 26800 \
        --preview-out outputs/style_preview.png

Consumes the render artifact run.py emits (render_frames.yaml in --out-dir: the
surviving in-zone boxes on each frame, post stitch / stationarity / staff, each tagged
with its canonical id and whether the track is a customer or staff) plus the source
video, and writes an annotated mp4. tracks.yaml carries only per-visit intervals, not
per-frame boxes, so the artifact is the box source — it is the same full-pipeline
result with the same provenance as tracks.yaml.

Each active track gets a box coloured consistently by id and a label carrying its id and
running dwell; staff render in a reserved red with a STAFF tag rather than being hidden,
so where the staff filter fires stays visible for review. A panel reports the live in-zone
customer count and the mean running dwell of active tracks. Output path, codec, and the
rendered span (inherited from the artifact's frame range, i.e. run.py --slice) are
config-driven. Drawing primitives live in src/overlay/style.py.

--preview-frame renders a single frame to a PNG instead of writing the video, for
reviewing the styling without spending a full render.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.overlay import style

_PANEL_TITLE = "Kiosk Analytics"
# The camera burns graphics into both top corners — a timestamp at the right, a station
# watermark at the left (measured ending near y=50). The panel starts below that band so
# it never overlaps either; a translucent panel laid over burnt-in text makes both
# unreadable.
_PANEL_TOP = 68
_PANEL_LEFT = 16
_PANEL_SIZE = (320, 128)


def _draw_zone(frame: np.ndarray, polygon: np.ndarray, text: style.TextLayer, font) -> None:
    style.fill_polygon(frame, polygon, style.ROI_COLOR, alpha=0.12)
    cv2.polylines(frame, [polygon], True, style.ROI_COLOR, 2, cv2.LINE_AA)
    # Anchor the label to the zone's left edge rather than its apex: the apex sits in the
    # middle of the scene among the kiosks, where the label competes with person labels,
    # while the left edge is open floor. A faint dark backing keeps it legible — the floor
    # there is bright tile, against which a thin cyan glyph washes out.
    corner = polygon[polygon[:, 0].argmin()]
    label = "Kiosk Zone"
    width, height = style.text_size(label, font)
    x, y = int(corner[0]) + 10, int(corner[1]) - 24
    style.rounded_rect(
        frame, (x - 7, y - 4), (x + width + 7, y + height + 6),
        style.PANEL_BG, radius=5, thickness=-1, alpha=0.55,
    )
    text.add((x, y), label, font, style.ROI_COLOR)


def _place_label(
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    taken: list[tuple[int, int, int, int]],
    frame_width: int,
) -> tuple[int, int, int, int]:
    """Label rect sitting just above the box, lifted while it collides with an earlier one.

    Boxes in a queue overlap heavily, so labels stack rather than sit on top of each other.
    Only vertical offset is used: shifting horizontally would separate a label from the
    box it names.
    """
    x1, y1, _x2, _y2 = box
    pad_x, pad_y = 8, 5
    width, height = size[0] + 2 * pad_x, size[1] + 2 * pad_y
    left = max(2, min(x1, frame_width - width - 2))
    top = y1 - height - 4
    for _ in range(6):
        rect = (left, top, left + width, top + height)
        if not any(
            rect[0] < t[2] and rect[2] > t[0] and rect[1] < t[3] and rect[3] > t[1] for t in taken
        ):
            break
        top -= height + 3
    if top < 2:  # no room above (box near the top edge): drop the label inside the box
        top = y1 + 4
    return left, top, left + width, top + height


def _draw_panel(
    frame: np.ndarray,
    text: style.TextLayer,
    fonts: dict,
    count: int,
    avg_dwell: float,
) -> None:
    panel_w, panel_h = _PANEL_SIZE
    x1, y1 = _PANEL_LEFT, _PANEL_TOP
    x2, y2 = x1 + panel_w, y1 + panel_h
    # 0.86 rather than lower: the ceiling behind the top-left corner carries a vent and
    # beam whose edges read through a thinner panel and make the figures look patchy.
    # The scene is still faintly visible, which is the point.
    style.rounded_rect(frame, (x1, y1), (x2, y2), style.PANEL_BG, radius=12, thickness=-1, alpha=0.86)
    style.rounded_rect(frame, (x1, y1), (x2, y2), (70, 66, 62), radius=12, thickness=1, alpha=0.9)
    # Accent rule under the header separates branding from the live numbers.
    cv2.line(frame, (x1 + 18, y1 + 42), (x2 - 18, y1 + 42), style.ACCENT, 1, cv2.LINE_AA)
    text.add((x1 + 18, y1 + 14), _PANEL_TITLE, fonts["title"], style.ACCENT)
    text.add((x1 + 18, y1 + 55), "People in zone", fonts["label"], (170, 170, 170))
    text.add((x2 - 18, y1 + 52), str(count), fonts["value"], style.TEXT_COLOR, anchor="ra")
    text.add((x1 + 18, y1 + 91), "Avg dwell (active)", fonts["label"], (170, 170, 170))
    text.add((x2 - 18, y1 + 88), f"{avg_dwell:.1f}s", fonts["value"], style.TEXT_COLOR, anchor="ra")


def render_frame(
    frame: np.ndarray,
    rows: list,
    polygon: np.ndarray,
    dwell_frames: dict[int, int],
    fps: float,
    fonts: dict,
) -> np.ndarray:
    """Draw one frame's overlay. Pure presentation: rows come from the render artifact."""
    text = style.TextLayer()
    _draw_zone(frame, polygon, text, fonts["zone"])

    customer_count = 0
    active_dwells: list[float] = []
    taken: list[tuple[int, int, int, int]] = []
    # Draw lower boxes first so a nearer person's label lands on top of a farther one's.
    for track_id, x1, y1, x2, y2, kind in sorted(rows, key=lambda r: r[4]):
        is_staff = kind == 1
        color = style.STAFF_COLOR if is_staff else style.color_for(track_id)
        if is_staff:
            label = "STAFF"
        else:
            customer_count += 1
            seconds = dwell_frames[track_id] / fps
            active_dwells.append(seconds)
            label = f"ID {track_id} · {seconds:.1f}s"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        size = style.text_size(label, fonts["label_bold"])
        lx1, ly1, lx2, ly2 = _place_label((x1, y1, x2, y2), size, taken, frame.shape[1])
        taken.append((lx1, ly1, lx2, ly2))
        # Tinted rather than solid, so the label reads as belonging to its box without
        # punching an opaque hole in the scene. White bold text still carries at this
        # alpha because the fill stays close to the box's saturated colour.
        style.rounded_rect(frame, (lx1, ly1), (lx2, ly2), color, radius=5, thickness=-1, alpha=0.72)
        text.add(((lx1 + lx2) // 2, (ly1 + ly2) // 2), label, fonts["label_bold"],
                 (255, 255, 255), anchor="mm")

    avg = sum(active_dwells) / len(active_dwells) if active_dwells else 0.0
    _draw_panel(frame, text, fonts, customer_count, avg)
    return text.flush(frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    # Render a single frame to a PNG and exit, for reviewing the styling cheaply.
    parser.add_argument("--preview-frame", type=int)
    parser.add_argument("--preview-out", type=Path, default=Path("outputs/style_preview.png"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    overlay_cfg = config["overlay"]
    polygon = np.array(config["kiosk_roi"]["roi_polygon"], dtype=np.int32)

    artifact = yaml.safe_load((args.out_dir / "render_frames.yaml").read_text())
    fps = artifact["fps"]
    start_frame, end_frame = artifact["start_frame"], artifact["end_frame"]
    frames = artifact["frames"]

    fonts = {
        "title": style.load_font(20, bold=True),
        "label": style.load_font(16),
        "value": style.load_font(24, bold=True),
        "label_bold": style.load_font(15, bold=True),
        "zone": style.load_font(14, bold=True),
    }

    cap = cv2.VideoCapture(str(args.video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Running dwell per canonical id: in-zone frames seen so far, which lands on the same
    # value the pipeline aggregates once a visit ends.
    dwell_frames: dict[int, int] = defaultdict(int)

    if args.preview_frame is not None:
        if args.preview_frame not in frames:
            raise SystemExit(
                f"frame {args.preview_frame} has no boxes in {args.out_dir}/render_frames.yaml "
                f"(artifact covers {start_frame}-{end_frame})"
            )
        # Accumulate dwell from the artifact's start so the preview shows the real running
        # figure at that frame, not a track's first frame.
        for index in range(start_frame, args.preview_frame + 1):
            for row in frames.get(index, []):
                dwell_frames[row[0]] += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.preview_frame)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"could not read frame {args.preview_frame} from {args.video}")
        out = render_frame(frame, frames[args.preview_frame], polygon, dwell_frames, fps, fonts)
        args.preview_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview_out), out)
        print(f"wrote {args.preview_out} (frame {args.preview_frame}, "
              f"{len(frames[args.preview_frame])} boxes)")
        return

    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(
        overlay_cfg["output_path"],
        cv2.VideoWriter_fourcc(*overlay_cfg["codec"]),
        fps,
        (width, height),
    )
    for frame_index in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        rows = frames.get(frame_index, [])
        for row in rows:
            dwell_frames[row[0]] += 1
        writer.write(render_frame(frame, rows, polygon, dwell_frames, fps, fonts))

    writer.release()
    cap.release()


if __name__ == "__main__":
    main()
