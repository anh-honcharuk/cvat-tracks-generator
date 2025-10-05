from __future__ import annotations

import random
from typing import Dict, Deque, Tuple
from collections import deque
import cv2

from .cvat_xml import Track, Box


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    rng = random.Random(track_id)
    return (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))


def render_xml_on_video(tracks: Dict[int, Track], in_video: str, out_video: str, trail: int = 10) -> None:
    cap = cv2.VideoCapture(in_video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open output video writer")

    # Build per-track frame maps and trails
    track_frame_map: Dict[int, Dict[int, Box]] = {tid: {b.frame: b for b in t.boxes} for tid, t in tracks.items()}
    trails: Dict[int, Deque[tuple[int, int]]] = {tid: deque(maxlen=trail) for tid in tracks}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for tid, fmap in track_frame_map.items():
            if frame_idx in fmap:
                b = fmap[frame_idx]
                color = _color_for_id(tid)
                p1 = (int(b.xtl), int(b.ytl))
                p2 = (int(b.xbr), int(b.ybr))
                cv2.rectangle(frame, p1, p2, color, 2)
                cx = int((b.xtl + b.xbr) / 2)
                cy = int((b.ytl + b.ybr) / 2)
                trails[tid].append((cx, cy))
                for i in range(1, len(trails[tid])):
                    cv2.line(frame, trails[tid][i - 1], trails[tid][i], color, 2)
                cv2.putText(frame, f"ID {tid}", (p1[0], max(0, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

