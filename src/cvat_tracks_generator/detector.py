from __future__ import annotations

from typing import Optional
import cv2
from ultralytics import YOLO

from .cvat_xml import write_cvat_xml
from .tracker import run_bytetrack
from .utils import video_meta, create_video_writer


def detect_and_track_to_xml(model_path: str, video_path: str, out_xml_path: str, out_video_path: Optional[str], use_sahi: bool = False) -> None:
    # For MVP: run full-frame inference with built-in tracker (ByteTrack) via Ultralytics
    model = YOLO(model_path)
    tracks = run_bytetrack(model, video_path)

    # Optional visualization: re-run once for frames to draw overlays as we write
    if out_video_path is not None:
        width, height, fps, _ = video_meta(video_path)
        writer = create_video_writer(out_video_path, width, height, fps)
        # Replay detections for visualization using model.track stream
        frame_index = -1
        for result in model.track(source=video_path, tracker="bytetrack.yaml", stream=True, verbose=False):
            frame_index += 1
            img = result.orig_img
            if result.boxes is None:
                writer.write(img)
                continue
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
            if ids is not None:
                for i in range(boxes_xyxy.shape[0]):
                    tid = int(ids[i])
                    x1, y1, x2, y2 = boxes_xyxy[i].astype(int).tolist()
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"ID {tid}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            writer.write(img)
        writer.release()

    write_cvat_xml(tracks, out_xml_path)

