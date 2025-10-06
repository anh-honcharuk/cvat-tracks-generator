from __future__ import annotations

from typing import Optional, Dict

import cv2
from sahi.models.yolov5 import Yolov5DetectionModel
from ultralytics import YOLO

from .cvat_xml import write_cvat_xml, Box
from .tracker import run_bytetrack, run_sahi_iou_track
from .config import sahi_cfg
from .utils import video_meta, create_video_writer


def detect_and_track_to_xml(model_path: str, video_path: str, out_xml_path: str, out_video_path: Optional[str], use_sahi: bool = False) -> None:
    """
    When use_sahi is True: use SAHI sliced inference + simple IoU tracker.
    Otherwise: run full-frame inference with built-in tracker (ByteTrack) via Ultralytics
    """
    if use_sahi:
        det_model = Yolov5DetectionModel(
            model_path=model_path,
            confidence_threshold=0.25,
            device="cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
            image_size=640,
        )

        tracks = run_sahi_iou_track(
            detection_model=det_model,
            video_path=video_path,
            slice_height=sahi_cfg.slice_height,
            slice_width=sahi_cfg.slice_width,
            overlap_height_ratio=sahi_cfg.overlap_height_ratio,
            overlap_width_ratio=sahi_cfg.overlap_width_ratio,
            conf=sahi_cfg.conf,
            iou_threshold=sahi_cfg.iou_threshold,
            max_age=sahi_cfg.max_age,
        )

        # Optional visualization: draw track boxes per frame and save
        if out_video_path is not None:
            width, height, fps, _ = video_meta(video_path)
            writer = create_video_writer(out_video_path, width, height, fps)

            # Build per-frame index
            frame_to_boxes: Dict[int, list[tuple[int, Box]]] = {}
            for tid, t in tracks.items():
                for b in t.boxes:
                    frame_to_boxes.setdefault(b.frame, []).append((tid, b))

            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            while True:
                ret, img = cap.read()
                if not ret:
                    break
                for pair in frame_to_boxes.get(frame_idx, []):
                    tid, b = pair
                    x1, y1, x2, y2 = int(b.xtl), int(b.ytl), int(b.xbr), int(b.ybr)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"ID {tid}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                writer.write(img)
                frame_idx += 1
            cap.release()
            writer.release()

    else:
        # Full-frame inference with built-in ByteTrack
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

