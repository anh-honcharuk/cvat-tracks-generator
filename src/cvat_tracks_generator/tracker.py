from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import cv2
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

from .cvat_xml import Track, Box


def run_bytetrack(model: YOLO, video_path: str) -> Dict[int, Track]:
    """Run Ultralytics ByteTrack on a video and return tracks as dict[id]=Track."""
    tracks: Dict[int, Track] = {}
    frame_index = -1
    for result in model.track(source=video_path, tracker="bytetrack.yaml", stream=True, verbose=False):
        frame_index += 1
        if result.boxes is None:
            continue
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
        clss = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None
        if ids is None:
            continue
        for i in range(boxes_xyxy.shape[0]):
            tid = int(ids[i])
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            label = str(int(clss[i])) if clss is not None else "object"
            if tid not in tracks:
                tracks[tid] = Track(id=tid, label=label, boxes=[])
            tracks[tid].boxes.append(Box(frame=frame_index, xtl=x1, ytl=y1, xbr=x2, ybr=y2, outside=0, occluded=0))
    for t in tracks.values():
        t.boxes.sort(key=lambda b: b.frame)
    return tracks


def _iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _ActiveTrack:
    def __init__(self, track_id: int, bbox: Tuple[float, float, float, float], frame_index: int, label: str) -> None:
        self.track_id = track_id
        self.last_bbox = bbox
        self.last_frame = frame_index
        self.label = label


class SimpleIoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.active: Dict[int, _ActiveTrack] = {}
        self.output_tracks: Dict[int, Track] = {}

    def _start_new_track(self, bbox: Tuple[float, float, float, float], frame_index: int, label: str) -> int:
        tid = self.next_id
        self.next_id += 1
        self.active[tid] = _ActiveTrack(tid, bbox, frame_index, label)
        if tid not in self.output_tracks:
            self.output_tracks[tid] = Track(id=tid, label=label, boxes=[])
        return tid

    def _assign(self, frame_index: int, detections: List[Tuple[float, float, float, float]], labels: List[str]) -> List[Optional[int]]:
        # Greedy matching by IoU
        assigned_ids: List[Optional[int]] = [None] * len(detections)
        # Filter active by age
        valid_active = {tid: tr for tid, tr in self.active.items() if frame_index - tr.last_frame <= self.max_age}
        used_tracks: set[int] = set()
        for i, det in enumerate(detections):
            best_tid: Optional[int] = None
            best_iou = 0.0
            for tid, tr in valid_active.items():
                if tid in used_tracks:
                    continue
                iou_val = _iou(det, tr.last_bbox)
                if iou_val > self.iou_threshold and iou_val > best_iou:
                    best_iou = iou_val
                    best_tid = tid
            if best_tid is not None:
                assigned_ids[i] = best_tid
                used_tracks.add(best_tid)
        return assigned_ids

    def update(self, frame_index: int, detections: List[Tuple[float, float, float, float]], labels: List[str]) -> List[int]:
        assigned = self._assign(frame_index, detections, labels)
        # Create ids for unassigned detections
        for i, det in enumerate(detections):
            if assigned[i] is None:
                assigned[i] = self._start_new_track(det, frame_index, labels[i])

        # Update active tracks and output
        for i, det in enumerate(detections):
            tid = assigned[i]
            assert tid is not None
            label = labels[i]
            if tid not in self.active:
                self.active[tid] = _ActiveTrack(tid, det, frame_index, label)
            tr = self.active[tid]
            tr.last_bbox = det
            tr.last_frame = frame_index
            if tid not in self.output_tracks:
                self.output_tracks[tid] = Track(id=tid, label=label, boxes=[])
            x1, y1, x2, y2 = det
            self.output_tracks[tid].boxes.append(
                Box(frame=frame_index, xtl=float(x1), ytl=float(y1), xbr=float(x2), ybr=float(y2), outside=0, occluded=0)
            )

        # Remove too-old tracks from active (kept in output)
        to_delete = [tid for tid, tr in self.active.items() if frame_index - tr.last_frame > self.max_age]
        for tid in to_delete:
            del self.active[tid]

        return [int(tid) for tid in assigned if tid is not None]

    def finalize(self) -> Dict[int, Track]:
        # Sort boxes by frame
        for t in self.output_tracks.values():
            t.boxes.sort(key=lambda b: b.frame)
        return self.output_tracks


def run_sahi_iou_track(
    detection_model,
    video_path: str,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    conf: float = 0.25,
    iou_threshold: float = 0.3,
    max_age: int = 30,
) -> Dict[int, Track]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video")

    tracker = SimpleIoUTracker(iou_threshold=iou_threshold, max_age=max_age)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = get_sliced_prediction(
            image=frame,
            detection_model=detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            postprocess_match_threshold=0.5,
            postprocess_type="NMS",
            verbose=0,
        )

        det_boxes: List[Tuple[float, float, float, float]] = []
        det_labels: List[str] = []
        for op in result.object_prediction_list:
            x1, y1, x2, y2 = op.bbox.to_xyxy()
            det_boxes.append((float(x1), float(y1), float(x2), float(y2)))
            # prefer category name if available, else id
            label = getattr(op.category, "name", None)
            if label is None:
                label = str(getattr(op.category, "id", "object"))
            det_labels.append(str(label))

        tracker.update(frame_index, det_boxes, det_labels)
        frame_index += 1

    cap.release()
    return tracker.finalize()

