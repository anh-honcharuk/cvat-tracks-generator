from __future__ import annotations

from typing import Dict
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

