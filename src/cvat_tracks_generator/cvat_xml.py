from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional
from lxml import etree
from datetime import datetime
import os


@dataclass
class Box:
    frame: int
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    outside: int = 0
    occluded: int = 0
    z_order: int = 0


@dataclass
class Track:
    id: int
    label: str
    boxes: List[Box]
    source: str = "manual"


@dataclass
class TaskMeta:
    id: int = 1
    name: str = "Generated Task"
    size: int = 0
    mode: str = "interpolation"
    overlap: int = 5
    created: str = ""
    updated: str = ""
    subset: str = "default"
    start_frame: int = 0
    stop_frame: int = 0
    width: int = 1920
    height: int = 1080
    source: str = ""


def _get_current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f+00:00")


def read_cvat_xml(path: str) -> Dict[int, Track]:
    tree = etree.parse(path)
    root = tree.getroot()
    tracks: Dict[int, Track] = {}
    for track_el in root.findall("./track"):
        track_id = int(track_el.get("id"))
        label = track_el.get("label", "object")
        source = track_el.get("source", "manual")
        boxes: List[Box] = []
        for box_el in track_el.findall("box"):
            boxes.append(
                Box(
                    frame=int(box_el.get("frame")),
                    xtl=float(box_el.get("xtl")),
                    ytl=float(box_el.get("ytl")),
                    xbr=float(box_el.get("xbr")),
                    ybr=float(box_el.get("ybr")),
                    outside=int(box_el.get("outside", "0")),
                    occluded=int(box_el.get("occluded", "0")),
                    z_order=int(box_el.get("z_order", "0")),
                )
            )
        boxes.sort(key=lambda b: b.frame)
        tracks[track_id] = Track(id=track_id, label=label, boxes=boxes, source=source)
    return tracks


def write_cvat_xml(tracks: Dict[int, Track], path: str, task_meta: Optional[TaskMeta] = None) -> None:
    root = etree.Element("annotations")
    
    # Add version
    version_el = etree.SubElement(root, "version")
    version_el.text = "1.1"
    
    # Add meta section
    meta_el = etree.SubElement(root, "meta")
    
    # Task info
    task_el = etree.SubElement(meta_el, "task")
    
    if task_meta is None:
        # Generate default meta
        max_frame = max((max((b.frame for b in t.boxes), default=0) for t in tracks.values()), default=0)
        task_meta = TaskMeta(
            size=max_frame + 1,
            stop_frame=max_frame,
            created=_get_current_timestamp(),
            updated=_get_current_timestamp(),
            source=os.path.basename(path).replace('.xml', '.mp4')
        )
    
    # Task fields
    etree.SubElement(task_el, "id").text = str(task_meta.id)
    etree.SubElement(task_el, "name").text = task_meta.name
    etree.SubElement(task_el, "size").text = str(task_meta.size)
    etree.SubElement(task_el, "mode").text = task_meta.mode
    etree.SubElement(task_el, "overlap").text = str(task_meta.overlap)
    etree.SubElement(task_el, "bugtracker")
    etree.SubElement(task_el, "created").text = task_meta.created
    etree.SubElement(task_el, "updated").text = task_meta.updated
    etree.SubElement(task_el, "subset").text = task_meta.subset
    etree.SubElement(task_el, "start_frame").text = str(task_meta.start_frame)
    etree.SubElement(task_el, "stop_frame").text = str(task_meta.stop_frame)
    etree.SubElement(task_el, "frame_filter")
    
    # Segments
    segments_el = etree.SubElement(task_el, "segments")
    segment_el = etree.SubElement(segments_el, "segment")
    etree.SubElement(segment_el, "id").text = str(task_meta.id)
    etree.SubElement(segment_el, "start").text = str(task_meta.start_frame)
    etree.SubElement(segment_el, "stop").text = str(task_meta.stop_frame)
    etree.SubElement(segment_el, "url").text = ""
    
    # Owner
    owner_el = etree.SubElement(task_el, "owner")
    etree.SubElement(owner_el, "username").text = "generator"
    etree.SubElement(owner_el, "email").text = ""
    
    etree.SubElement(task_el, "assignee")
    
    # Labels
    labels_el = etree.SubElement(task_el, "labels")
    unique_labels = set(t.label for t in tracks.values())
    colors = ["#ff1616", "#004fff", "#b83df5", "#00ff00", "#ffff00", "#ff00ff", "#00ffff"]
    for i, label in enumerate(unique_labels):
        label_el = etree.SubElement(labels_el, "label")
        etree.SubElement(label_el, "name").text = label
        etree.SubElement(label_el, "color").text = colors[i % len(colors)]
        etree.SubElement(label_el, "type").text = "rectangle"
        etree.SubElement(label_el, "attributes")
    
    # Original size
    orig_size_el = etree.SubElement(task_el, "original_size")
    etree.SubElement(orig_size_el, "width").text = str(task_meta.width)
    etree.SubElement(orig_size_el, "height").text = str(task_meta.height)
    
    # Source
    etree.SubElement(task_el, "source").text = task_meta.source
    
    # Dumped timestamp
    etree.SubElement(meta_el, "dumped").text = _get_current_timestamp()
    
    # Tracks
    for track in tracks.values():
        tr_el = etree.SubElement(root, "track", id=str(track.id), label=track.label, source=track.source)
        for b in sorted(track.boxes, key=lambda bb: bb.frame):
            etree.SubElement(
                tr_el,
                "box",
                frame=str(b.frame),
                keyframe="1",
                outside=str(b.outside),
                occluded=str(b.occluded),
                xtl=str(b.xtl),
                ytl=str(b.ytl),
                xbr=str(b.xbr),
                ybr=str(b.ybr),
                z_order=str(b.z_order),
            )
    
    tree = etree.ElementTree(root)
    tree.write(path, pretty_print=True, xml_declaration=True, encoding="utf-8")


def _boxes_to_map(boxes: Iterable[Box]) -> Dict[int, Box]:
    return {b.frame: b for b in boxes}


def _clone_box_with_frame(box: Box, frame: int) -> Box:
    return Box(frame=frame, xtl=box.xtl, ytl=box.ytl, xbr=box.xbr, ybr=box.ybr, outside=box.outside, occluded=box.occluded, z_order=box.z_order)


def merge_tracks_by_ids(tracks: Dict[int, Track], ids_in_priority: List[int]) -> Dict[int, Track]:
    if not ids_in_priority:
        return tracks
    ids_in_priority = [tid for tid in ids_in_priority if tid in tracks]
    if len(ids_in_priority) <= 1:
        return tracks

    base_id = ids_in_priority[0]
    base_track = tracks[base_id]
    merged_map = _boxes_to_map(base_track.boxes)

    for tid in ids_in_priority[1:]:
        t = tracks[tid]
        t_map = _boxes_to_map(t.boxes)
        if not merged_map and not t_map:
            continue
        frames_all = sorted(set(merged_map.keys()) | set(t_map.keys()))
        if not frames_all:
            continue

        # Ensure continuity: fill gaps in merged_map before integrating t
        merged_map = _fill_gaps_with_last_value(merged_map)
        t_map = _fill_gaps_with_last_value(t_map)

        for f in frames_all:
            if f in merged_map and f in t_map:
                # Overlap: keep base priority (existing merged value)
                continue
            elif f in merged_map:
                continue
            else:
                merged_map[f] = t_map[f]

    # After all merges, ensure continuity again (case A between distant segments)
    merged_map = _fill_gaps_with_last_value(merged_map)

    # Reassign into base track and remove others
    merged_boxes = [merged_map[f] for f in sorted(merged_map.keys())]
    tracks[base_id] = Track(id=base_id, label=base_track.label, boxes=merged_boxes, source=base_track.source)
    for tid in ids_in_priority[1:]:
        if tid in tracks:
            del tracks[tid]
    return tracks


def _fill_gaps_with_last_value(frame_to_box: Dict[int, Box]) -> Dict[int, Box]:
    if not frame_to_box:
        return frame_to_box
    frames_sorted = sorted(frame_to_box.keys())
    filled: Dict[int, Box] = {}
    last_box: Optional[Box] = None
    for f in range(frames_sorted[0], frames_sorted[-1] + 1):
        if f in frame_to_box:
            last_box = frame_to_box[f]
            filled[f] = frame_to_box[f]
        elif last_box is not None:
            filled[f] = _clone_box_with_frame(last_box, f)
    return filled


def delete_tracks_by_ids(tracks: Dict[int, Track], ids_to_delete: set[int]) -> Dict[int, Track]:
    return {tid: t for tid, t in tracks.items() if tid not in ids_to_delete}

