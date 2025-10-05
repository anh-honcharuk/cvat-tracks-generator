from cvat_tracks_generator.cvat_xml import Track, Box, merge_tracks_by_ids, delete_tracks_by_ids, read_cvat_xml, write_cvat_xml
import tempfile
import os


def _make_track(tid: int, frames: list[int], box=(0, 0, 10, 10)) -> Track:
    return Track(id=tid, label="obj", boxes=[Box(frame=f, xtl=box[0], ytl=box[1], xbr=box[2], ybr=box[3]) for f in frames])


def test_merge_gap_filled_case_one():
    # A: 10..50, B: 60..80 -> fill 51..59 with frame 50 values
    A = _make_track(1, list(range(10, 51)))
    B = _make_track(2, list(range(60, 81)))
    tracks = {1: A, 2: B}
    merged = merge_tracks_by_ids(tracks, [1, 2])

    assert 1 in merged and 2 not in merged
    frames = [b.frame for b in merged[1].boxes]
    assert frames[0] == 10
    assert frames[-1] == 80
    # Check filled gap exists
    assert all(f in frames for f in range(51, 60))


def test_merge_overlap_priority_case_two():
    # A: 1..50, B: 40..90 -> frames 40..50 should be from A
    A = _make_track(1, list(range(1, 51)), box=(0, 0, 10, 10))
    B = _make_track(2, list(range(40, 91)), box=(100, 100, 110, 110))
    tracks = {1: A, 2: B}
    merged = merge_tracks_by_ids(tracks, [1, 2])

    frames = {b.frame: b for b in merged[1].boxes}
    # Overlap frames should match A box
    for f in range(40, 51):
        b = frames[f]
        assert (b.xtl, b.ytl, b.xbr, b.ybr) == (0, 0, 10, 10)


def test_delete_tracks():
    A = _make_track(1, [1, 2, 3])
    B = _make_track(2, [4, 5, 6])
    tracks = {1: A, 2: B}
    kept = delete_tracks_by_ids(tracks, {2})
    assert 1 in kept and 2 not in kept


def test_merge_single_track():
    """Test merging single track (should do nothing)"""
    A = _make_track(1, [1, 2, 3])
    tracks = {1: A}
    merged = merge_tracks_by_ids(tracks, [1])
    assert len(merged) == 1
    assert 1 in merged
    assert len(merged[1].boxes) == 3


def test_merge_empty_list():
    """Test merging empty list (should do nothing)"""
    A = _make_track(1, [1, 2, 3])
    tracks = {1: A}
    merged = merge_tracks_by_ids(tracks, [])
    assert len(merged) == 1
    assert 1 in merged


def test_merge_nonexistent_tracks():
    """Test merging tracks that don't exist"""
    A = _make_track(1, [1, 2, 3])
    tracks = {1: A}
    merged = merge_tracks_by_ids(tracks, [1, 999, 888])
    assert len(merged) == 1
    assert 1 in merged


def test_merge_multiple_tracks():
    """Test merging 3+ tracks"""
    A = _make_track(1, [1, 2], box=(0, 0, 10, 10))
    B = _make_track(2, [5, 6], box=(20, 20, 30, 30))
    C = _make_track(3, [10, 11], box=(40, 40, 50, 50))
    tracks = {1: A, 2: B, 3: C}
    merged = merge_tracks_by_ids(tracks, [1, 2, 3])
    
    assert len(merged) == 1
    assert 1 in merged
    frames = [b.frame for b in merged[1].boxes]
    assert frames == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_delete_multiple_tracks():
    """Test deleting multiple tracks"""
    A = _make_track(1, [1, 2, 3])
    B = _make_track(2, [4, 5, 6])
    C = _make_track(3, [7, 8, 9])
    tracks = {1: A, 2: B, 3: C}
    kept = delete_tracks_by_ids(tracks, {2, 3})
    assert len(kept) == 1
    assert 1 in kept
    assert 2 not in kept
    assert 3 not in kept


def test_delete_nonexistent_tracks():
    """Test deleting tracks that don't exist"""
    A = _make_track(1, [1, 2, 3])
    tracks = {1: A}
    kept = delete_tracks_by_ids(tracks, {2, 3, 999})
    assert len(kept) == 1
    assert 1 in kept


def test_delete_all_tracks():
    """Test deleting all tracks"""
    A = _make_track(1, [1, 2, 3])
    B = _make_track(2, [4, 5, 6])
    tracks = {1: A, 2: B}
    kept = delete_tracks_by_ids(tracks, {1, 2})
    assert len(kept) == 0


def test_xml_read_write_roundtrip():
    """Test reading and writing XML maintains data integrity"""
    # Create test tracks
    tracks = {
        1: Track(id=1, label="person", source="manual", boxes=[
            Box(frame=0, xtl=100, ytl=100, xbr=200, ybr=200, outside=0, occluded=0, z_order=0),
            Box(frame=1, xtl=105, ytl=105, xbr=205, ybr=205, outside=0, occluded=0, z_order=0)
        ]),
        2: Track(id=2, label="car", source="manual", boxes=[
            Box(frame=2, xtl=300, ytl=300, xbr=400, ybr=400, outside=0, occluded=0, z_order=0)
        ])
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        temp_path = f.name
    
    try:
        write_cvat_xml(tracks, temp_path)
        
        # Read back
        read_tracks = read_cvat_xml(temp_path)
        
        # Verify data integrity
        assert len(read_tracks) == 2
        assert 1 in read_tracks
        assert 2 in read_tracks
        
        # Check track 1
        track1 = read_tracks[1]
        assert track1.label == "person"
        assert track1.source == "manual"
        assert len(track1.boxes) == 2
        assert track1.boxes[0].frame == 0
        assert track1.boxes[0].xtl == 100.0
        assert track1.boxes[1].frame == 1
        assert track1.boxes[1].xtl == 105.0
        
        # Check track 2
        track2 = read_tracks[2]
        assert track2.label == "car"
        assert track2.source == "manual"
        assert len(track2.boxes) == 1
        assert track2.boxes[0].frame == 2
        assert track2.boxes[0].xtl == 300.0
        
    finally:
        os.unlink(temp_path)


def test_merge_preserves_track_properties():
    """Test that merging preserves track properties (label, source)"""
    A = Track(id=1, label="person", source="manual", boxes=[
        Box(frame=1, xtl=100, ytl=100, xbr=200, ybr=200)
    ])
    B = Track(id=2, label="car", source="auto", boxes=[
        Box(frame=5, xtl=300, ytl=300, xbr=400, ybr=400)
    ])
    
    tracks = {1: A, 2: B}
    merged = merge_tracks_by_ids(tracks, [1, 2])
    
    # Should preserve properties from first track
    merged_track = merged[1]
    assert merged_track.label == "person"  # From track A
    assert merged_track.source == "manual"  # From track A


def test_gap_filling_with_different_box_values():
    """Test gap filling uses exact values from last frame"""
    A = Track(id=1, label="person", boxes=[
        Box(frame=10, xtl=100, ytl=100, xbr=200, ybr=200, outside=1, occluded=1, z_order=5),
        Box(frame=15, xtl=150, ytl=150, xbr=250, ybr=250, outside=0, occluded=0, z_order=3)
    ])
    B = Track(id=2, label="person", boxes=[
        Box(frame=20, xtl=300, ytl=300, xbr=400, ybr=400, outside=0, occluded=0, z_order=1)
    ])
    
    tracks = {1: A, 2: B}
    merged = merge_tracks_by_ids(tracks, [1, 2])
    
    # Check gap frames 16-19 use frame 15 values
    frame_15_box = next(b for b in merged[1].boxes if b.frame == 15)
    frame_16_box = next(b for b in merged[1].boxes if b.frame == 16)
    
    assert frame_16_box.xtl == frame_15_box.xtl
    assert frame_16_box.ytl == frame_15_box.ytl
    assert frame_16_box.xbr == frame_15_box.xbr
    assert frame_16_box.ybr == frame_15_box.ybr
    assert frame_16_box.outside == frame_15_box.outside
    assert frame_16_box.occluded == frame_15_box.occluded
    assert frame_16_box.z_order == frame_15_box.z_order


def test_complex_merge_scenario():
    """Test complex scenario with multiple gaps and overlaps"""
    # Track A: 1-10, Track B: 5-15, Track C: 20-25
    A = _make_track(1, list(range(1, 11)), box=(0, 0, 10, 10))
    B = _make_track(2, list(range(5, 16)), box=(100, 100, 110, 110))
    C = _make_track(3, list(range(20, 26)), box=(200, 200, 210, 210))
    
    tracks = {1: A, 2: B, 3: C}
    merged = merge_tracks_by_ids(tracks, [1, 2, 3])
    
    assert len(merged) == 1
    frames = [b.frame for b in merged[1].boxes]
    
    # Should have continuous frames 1-25
    assert frames == list(range(1, 26))
    
    # Overlap 5-10 should use Track A values
    for f in range(5, 11):
        box = next(b for b in merged[1].boxes if b.frame == f)
        assert (box.xtl, box.ytl, box.xbr, box.ybr) == (0, 0, 10, 10)
    
    # Gap 16-19 should use frame 15 values
    frame_15_box = next(b for b in merged[1].boxes if b.frame == 15)
    for f in range(16, 20):
        box = next(b for b in merged[1].boxes if b.frame == f)
        assert (box.xtl, box.ytl, box.xbr, box.ybr) == (frame_15_box.xtl, frame_15_box.ytl, frame_15_box.xbr, frame_15_box.ybr)


def test_edge_case_empty_tracks():
    """Test edge case with empty tracks"""
    A = Track(id=1, label="person", boxes=[])
    B = Track(id=2, label="car", boxes=[])
    tracks = {1: A, 2: B}
    merged = merge_tracks_by_ids(tracks, [1, 2])
    assert len(merged) == 1
    assert len(merged[1].boxes) == 0


def test_edge_case_single_frame_tracks():
    """Test edge case with single frame tracks"""
    A = _make_track(1, [5])
    B = _make_track(2, [10])
    tracks = {1: A, 2: B}
    merged = merge_tracks_by_ids(tracks, [1, 2])
    assert len(merged) == 1
    frames = [b.frame for b in merged[1].boxes]
    assert frames == [5, 6, 7, 8, 9, 10]


def test_edge_case_identical_tracks():
    """Test edge case with identical tracks"""
    A = _make_track(1, [1, 2, 3], box=(0, 0, 10, 10))
    B = _make_track(2, [1, 2, 3], box=(0, 0, 10, 10))
    tracks = {1: A, 2: B}
    merged = merge_tracks_by_ids(tracks, [1, 2])
    assert len(merged) == 1
    assert len(merged[1].boxes) == 3