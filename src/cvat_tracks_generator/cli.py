import sys
import click

from .cvat_xml import (
    read_cvat_xml,
    write_cvat_xml,
    merge_tracks_by_ids,
    delete_tracks_by_ids,
)
from .renderer import render_xml_on_video
from .detector import detect_and_track_to_xml


@click.group()
def main() -> None:
    """cvat-tracks-generator CLI."""


@main.command("detect-track")
@click.option("--model", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--video", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out-xml", required=True, type=click.Path())
@click.option("--save-video", type=click.Path(), default=None)
@click.option("--use-sahi/--no-sahi", default=False)
def detect_track_cmd(model: str, video: str, out_xml: str, save_video: str | None, use_sahi: bool) -> None:
    """Run detection + ByteTrack and export CVAT XML (optionally save visualization video)."""
    detect_and_track_to_xml(model_path=model, video_path=video, out_xml_path=out_xml, out_video_path=save_video, use_sahi=use_sahi)


@main.command("render")
@click.option("--xml", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--video", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out-video", required=True, type=click.Path())
def render_cmd(xml: str, video: str, out_video: str) -> None:
    """Visualize tracks from CVAT XML over the given video and save output."""
    tracks = read_cvat_xml(xml)
    render_xml_on_video(tracks, video, out_video)


@main.command("edit")
@click.option("--xml", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out-xml", required=True, type=click.Path())
@click.option("--merge", default="", help="Comma-separated track IDs to merge into one (priority by order)")
@click.option("--delete", "delete_ids", default="", help="Comma-separated track IDs to delete")
@click.option("--video", type=click.Path(exists=True, dir_okay=False), help="Input video for visualization")
@click.option("--save-video", type=click.Path(), help="Output video with edited tracks visualization")
def edit_cmd(xml: str, out_xml: str, merge: str, delete_ids: str, video: str | None, save_video: str | None) -> None:
    """Merge and/or delete tracks and save to new XML. Optionally render visualization video."""
    tracks = read_cvat_xml(xml)

    merge_list = [int(x) for x in merge.split(",") if x.strip().isdigit()] if merge else []
    delete_list = [int(x) for x in delete_ids.split(",") if x.strip().isdigit()] if delete_ids else []

    if merge_list:
        tracks = merge_tracks_by_ids(tracks, merge_list)
    if delete_list:
        tracks = delete_tracks_by_ids(tracks, set(delete_list))

    write_cvat_xml(tracks, out_xml)
    
    # Optional video visualization
    if video and save_video:
        render_xml_on_video(tracks, video, save_video)


if __name__ == "__main__":
    sys.exit(main())

