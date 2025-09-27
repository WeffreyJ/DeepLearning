#!/usr/bin/env python3
import argparse, os, time
import numpy as np
import imageio.v2 as iio
import imageio_ffmpeg as ioff
from yt_dlp import YoutubeDL
import cv2
import yaml
import torch

from dlrepo.models.depth.midas_small import MiDaSSmall
from dlrepo.perception.utils import colorize_depth

def resolve_youtube_stream(url: str, max_height=360, max_fps=30) -> str:
    """Return a direct video URL (no download)."""
    ydl_opts = {
        "quiet": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "format": f"mp4[height<={max_height}][fps<={max_fps}]/best[height<={max_height}][fps<={max_fps}]"
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        # HLS or direct; prefer "url" if present, else pick best format
        if "url" in info:
            return info["url"]
        fmts = info.get("formats", [])
        if not fmts:
            raise RuntimeError("No playable formats found for this URL.")
        # choose first playable
        for f in fmts:
            if f.get("url"):
                return f["url"]
        raise RuntimeError("Could not find a playable direct URL.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--youtube", required=True, help="YouTube URL")
    ap.add_argument("--config", default="configs/depth_midas_infer.yaml")
    ap.add_argument("--seconds", type=int, default=20, help="Max seconds to process")
    ap.add_argument("--stride", type=int, default=2, help="Process every Nth frame")
    ap.add_argument("--output", default="outputs/depth_yt.mp4", help="Output video path")
    ap.add_argument("--no-save", action="store_true", help="Don’t save; just run through")
    ap.add_argument("--preview", action="store_true", help="Show a live preview window")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Config (optional device override)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    device = cfg.get("runtime", {}).get("device", "auto")

    # Depth model
    dmodel = MiDaSSmall(device=device)

    # Resolve YouTube stream URL
    stream_url = resolve_youtube_stream(args.youtube)

    # Open reader via ffmpeg (imageio-ffmpeg)
    reader = iio.get_reader(stream_url, plugin="ffmpeg")

    # Try to read metadata
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 24.0))
    size = meta.get("size", None)  # (W, H)
    if not size:
        # Fallback: probe first frame
        try:
            first = reader.get_next_data()
            h, w = first.shape[:2]
            size = (w, h)
            reader.close()
            reader = iio.get_reader(stream_url, plugin="ffmpeg")
        except Exception:
            size = (640, 360)
    W, H = size

    # Writer (unless --no-save)
    writer = None
    if not args.no_save:
        writer = iio.get_writer(
            args.output,
            fps=min(fps, 30),
            codec="libx264",  # falls back if unavailable
            quality=7,
            pixelformat="yuv420p"
        )

    start = time.time()
    frame_idx = 0
    written = 0

    try:
        for frame in reader:
            elapsed = time.time() - start
            if elapsed > args.seconds:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue

            # Ensure BGR (OpenCV) for our depth wrapper API
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            with torch.no_grad():
                depth = dmodel.infer(bgr)  # HxW float32 [0,1]
            depth_color = colorize_depth(depth)  # BGR uint8

            # Make side-by-side view (input RGB + depth BGR)
            vis_left = cv2.resize(bgr, (W, H))
            vis_right = cv2.resize(depth_color, (W, H))
            vis = np.hstack([vis_left, vis_right])  # BGR

            if args.preview:
                cv2.imshow("YouTube depth (left=input, right=depth)", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if writer is not None:
                # Writer expects RGB
                writer.append_data(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                written += 1

            frame_idx += 1
    finally:
        try:
            reader.close()
        except Exception:
            pass
        if writer is not None:
            writer.close()
        if args.preview:
            cv2.destroyAllWindows()

    if writer is not None:
        print(f"Wrote ~{written} frames to {args.output}")
    else:
        print("Stream processed (no file saved).")

if __name__ == "__main__":
    main()
