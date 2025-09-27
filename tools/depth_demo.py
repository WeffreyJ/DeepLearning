#!/usr/bin/env python3
"""
Depth demo (file / webcam / YouTube) with MiDaS-small.

Features
- Input: --source path|rtsp|http(s) or --youtube URL (uses pytube to resolve direct media URL)
- Robust reader: prefer imageio-ffmpeg; fallback to OpenCV if ffmpeg fails
- Limits: --seconds and --stride to downsample in time (stress test without huge files)
- Output: MP4 side-by-side (left RGB, right depth) unless --no-save
- Preview: --preview to see a live window while processing
- Config: uses configs/depth_midas_infer.yaml (optional) to set device, resize, etc.

Requires:
  pip install imageio imageio-ffmpeg opencv-python pytube pyyaml
"""

import os
import sys
import time
import argparse
import warnings
import yaml
import numpy as np
import cv2
import torch
import imageio.v2 as iio

# Our repo modules
from dlrepo.models.depth.midas_small import MiDaSSmall
from dlrepo.perception.utils import colorize_depth

warnings.filterwarnings("ignore", category=UserWarning)

# -------------- YouTube helpers --------------

def resolve_youtube_direct(url: str) -> str:
    """
    Return a direct media URL for a YouTube video.
    1) Try pytube (progressive mp4)
    2) Fallback: yt-dlp Python API (best mp4 or best available)
    """
    # First: pytube (works for many videos, but can break when YT changes)
    try:
        from pytube import YouTube
        yt = YouTube(url)
        # Prefer progressive mp4 so it's a single URL (video+audio)
        stream = (yt.streams.filter(progressive=True, file_extension="mp4")
                  .order_by("resolution").desc().first())
        if stream and stream.url:
            return stream.url
    except Exception as e:
        print(f"pytube failed: {e}")

    # Fallback: yt-dlp (far more robust)
    try:
        from yt_dlp import YoutubeDL
        ydl_opts = {
            "quiet": True,
            "noplaylist": True,
            # Prefer mp4 if possible; otherwise best available
            "format": "best[ext=mp4]/best",
            # Don’t actually download, just get the direct media URL
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            # info["url"] is a direct URL that ffmpeg/imageio can read
            if "url" in info and info["url"]:
                return info["url"]
            # Some entries store URLs under "entries" (playlists), defend against that
            if "entries" in info and info["entries"]:
                first = info["entries"][0]
                if "url" in first and first["url"]:
                    return first["url"]
    except Exception as e:
        print(f"yt-dlp failed: {e}")

    raise RuntimeError(
        "Could not resolve a direct stream URL via pytube or yt-dlp. "
        "Try another video, or pass --source <localfile> as a fallback."
    )



# -------------- Frame reading (robust) --------------

def frame_iter_ffmpeg(url_or_path: str):
    """
    Prefer imageio-ffmpeg, return (fps, (W,H), iterator).
    Raises if it cannot open.
    """
    reader = iio.get_reader(
        url_or_path,
        format="FFMPEG",
        plugin="ffmpeg",
    )
    meta = reader.get_meta_data()  # may query ffprobe internally
    fps = float(meta.get("fps", 24.0))
    size = meta.get("size", (640, 360))

    def _iter():
        for fr in reader:
            # imageio returns RGB uint8
            yield fr

    return fps, size, _iter()


def frame_iter_opencv(url_or_path: str):
    """
    Fallback: OpenCV VideoCapture (supports files, webcams, some URLs).
    Returns (fps, (W,H), iterator).
    """
    cap = cv2.VideoCapture(url_or_path if not url_or_path.isdigit() else int(url_or_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open source: {url_or_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)

    def _iter():
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            # Convert to RGB to unify downstream
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cap.release()

    return float(fps), (w, h), _iter()


def open_frames(source: str):
    """
    Try ffmpeg (imageio) first; if that fails, fallback to OpenCV.
    Returns (fps, (W,H), iterator).
    """
    # Try ffmpeg first
    try:
        return frame_iter_ffmpeg(source)
    except Exception:
        pass

    # Fallback to OpenCV
    return frame_iter_opencv(source)


# -------------- Video writer (imageio-ffmpeg) --------------

class VideoWriter:
    def __init__(self, path: str, fps: float, size_hw):
        """
        path: output mp4
        fps: frames per second
        size_hw: (H, W) for frames expected (RGB)
        """
        if path is None:
            self.writer = None
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        H, W = size_hw
        self.writer = iio.get_writer(
            path,
            format="FFMPEG",
            codec="libx264",
            fps=fps,
            quality=8,  # reasonable default
        )
        self.size = (H, W)

    def write(self, frame_rgb: np.ndarray):
        if self.writer is None:
            return
        # Expect RGB; imageio wants RGB
        self.writer.append_data(frame_rgb)

    def close(self):
        if self.writer is not None:
            self.writer.close()


# -------------- Main --------------

def parse_args():
    ap = argparse.ArgumentParser(description="Monocular depth demo (MiDaS-small).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", type=str,
                     help="Video path/URL or webcam index (string, e.g. '0').")
    src.add_argument("--youtube", type=str,
                     help="YouTube URL; requires pytube to resolve stream URL.")
    ap.add_argument("--config", type=str, default="configs/depth_midas_infer.yaml",
                    help="Config YAML (device, resize). Optional.")
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="Max seconds to process (approx).")
    ap.add_argument("--stride", type=int, default=1,
                    help="Process every N-th frame.")
    ap.add_argument("--output", type=str, default="outputs/depth_demo.mp4",
                    help="Output MP4 path. Use --no-save to disable writing.")
    ap.add_argument("--no-save", action="store_true", help="Disable writing the output video.")
    ap.add_argument("--preview", action="store_true", help="Live preview window.")
    ap.add_argument("--device", type=str, default=None,
                    help="Override device: cpu|cuda|mps|auto (else from config).")
    return ap.parse_args()


def load_cfg(path: str) -> dict:
    if path and os.path.isfile(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    # Resolve source
    if args.youtube:
        print("Resolving YouTube stream…")
        stream_url = resolve_youtube_direct(args.youtube)
        source = stream_url
    else:
        source = args.source

    # Open frames (prefer ffmpeg)
    fps, (W, H), frames = open_frames(source)
    print(f"[INFO] Source opened: {source}")
    print(f"[INFO] FPS={fps:.2f}  Size={W}x{H}")

    # Init model
    runtime = cfg.get("runtime", {})
    device = (args.device or runtime.get("device", "auto")).lower()
    resize_hw = runtime.get("resize")  # e.g., [384, 384] or None
    dmodel = MiDaSSmall(device=device)
    print(f"[INFO] MiDaS-small on device: {dmodel.device}")

    # Output writer
    out_path = None if args.no_save else args.output
    view_h, view_w = H, W * 2  # side-by-side
    writer = VideoWriter(out_path, fps=fps, size_hw=(view_h, view_w))
    if out_path:
        print(f"[INFO] Writing: {out_path}")

    start = time.time()
    frame_idx = 0
    written = 0

    try:
        for rgb in frames:
            elapsed = time.time() - start
            if elapsed > float(args.seconds):
                break

            if (frame_idx % max(1, args.stride)) != 0:
                frame_idx += 1
                continue

            # Optionally resize input to a fixed inference size
            if resize_hw and isinstance(resize_hw, (list, tuple)) and len(resize_hw) == 2:
                rH, rW = int(resize_hw[0]), int(resize_hw[1])
                rgb_in = cv2.resize(rgb, (rW, rH), interpolation=cv2.INTER_AREA)
            else:
                rgb_in = rgb

            # Inference expects BGR
            bgr_in = cv2.cvtColor(rgb_in, cv2.COLOR_RGB2BGR)
            with torch.no_grad():
                depth = dmodel.infer(bgr_in)  # HxW float32 [0,1]
            depth_color = colorize_depth(depth)  # BGR uint8

            # Build side-by-side view at source size for consistency
            left = cv2.resize(cv2.cvtColor(rgb_in, cv2.COLOR_RGB2BGR), (W, H))
            right = cv2.resize(depth_color, (W, H))
            vis_bgr = np.hstack([left, right])  # BGR
            vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

            if args.preview:
                cv2.imshow("depth_demo (left=RGB, right=Depth)", vis_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            writer.write(vis_rgb)
            written += 1
            frame_idx += 1

    finally:
        writer.close()
        if args.preview:
            cv2.destroyAllWindows()

    print(f"[DONE] Wrote {written} frames." if out_path else f"[DONE] Previewed {written} frames (no-save).")


if __name__ == "__main__":
    main()
