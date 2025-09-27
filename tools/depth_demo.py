#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import warnings

import cv2
import imageio.v2 as iio
import numpy as np
import torch

from typing import Optional, Tuple, Union, Iterator

from dlrepo.models.depth.midas_small import MiDaSSmall
from dlrepo.perception.utils import colorize_depth


# ------------------------- YouTube URL resolver (yt-dlp) -------------------------
def resolve_youtube_direct(url: str) -> str:
    """
    Use yt-dlp to fetch a direct, streamable URL.
    We prefer progressive MP4 with both audio+video; else accept HLS (m3u8).
    Never return a storyboard/thumbnail URL.
    """
    try:
        from yt_dlp import YoutubeDL
    except Exception as e:
        raise RuntimeError("yt-dlp is required. Install with: pip install yt-dlp") from e

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        # Prefer progressive mp4 (<=720p) + audio, then best mp4, then HLS, else best
        "format": (
            "bestvideo[ext=mp4][vcodec!=none][height<=720]+bestaudio[ext=m4a]/"
            "best[ext=mp4][acodec!=none][vcodec!=none]/"
            "best[protocol^=m3u8]/best"
        ),
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        # Direct URL field on result?
        if isinstance(info, dict):
            direct = info.get("url")
            if direct and "i.ytimg.com" not in direct:
                return direct

            # Scan formats for a suitable candidate
            for cand in (info.get("formats") or []):
                u = cand.get("url")
                if not u or "i.ytimg.com" in u:
                    continue
                proto = (cand.get("protocol") or "")
                ext = (cand.get("ext") or "")
                if "m3u8" in proto or ext in ("mp4", "m3u8"):
                    return u

            # Last resort: any non-thumbnail
            for cand in (info.get("formats") or []):
                u = cand.get("url")
                if u and "i.ytimg.com" not in u:
                    return u

    raise RuntimeError("Could not resolve a playable stream URL via yt-dlp.")


# ------------- Readers (OpenCV for webcam/HLS, ImageIO for file/HTTP) -------------
class Cv2StreamReader:
    """Simple wrapper to iterate frames from OpenCV VideoCapture (for webcam or HLS)."""
    def __init__(self, src: Union[int, str]):
        self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open source with OpenCV: {src}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        self.size = (w, h)

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                break
            yield frame

    def close(self):
        self.cap.release()


def open_source_ffmpeg(source: str):
    """
    For HLS (.m3u8), use OpenCV (more robust). For other URLs/files, use imageio-ffmpeg.
    Returns (handle, fps, (w,h)).
    """
    if "m3u8" in source or source.endswith(".m3u8"):
        r = Cv2StreamReader(source)
        print(f"[INFO] Source opened (HLS via OpenCV): {source}")
        return r, r.fps, r.size

    # Otherwise use imageio-ffmpeg
    reader = iio.get_reader(source, plugin="ffmpeg")
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))
    size = meta.get("size")
    if size is None:
        try:
            frame = reader.get_next_data()
            size = (int(frame.shape[1]), int(frame.shape[0]))
            reader.set_image_index(0)
        except Exception:
            size = (640, 360)
    print(f"[INFO] Source opened: {source}")
    return reader, fps, (int(size[0]), int(size[1]))


def open_source(source: str):
    """
    Open a source which could be a webcam index (e.g., "0"), a file path, or an URL.
    Returns (handle, fps, (w,h)).
    """
    if source.isdigit():
        # Webcam via OpenCV
        cam = Cv2StreamReader(int(source))
        print(f"[INFO] Source opened: {source}")
        print(f"[INFO] FPS={cam.fps:.2f}  Size={cam.size[0]}x{cam.size[1]}")
        return cam, cam.fps, cam.size

    # URLs / local files
    return open_source_ffmpeg(source)


def iter_frames(handle):
    if isinstance(handle, Cv2StreamReader):
        yield from handle
    else:
        # imageio reader
        for frame in handle:
            yield frame


def close_source(handle):
    if isinstance(handle, Cv2StreamReader):
        handle.close()
    else:
        handle.close()


# ------------------------------ utils / visualization ------------------------------
def downscale_keep_aspect(img: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    if not max_side:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def hstack_safe(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pad the shorter image vertically so we can hstack without distortion."""
    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    H = max(lh, rh)

    def pad(img):
        h, w = img.shape[:2]
        if h == H:
            return img
        pad = np.zeros((H - h, w, 3), dtype=img.dtype)
        return np.vstack([img, pad])

    return np.hstack([pad(left), pad(right)])


# --------------------------------------- main ---------------------------------------
def main():
    ap = argparse.ArgumentParser(description="MiDaS depth demo from webcam/file/YouTube.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--source", type=str, help="Webcam index like '0', a file path, or a URL")
    g.add_argument("--youtube", type=str, help="YouTube URL")

    ap.add_argument("--config", type=str, default="configs/depth_midas_infer.yaml",
                    help="(reserved for future use)")
    ap.add_argument("--seconds", type=int, default=20, help="Limit duration (approx)")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    ap.add_argument("--max-size", type=int, default=None,
                    help="Max side length; downscale before inference")
    ap.add_argument("--output", type=str, default="outputs/depth_demo.mp4",
                    help="Output video path")
    ap.add_argument("--no-save", action="store_true", help="Do not write output video")
    ap.add_argument("--preview", action="store_true", help="OpenCV preview window")
    ap.add_argument("--device", type=str, default="auto", help="cpu/cuda/mps/auto")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Resolve source
    if args.youtube:
        print("Resolving YouTube stream…")
        source = resolve_youtube_direct(args.youtube)
    else:
        source = args.source

    # Open stream
    handle, fps, _size = open_source(source)

    # Load model
    m = MiDaSSmall(device=args.device)
    print(f"[INFO] MiDaS-small on device: {m.device.type}")

    # Writer
    writer = None
    if not args.no_save:
        writer = iio.get_writer(args.output, fps=float(max(fps, 1.0)))

    frames_written = 0
    frame_idx = 0
    max_frames = int(fps * args.seconds) if args.seconds else None

    try:
        for frame in iter_frames(handle):
            frame_idx += 1
            if args.stride > 1 and (frame_idx % args.stride != 0):
                continue
            if max_frames and frames_written >= max_frames:
                break

            # Ensure BGR image (imageio gives RGB; OpenCV gives BGR)
            if frame.ndim == 3 and frame.shape[2] == 3:
                # Heuristic: imageio-ffmpeg returns RGB; OpenCV returns BGR.
                # We can detect by trusting source type; simplest is assume RGB from iio.
                if not isinstance(handle, Cv2StreamReader):
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    bgr = frame
            else:
                # Greatly uncommon, but guard anyway
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Optional downscale before inference
            bgr_small = downscale_keep_aspect(bgr, args.max_size)

            # Depth inference -> [0,1] float map
            with torch.no_grad():
                d = m.infer(bgr_small)

            # Colorize and compose side-by-side
            col = colorize_depth(d)  # BGR
            vis = hstack_safe(bgr_small, col)

            # Preview
            if args.preview:
                cv2.imshow("depth_demo", vis)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

            # Save
            if writer:
                writer.append_data(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                frames_written += 1

    finally:
        close_source(handle)
        if writer:
            writer.close()
        if args.preview:
            cv2.destroyAllWindows()

    print(f"[DONE] Wrote {frames_written} frames." if writer else "[DONE] Preview only (no-save).")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
