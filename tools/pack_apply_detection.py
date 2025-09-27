#!/usr/bin/env python3
import argparse, os, sys, subprocess, textwrap, pathlib, shutil

REPO = pathlib.Path(__file__).resolve().parents[1]
PYBIN = sys.executable

def sh(cmd, check=True, cwd=REPO):
    print(cmd)
    return subprocess.run(cmd, shell=True, check=check, cwd=cwd)

def write(path: pathlib.Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"  + wrote: {path}")

def ensure_line(path: pathlib.Path, needle: str):
    if not path.exists():
        path.write_text(needle + "\n", encoding="utf-8")
        print(f"  * created {path} with '{needle}'")
        return
    txt = path.read_text(encoding="utf-8").splitlines()
    if not any(line.strip() == needle for line in txt):
        txt.append(needle)
        path.write_text("\n".join(txt) + "\n", encoding="utf-8")
        print(f"  * updated {path} (+ {needle})")

def py_compile(paths):
    joined = " ".join(str(p) for p in paths)
    sh(f"{PYBIN} -m py_compile {joined}")

# ------------------------ FILE CONTENTS ------------------------

DETECT_CFG = """\
# configs/detect_yolo.yaml
runtime:
  device: auto             # auto | cuda | mps | cpu
  conf_thr: 0.25
  iou_thr: 0.45
  max_det: 300
  model: yolov8n           # ultralytics model name or path (yolov8n/yolov11n/...)
"""

YOLO_WRAPPER = """\
# src/dlrepo/detection/yolo.py
from __future__ import annotations
import torch, numpy as np
from typing import List, Tuple, Optional, Dict

class YOLODetector:
    \"\"\"Thin wrapper around Ultralytics YOLO for consistent outputs.

    Outputs: list of dicts with keys:
      - 'xyxy': np.ndarray [4] (x1,y1,x2,y2)
      - 'conf': float
      - 'cls':  int
      - 'name': str
    \"\"\"
    def __init__(
        self,
        model_name: str = "yolov8n",
        device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
    ):
        if device == "auto":
            device = ("cuda" if torch.cuda.is_available() else
                      ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                              else "cpu"))
        self.device = device

        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("Ultralytics not installed. Try: pip install ultralytics") from e

        self.model = YOLO(model_name)
        # Send to device if possible (YOLO handles it, but set attribute for clarity)
        self.model.to(self.device)

        self.conf = float(conf)
        self.iou  = float(iou)
        self.max_det = int(max_det)

        # Class names (dict: id->name)
        self.names = self.model.model.names if hasattr(self.model, "model") else getattr(self.model, "names", {})
        if isinstance(self.names, list):
            self.names = {i:n for i,n in enumerate(self.names)}

    @torch.no_grad()
    def infer(self, bgr: np.ndarray) -> List[Dict]:
        \"\"\"Run detection on a single BGR frame (H,W,3 uint8).\"\"\"
        # Ultralytics expects RGB; it converts internally too, but we keep explicit & fast
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        res = self.model.predict(
            source=rgb,
            verbose=False,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            imgsz=None,   # infer from input shape
        )
        out = []
        if not res:
            return out
        r0 = res[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:  # no detections
            return out

        # boxes.xyxy (N,4), boxes.conf (N,1), boxes.cls (N,1)
        xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes, "xyxy") else np.zeros((0,4))
        conf = boxes.conf.detach().cpu().numpy().reshape(-1) if hasattr(boxes, "conf") else np.zeros((0,), dtype=np.float32)
        cls  = boxes.cls.detach().cpu().numpy().astype(int).reshape(-1) if hasattr(boxes, "cls") else np.zeros((0,), dtype=np.int32)

        for i in range(xyxy.shape[0]):
            cid = int(cls[i])
            out.append({
                "xyxy": xyxy[i].astype(np.float32),
                "conf": float(conf[i]),
                "cls":  cid,
                "name": str(self.names.get(cid, str(cid))),
            })
        return out
"""

DETECT_DEMO = """\
# tools/detect_demo.py
#!/usr/bin/env python3
import argparse, os, sys, warnings
import numpy as np, cv2, imageio.v2 as iio
from typing import Optional
import yaml

from dlrepo.detection.yolo import YOLODetector

# ---------- utilities (open source) ----------
def resolve_youtube_direct(url: str) -> str:
    \"\"\"Resolve a streamable URL via yt-dlp (mp4 or m3u8).\"\"\"
    try:
        from yt_dlp import YoutubeDL
    except Exception as e:
        raise RuntimeError("yt-dlp is required (pip install yt-dlp)") from e

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "format": (
            "bestvideo[ext=mp4][vcodec!=none][height<=720]+bestaudio[ext=m4a]/"
            "best[ext=mp4][acodec!=none][vcodec!=none]/"
            "best[protocol^=m3u8]/best"
        ),
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if isinstance(info, dict) and info.get("url") and "i.ytimg.com" not in info["url"]:
            return info["url"]
        for cand in (info.get("formats") or []):
            u = cand.get("url")
            if u and "i.ytimg.com" not in u:
                return u
    raise RuntimeError("Could not resolve a playable URL for YouTube.")

def open_source(source: str):
    \"\"\"Return (reader, fps, (w,h)). Use imageio-ffmpeg for files/urls; OpenCV for webcams.\"\"\"
    if source.isdigit():  # webcam index
        cam = cv2.VideoCapture(int(source))
        if not cam.isOpened():
            raise FileNotFoundError(f"Cannot open webcam index {source}")
        fps = cam.get(cv2.CAP_PROP_FPS) or 30.0
        w   = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h   = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        print(f"[INFO] Source opened (webcam): {source}")
        print(f"[INFO] FPS={fps:.2f}  Size={w}x{h}")
        return cam, float(fps), (w, h)

    # For files / URLs
    reader = iio.get_reader(source, plugin="ffmpeg")
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))
    size = meta.get("size", None)
    print(f"[INFO] Source opened: {source}")
    if size is None:
        try:
            frame = reader.get_next_data()
            size = (frame.shape[1], frame.shape[0])
        except Exception:
            size = (640, 360)
    return reader, fps, size

def iter_frames(handle):
    if isinstance(handle, cv2.VideoCapture):
        while True:
            ok, frame = handle.read()
            if not ok or frame is None: break
            yield frame
    else:
        for frame in handle:
            yield frame

def close_source(handle):
    if isinstance(handle, cv2.VideoCapture):
        handle.release()
    else:
        handle.close()

def downscale_keep_aspect(img, max_side: Optional[int]):
    if not max_side: return img
    h, w = img.shape[:2]; m = max(h, w)
    if m <= max_side: return img
    scale = max_side / float(m)
    nh, nw = int(round(h*scale)), int(round(w*scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def draw_boxes(bgr, dets, thickness=2):
    out = bgr.copy()
    for d in dets:
        x1,y1,x2,y2 = d["xyxy"].astype(int).tolist()
        conf = d["conf"]; name = d["name"]
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), thickness)
        cv2.putText(out, f"{name} {conf:.2f}", (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return out

def hstack_safe(left, right):
    lh, lw = left.shape[:2]; rh, rw = right.shape[:2]
    H = max(lh, rh)
    def pad(img):
        h, w = img.shape[:2]
        if h == H: return img
        pad = np.zeros((H-h, w, 3), dtype=img.dtype)
        return np.vstack([img, pad])
    return np.hstack([pad(left), pad(right)])

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--source",  type=str, help="Webcam index like '0' or file/URL")
    g.add_argument("--youtube", type=str, help="YouTube URL")
    ap.add_argument("--config",  type=str, default="configs/detect_yolo.yaml")
    ap.add_argument("--seconds", type=int, default=15)
    ap.add_argument("--stride",  type=int, default=1, help="Process every Nth frame")
    ap.add_argument("--max-size",type=int, default=None, help="Downscale before inference")
    ap.add_argument("--output",  type=str, default="outputs/detect_demo.mp4")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    runtime = cfg.get("runtime", {})
    device  = runtime.get("device", "auto")
    conf    = float(runtime.get("conf_thr", 0.25))
    iou     = float(runtime.get("iou_thr", 0.45))
    max_det = int(runtime.get("max_det", 300))
    model   = runtime.get("model", "yolov8n")

    if args.youtube:
        print("Resolving YouTube stream…")
        source = resolve_youtube_direct(args.youtube)
    else:
        source = args.source

    handle, fps, _ = open_source(source)
    max_frames = int(fps * args.seconds) if args.seconds else None

    det = YOLODetector(model_name=model, device=device, conf=conf, iou=iou, max_det=max_det)
    print(f"[INFO] YOLO model '{model}' on device: {det.device}")

    writer = None
    frames_written = 0
    if not args.no_save:
        from imageio.v2 import get_writer
        writer = get_writer(args.output, fps=max(1, int(round(fps/args.stride))), codec="libx264", quality=8)
        print(f"[INFO] Writing: {args.output}")

    try:
        frame_idx = 0
        for frame in iter_frames(handle):
            frame_idx += 1
            if args.stride > 1 and (frame_idx % args.stride != 0):
                continue
            if max_frames and frames_written >= max_frames:
                break

            bgr = frame
            bgr_small = downscale_keep_aspect(bgr, args.max_size)
            dets = det.infer(bgr_small)

            vis = draw_boxes(bgr_small, dets)
            side = hstack_safe(bgr_small, vis)

            if args.preview:
                cv2.imshow("detect_demo", side)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if writer:
                import cv2 as _cv
                writer.append_data(_cv.cvtColor(side, _cv.COLOR_BGR2RGB))
                frames_written += 1
    finally:
        close_source(handle)
        if writer:
            writer.close()
        if args.preview:
            cv2.destroyAllWindows()

    print(f"[DONE] Wrote {frames_written} frames." if writer else "[DONE] Preview only.")
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
"""

# ------------------------ PACK STEPS ------------------------

def pack_yolo_detect(commit: bool):
    changed = []

    # files
    write(REPO / "configs" / "detect_yolo.yaml", DETECT_CFG); changed.append(REPO / "configs" / "detect_yolo.yaml")
    write(REPO / "src" / "dlrepo" / "detection" / "__init__.py", "from .yolo import YOLODetector"); changed.append(REPO / "src" / "dlrepo" / "detection" / "__init__.py")
    write(REPO / "src" / "dlrepo" / "detection" / "yolo.py", YOLO_WRAPPER); changed.append(REPO / "src" / "dlrepo" / "detection" / "yolo.py")
    write(REPO / "tools" / "detect_demo.py", DETECT_DEMO); changed.append(REPO / "tools" / "detect_demo.py")
    os.chmod(REPO / "tools" / "detect_demo.py", 0o755)

    # requirements
    ensure_line(REPO / "requirements.txt", "ultralytics")
    ensure_line(REPO / "requirements.txt", "yt-dlp")

    # compile quick sanity (py only)
    py_compile([REPO / "src" / "dlrepo" / "detection" / "yolo.py",
                REPO / "tools" / "detect_demo.py"])

    if commit:
        sh("git add -A")
        sh('git commit -m "feat(detect): YOLO wrapper + detect demo + sample config"')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, choices=["yolo_detect"], help="Which pack to apply")
    ap.add_argument("--commit", action="store_true", help="git commit after applying")
    args = ap.parse_args()
    if args.pack == "yolo_detect":
        pack_yolo_detect(commit=args.commit)

if __name__ == "__main__":
    main()
