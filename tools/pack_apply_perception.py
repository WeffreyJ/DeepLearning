#!/usr/bin/env python3
import argparse, os, subprocess, sys, textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYBIN = sys.executable

def sh(cmd, check=True, cwd=REPO):
    print(cmd)
    return subprocess.run(cmd, shell=True, check=check, cwd=cwd)

def py_compile(paths):
    joined = " ".join(paths)
    if joined:
        sh(f"{PYBIN} -m py_compile {joined}")

def git_commit(message):
    try:
        sh('git add -A')
        sh(f'git commit -m "{message}"')
    except subprocess.CalledProcessError:
        print("git commit skipped (nothing to commit?)")

def write_files(filemap):
    changed=[]
    for rel, content in filemap.items():
        p = REPO / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content.rstrip() + "\n")
        print(f"  + wrote: {p}")
        changed.append(str(p))
    return changed

def ensure_requirements(lines):
    req = REPO / "requirements.txt"
    req.touch()
    t = req.read_text()
    added=False
    for line in lines:
        if line not in t:
            t += ("" if t.endswith("\n") else "\n") + line + "\n"
            added=True
    if added:
        req.write_text(t)
        print(f"  * updated {req}")
        return [str(req)]
    return []

# ---------------- PACKS ----------------

def perception_core_pack(commit=False, run_tests=True):
    files = {
"src/dlrepo/perception/__init__.py": "from .core import PerceptionGraph",
"src/dlrepo/perception/core.py": textwrap.dedent("""\
    from typing import Dict, Any
    import numpy as np, torch

    class PerceptionGraph:
        def __init__(self, *, depth=None, detector=None, tracker=None, navigator=None, device: str = "auto"):
            self.depth = depth; self.detector = detector; self.tracker = tracker; self.navigator = navigator
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            self.device = torch.device(device)

        @torch.no_grad()
        def process_frame(self, rgb: np.ndarray, t: float) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            if self.depth is not None: out["depth"] = self.depth.infer(rgb)
            if self.detector is not None:
                det = self.detector.detect(rgb); out["detections"] = det
                if self.tracker is not None: out["tracks"] = self.tracker.update(det, t)
            if self.navigator is not None and "depth" in out: out["nav"] = self.navigator.step(out["depth"])
            return out
"""),
"src/dlrepo/perception/io_video.py": textwrap.dedent("""\
    from typing import Union, Generator, Tuple
    import cv2, time, numpy as np

    def video_reader(src: Union[str,int]) -> Generator[Tuple[float, np.ndarray], None, None]:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened(): raise RuntimeError(f"Failed to open video source: {src}")
        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok: break
            yield (time.time() - t0), frame  # BGR uint8
        cap.release()
"""),
"src/dlrepo/perception/utils.py": textwrap.dedent("""\
    import numpy as np, cv2

    def colorize_depth(d: np.ndarray) -> np.ndarray:
        d = (d - d.min()) / (d.ptp() + 1e-8)
        cm = cv2.applyColorMap((d*255).astype("uint8"), cv2.COLORMAP_MAGMA)
        return cm  # BGR

    def draw_boxes(img: np.ndarray, dets):
        out = img.copy()
        for x1,y1,x2,y2,score,cls in dets or []:
            cv2.rectangle(out, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            cv2.putText(out, f"{int(cls)}:{score:.2f}", (int(x1), int(y1)-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        return out
"""),
"configs/perceive_depth_track_nav.yaml": textwrap.dedent("""\
    runtime:
      device: auto
      out_dir: outputs/perceive

    depth:
      backend: torchhub
      model: midas_small

    detector:
      backend: torchvision
      conf_th: 0.4

    tracker:
      name: bytetrack_lite
      conf_th: 0.4
      match_th: 0.7
      max_age: 30

    nav:
      near_th: 0.3
"""),
"tools/perceive.py": textwrap.dedent("""\
    import argparse, os, time, imageio.v2 as imageio, numpy as np, torch
    from torch.utils.tensorboard import SummaryWriter
    from dlrepo.perception.core import PerceptionGraph
    from dlrepo.perception.io_video import video_reader
    from dlrepo.perception.utils import colorize_depth, draw_boxes

    def _load_depth(cfg, device):
        from dlrepo.models.depth.midas_small import MiDaSSmall
        return MiDaSSmall(device=device)

    def _load_detector(cfg, device):
        import torchvision
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()
        class Wrapper:
            def __init__(self, m, conf): self.m, self.conf= m, conf
            @torch.no_grad()
            def detect(self, bgr):
                rgb = bgr[..., ::-1]
                t = torch.from_numpy(rgb).float().permute(2,0,1)/255.0
                pred = self.m([t.to(next(self.m.parameters()).device)])[0]
                det=[]
                for b,s,c in zip(pred["boxes"].cpu().numpy(), pred["scores"].cpu().numpy(), pred["labels"].cpu().numpy()):
                    if s>=self.conf: det.append([*b.tolist(), float(s), float(c)])
                return det
        return Wrapper(m, cfg.get("conf_th",0.4))

    def _load_tracker(cfg):
        from dlrepo.tracking.bytetrack import ByteTrackLite
        return ByteTrackLite(conf_th=cfg.get("conf_th",0.3),
                             match_th=cfg.get("match_th",0.7),
                             max_age=cfg.get("max_age",30))

    def _load_nav(cfg):
        from dlrepo.nav.costmap import DepthCostmap
        return DepthCostmap(near_th=cfg.get("near_th",0.3))

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--video", required=True)
        ap.add_argument("--config", default="configs/perceive_depth_track_nav.yaml")
        ap.add_argument("--out", default=None)
        args = ap.parse_args()

        import yaml; cfg = yaml.safe_load(open(args.config))
        device = cfg.get("runtime",{}).get("device","auto")
        out_dir = cfg.get("runtime",{}).get("out_dir","outputs/perceive"); os.makedirs(out_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("saved_models","perceive_tb", time.strftime("%Y%m%d-%H%M%S")))
        vid = args.video
        try: vid = int(vid)
        except: pass

        depth = _load_depth(cfg.get("depth",{}), device)
        det   = _load_detector(cfg.get("detector",{}), device)
        trk   = _load_tracker(cfg.get("tracker",{}))
        nav   = _load_nav(cfg.get("nav",{}))
        g = PerceptionGraph(depth=depth, detector=det, tracker=trk, navigator=nav, device=device)

        out_path = args.out or os.path.join(out_dir, f"perceive_{int(time.time())}.mp4")
        writer_mp4 = imageio.get_writer(out_path, fps=20)

        fps_t0 = time.time(); frames=0
        for t, bgr in video_reader(vid):
            out = g.process_frame(bgr, t)
            vis = bgr.copy()
            if "detections" in out: vis = draw_boxes(vis, out["detections"])
            if "depth" in out:
                dep = colorize_depth(out["depth"])      # BGR
                h = min(vis.shape[0], dep.shape[0])
                vis = np.concatenate([vis[:h], dep[:h]], axis=1)
                writer.add_image("depth/colorized", np.transpose(dep[..., ::-1], (2,0,1)), frames, dataformats="CHW")
            if "nav" in out:
                writer.add_scalar("nav/steer", out["nav"]["steer"], frames)
            writer_mp4.append_data(vis[..., ::-1])  # RGB for imageio

            frames += 1
            if frames % 20 == 0:
                fps = frames / (time.time()-fps_t0 + 1e-6)
                writer.add_scalar("perf/fps", fps, frames)

        writer_mp4.close(); writer.flush(); writer.close()
        print(f"Wrote {out_path}")

    if __name__ == "__main__": main()
"""),
}
    changed = write_files(files)
    changed += ensure_requirements(["opencv-python", "imageio[ffmpeg]"])
    py_compile(changed)
    if run_tests:
        try: sh("pytest -q")
        except subprocess.CalledProcessError: print("pytest failed (continuing)")
    if commit: git_commit("feat(perception): add PerceptionGraph core + IO + perceive runner")

def depth_track_nav_pack(commit=False, run_tests=True):
    files = {
"src/dlrepo/models/depth/__init__.py": "from .midas_small import MiDaSSmall",
"src/dlrepo/models/depth/midas_small.py": textwrap.dedent("""\
    import torch, numpy as np, cv2
    class MiDaSSmall:
        def __init__(self, device="cpu"):
            if device=="auto":
                device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            self.device = torch.device(device)
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Small").to(self.device).eval()
            self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        @torch.no_grad()
        def infer(self, bgr: np.ndarray) -> np.ndarray:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = self.transforms(rgb).to(self.device)
            pred = self.model(t.unsqueeze(0))
            depth = torch.nn.functional.interpolate(pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False).squeeze().cpu().numpy()
            d = (depth - depth.min()) / (depth.ptp() + 1e-8)
            return d
"""),
"src/dlrepo/tracking/__init__.py": "from .bytetrack import ByteTrackLite",
"src/dlrepo/tracking/bytetrack.py": textwrap.dedent("""\
    import numpy as np
    def iou(a, b):
        xx1 = np.maximum(a[0], b[0]); yy1 = np.maximum(a[1], b[1])
        xx2 = np.minimum(a[2], b[2]); yy2 = np.minimum(a[3], b[3])
        w = np.maximum(0., xx2-xx1); h = np.maximum(0., yy2-yy1)
        inter = w*h; ra = (a[2]-a[0])*(a[3]-a[1]); rb = (b[2]-b[0])*(b[3]-b[1])
        return inter / (ra + rb - inter + 1e-9)
    class ByteTrackLite:
        def __init__(self, conf_th=0.3, match_th=0.7, max_age=30):
            self.conf_th, self.match_th, self.max_age = conf_th, match_th, max_age
            self.next_id = 1; self.tracks = {}  # id -> dict(box, cls, score, age)
        def update(self, detections, t):
            dets = [d for d in (detections or []) if d[4] >= self.conf_th]
            for tr in self.tracks.values(): tr["age"] += 1
            used=set()
            for tid, tr in list(self.tracks.items()):
                best=-1; besti=-1
                for i,d in enumerate(dets):
                    if i in used: continue
                    ov = iou(tr["box"], d[:4])
                    if ov>best: best, besti = ov, i
                if best>=self.match_th:
                    d = dets[besti]; used.add(besti)
                    tr["box"], tr["score"], tr["cls"], tr["age"] = d[:4], d[4], d[5], 0
            for i,d in enumerate(dets):
                if i in used: continue
                self.tracks[self.next_id] = {"box": d[:4], "score": d[4], "cls": d[5], "age": 0}
                self.next_id += 1
            self.tracks = {k:v for k,v in self.tracks.items() if v["age"] <= self.max_age}
            out=[]
            for tid,tr in self.tracks.items():
                x1,y1,x2,y2 = tr["box"]
                out.append([tid, x1,y1,x2,y2, tr["score"], tr["cls"]])
            return out
"""),
"src/dlrepo/nav/__init__.py": "from .costmap import DepthCostmap",
"src/dlrepo/nav/costmap.py": textwrap.dedent("""\
    import numpy as np
    class DepthCostmap:
        def __init__(self, near_th=0.3): self.near_th = near_th
        def step(self, depth_norm: np.ndarray):
            mask = (depth_norm < self.near_th).astype(np.uint8)
            h, w = mask.shape
            left = mask[:, : w//2].mean(); right = mask[:, w//2 :].mean()
            steer = float(np.clip(right - left, -1.0, 1.0))
            return {"mask": mask, "steer": steer}
"""),
"tools/depth_infer.py": textwrap.dedent("""\
    import argparse, os, glob, time, numpy as np, imageio.v2 as imageio
    from torch.utils.tensorboard import SummaryWriter
    from dlrepo.models.depth.midas_small import MiDaSSmall
    from dlrepo.perception.utils import colorize_depth
    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--images", nargs="+", help="glob(s) like demo/*.jpg")
        ap.add_argument("--config", default="configs/depth_midas_infer.yaml")
        args = ap.parse_args()
        import yaml; cfg = yaml.safe_load(open(args.config))
        out_dir = cfg.get("output",{}).get("dir", "outputs/depth"); os.makedirs(out_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("saved_models","depth_tb", time.strftime("%Y%m%d-%H%M%S")))
        m = MiDaSSmall(device=cfg.get("runtime",{}).get("device","auto"))
        files=[]
        for pat in (args.images or []): import glob as g; files.extend(g.glob(pat))
        for i, fp in enumerate(sorted(files)):
            import cv2; bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
            d = m.infer(bgr); col = colorize_depth(d)  # BGR
            imageio.imwrite(os.path.join(out_dir, f"depth_{i:03d}.png"), col[..., ::-1])  # RGB
            writer.add_image("depth/colorized", np.transpose(col[..., ::-1], (2,0,1)), i, dataformats="CHW")
        writer.flush(); writer.close(); print(f"Wrote {len(files)} frames to {out_dir}")
    if __name__ == "__main__": main()
"""),
"tools/track_video.py": textwrap.dedent("""\
    import argparse, os, time, numpy as np, imageio.v2 as imageio, torch
    from torch.utils.tensorboard import SummaryWriter
    from dlrepo.perception.io_video import video_reader
    from dlrepo.perception.utils import draw_boxes
    def _load_detector(device, conf=0.4):
        import torchvision
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()
        class Wrapper:
            def __init__(self, m, conf): self.m, self.conf= m, conf
            @torch.no_grad()
            def detect(self, bgr):
                rgb = bgr[..., ::-1]
                t = torch.from_numpy(rgb).float().permute(2,0,1)/255.0
                pred = self.m([t.to(next(self.m.parameters()).device)])[0]
                det=[]
                for b,s,c in zip(pred["boxes"].cpu().numpy(), pred["scores"].cpu().numpy(), pred["labels"].cpu().numpy()):
                    if s>=self.conf: det.append([*b.tolist(), float(s), float(c)])
                return det
        return Wrapper(m, conf)
    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--video", required=True)
        ap.add_argument("--config", default="configs/track_bytetrack.yaml")
        args = ap.parse_args()
        import yaml; cfg = yaml.safe_load(open(args.config))
        device = cfg.get("runtime",{}).get("device","auto")
        device = ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")) if device=="auto" else device
        out_dir = cfg.get("output",{}).get("dir", "outputs/tracks"); os.makedirs(out_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("saved_models","tracks_tb", time.strftime("%Y%m%d-%H%M%S")))
        from dlrepo.tracking.bytetrack import ByteTrackLite
        trk = ByteTrackLite(conf_th=cfg.get("tracker",{}).get("conf_th",0.4))
        det = _load_detector(device, conf=cfg.get("detector",{}).get("conf_th",0.4))
        out_path = os.path.join(out_dir, f"tracks_{int(time.time())}.mp4")
        W = imageio.get_writer(out_path, fps=20)
        frames=0; t0=time.time()
        src = args.video
        try: src=int(src)
        except: pass
        for t, bgr in video_reader(src):
            pred = det.detect(bgr)
            tracks = trk.update(pred, t)
            vis = draw_boxes(bgr, [[x1,y1,x2,y2,score,cls] for _,x1,y1,x2,y2,score,cls in tracks])
            if frames % 10 == 0:
                writer.add_image("tracks/overlay", np.transpose(vis[..., ::-1], (2,0,1)), frames, dataformats="CHW")
            W.append_data(vis[..., ::-1])
            frames += 1
            if frames % 20 == 0:
                fps = frames/ (time.time()-t0 + 1e-6)
                writer.add_scalar("perf/fps", fps, frames)
        W.close(); writer.flush(); writer.close(); print(f"Wrote {out_path}")
    if __name__=="__main__": main()
"""),
"tools/export_onnx_trt.py": textwrap.dedent("""\
    import argparse, torch, onnx
    def main():
        ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/midas_small.onnx"); args=ap.parse_args()
        from dlrepo.models.depth.midas_small import MiDaSSmall
        m = MiDaSSmall(device="cpu").model.eval()
        x = torch.randn(1,3,256,256)
        torch.onnx.export(m, x, args.out, opset_version=12, input_names=['input'], output_names=['depth'])
        onnx.load(args.out); print("Exported:", args.out)
    if __name__ == "__main__": main()
"""),
"configs/depth_midas_infer.yaml": "runtime: { device: auto }\noutput: { dir: outputs/depth }\n",
"configs/track_bytetrack.yaml":  "runtime: { device: auto }\ndetector: { conf_th: 0.4 }\ntracker: { conf_th: 0.4, match_th: 0.7, max_age: 30 }\noutput: { dir: outputs/tracks }\n",
"configs/det_torchvision_frcnn.yaml": "runtime: { device: auto }\ndetector: { conf_th: 0.4 }\n",
}
    changed = write_files(files)
    changed += ensure_requirements(["opencv-python", "imageio[ffmpeg]"])
    py_compile(changed)
    if run_tests:
        try: sh("pytest -q")
        except subprocess.CalledProcessError: print("pytest failed (continuing)")
    if commit: git_commit("feat(perception): depth + tracker + nav tools/configs (inference-ready)")

PACKS = {
    "perception_core_pack": perception_core_pack,
    "depth_track_nav_pack": depth_track_nav_pack,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, choices=PACKS.keys())
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--no-test", action="store_true", help="skip pytest")
    args = ap.parse_args()
    PACKS[args.pack](commit=args.commit, run_tests=not args.no_test)

if __name__ == "__main__":
    main()
