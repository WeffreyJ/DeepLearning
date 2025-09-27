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
