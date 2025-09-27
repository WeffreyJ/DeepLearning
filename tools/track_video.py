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
