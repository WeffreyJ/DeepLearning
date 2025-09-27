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
