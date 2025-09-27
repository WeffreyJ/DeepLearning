import argparse, torch, onnx
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/midas_small.onnx"); args=ap.parse_args()
    from dlrepo.models.depth.midas_small import MiDaSSmall
    m = MiDaSSmall(device="cpu").model.eval()
    x = torch.randn(1,3,256,256)
    torch.onnx.export(m, x, args.out, opset_version=12, input_names=['input'], output_names=['depth'])
    onnx.load(args.out); print("Exported:", args.out)
if __name__ == "__main__": main()
