import argparse, pandas as pd, matplotlib.pyplot as plt
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); args = ap.parse_args()
    df = pd.read_csv(args.csv)
    plt.figure(); plt.plot(df.epoch, df.train_loss, label='train_loss'); plt.plot(df.epoch, df.val_loss, label='val_loss'); plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.title('Loss vs Epoch'); plt.show()
    plt.figure(); plt.plot(df.epoch, df.train_acc, label='train_acc'); plt.plot(df.epoch, df.val_acc, label='val_acc'); plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend(); plt.title('Accuracy vs Epoch'); plt.show()
if __name__ == "__main__":
    main()
