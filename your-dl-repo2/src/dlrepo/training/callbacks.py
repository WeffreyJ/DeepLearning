class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience; self.best=None; self.epochs_no_improve=0; self.should_stop=False
    def step(self, metric: float):
        if self.best is None or metric > self.best:
            self.best = metric; self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.should_stop = True
