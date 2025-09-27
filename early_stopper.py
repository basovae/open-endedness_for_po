class EarlyStopper:
    '''Class to monitor validation loss during training and stop the training process early if no improvement is observed.'''
    def __init__(
        self,
        patience: int = 1,
        min_delta: float = 0,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def early_stop(self, loss, verbose: bool = True):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss >= (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                if verbose:
                    print(f'Early stopping triggered.')
                return True
        return False