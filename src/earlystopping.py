import torch
import numpy as np


class EarlyStopping:
    def __init__(self,
                 patience=5,
                 delta=0,
                 flag='transformer_model',
                 path='checkpoint.pt'):
        
        '''
        Description: Early stopping if the dev loss does not decrease by delta within a given patience.

        Arguments:
        ----------
        patience: (int) defaults to 5. Number of iterations to wait before terminating the training if dev loss doesn't decrease by atleast delta.
        delta: (int) defaults to 0.
        flag: (str) defaults to transformer_model. Required for saving trained models into certain format.
        path: (str) defaults to checkpoint.pt , path to save the best model after every iteration.
        '''
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.flag = flag
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.flag == 'transformer_model':
            model.save_pretrained(self.path)
        else:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        