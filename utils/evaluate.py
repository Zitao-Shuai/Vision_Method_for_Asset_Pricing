import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

def evaluator(test_minibatches_iterator, step_per_epoch, algorithm, args, hparams, mode = "reg"):
    # calculate the out-of-sample accuracy
    acc = 0
    for n_batch in range(step_per_epoch):
        if args.dataset == 'imgVolOptDataset':
            x, x_v, y = next(test_minibatches_iterator)
            x = torch.cat([x, x_v], dim = 1)
            y_hat = algorithm.predict(x)
        elif args.dataset == 'imgOptDataset':
            x, y = next(test_minibatches_iterator)
            y_hat = algorithm.predict(x)
        else:
            x, y = next(test_minibatches_iterator)
            y_hat = algorithm.predict(x)
        if mode == "clf":
            acc = acc + F.cross_entropy(y_hat, y) / step_per_epoch
        else:
            acc = acc + ((y - y_hat)**2).mean() / step_per_epoch

    return acc