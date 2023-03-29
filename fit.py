import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from picAsset.utils.evaluate import evaluator
from picAsset.utils import datasets
from picAsset.utils import algorithms
from picAsset.utils.hparam_set import set_hparams
DATASETS = [
     'assetDataset'
    ,'imgDataset'
    ,'imgVolOptDataset'
    ,'imgOptDataset'
]
ALGORITHMS = [
     'SimpleCNN'
    ,'AR'
    ,'DNN'
    ,'AECNN'
]

if __name__ == "__main__":
    # step1: basic settings for training process
    parser = argparse.ArgumentParser(description='Asset Pricing')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--algorithm', type=str, default="SimpleCNN")
    parser.add_argument('--dataset', type=str, default="assetDataset")
    parser.add_argument('--pic_size', type=int, default=120)
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--market', type=str, default="SPX")
    parser.add_argument('--train_split', type=float, default = 0.8)
    parser.add_argument('--output_dir', type=str, default="./picAsset/output")
    parser.add_argument('--skip_model_save', type=bool, default=False)
    args = parser.parse_args()
    start_step = 0
    algorithm_dict = None

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))
    device = "cpu"
    # step2: set the hyparameters
    hparams = {}
    hparams = set_hparams(hparams)
    hparams['pic_size'] = args.pic_size
    hparams['window_size'] = args.window_size
    hparams['overlap'] = args.window_size - 1
    # step3: select the dataset
    if args.dataset in vars(datasets):
        # path: market + "_" + "Trading/Vol"
        dataset = vars(datasets)[args.dataset](args.data_dir,
            hparams, args.market)
    else:
        raise NotImplementedError
    # step4: select the models
    if args.algorithm in vars(algorithms):
        algorithm = vars(algorithms)[args.algorithm](
            hparams)
    else:
        raise NotImplementedError
    dataset = algorithm.set_dataset(dataset.dataframe)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    dataset_use = dataset.reset_index()
    for i, (train_index, test_index) in enumerate(kf.split(dataset)):
        if args.algorithm in vars(algorithms):
            algorithm = vars(algorithms)[args.algorithm](
                hparams)
        else:
            raise NotImplementedError
        train_x, train_y = dataset_use.loc[train_index, dataset.columns.to_list()], dataset_use.loc[train_index, algorithm.label]
        del train_x[algorithm.label]
        
        test_x, test_y = dataset_use.loc[test_index, dataset.columns.to_list()], dataset_use.loc[test_index, algorithm.label]
        del test_x[algorithm.label]
        algorithm.update(train_x, train_y)
        y_hat = algorithm.predict(test_x)
        loss = ((y_hat - test_y)**2).mean()
        print("mse loss:", loss)
    
    