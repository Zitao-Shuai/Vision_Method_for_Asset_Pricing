import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
from copy import deepcopy
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
asset_list = [
        '50SH',
        '600016',
        '600030',
        '600031',
        '600028'
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
    
    # step4: select the models
    if args.algorithm in vars(algorithms):
        algorithm = vars(algorithms)[args.algorithm](
            hparams)
    else:
        raise NotImplementedError
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    
    for i in range(5):
        if args.algorithm in vars(algorithms):
            algorithm = vars(algorithms)[args.algorithm](
                hparams)
        else:
            raise NotImplementedError
        for index, name in enumerate(asset_list):
            # merge the dataset from different assets
            if args.dataset in vars(datasets):
                # path: market + "_" + "Trading/Vol"
                dataset = vars(datasets)[args.dataset](args.data_dir,
                    hparams, name)
            else:
                raise NotImplementedError    
            dataset = algorithm.set_dataset(dataset.dataframe)
            dataset_use = dataset.reset_index()
            for index, (train_index, test_index) in enumerate(kf.split(dataset)):
                if index == i:
                    break
            
            temp_train_x, temp_train_y = dataset_use.loc[train_index, dataset.columns.to_list()], dataset_use.loc[train_index, algorithm.label]
            del temp_train_x[algorithm.label]
        
            temp_test_x, temp_test_y = dataset_use.loc[test_index, dataset.columns.to_list()], dataset_use.loc[test_index, algorithm.label]
            del temp_test_x[algorithm.label]
            if index == 0:
                train_x = temp_train_x
                test_x = temp_test_x
                train_y = temp_train_y
                test_y = temp_test_y
            else:
                train_x = pd.concat([train_x, temp_train_x], axis = 0)
                train_y = pd.concat([train_y, temp_train_y], axis = 0)
                test_x = pd.concat([test_x, temp_test_x], axis = 0)
                test_y = pd.concat([test_y, temp_test_y], axis = 0)
        algorithm.update(train_x, train_y)
        y_hat = algorithm.predict(test_x)
        loss = ((y_hat - test_y)**2).mean()
        print("mse loss:", loss)
    
    