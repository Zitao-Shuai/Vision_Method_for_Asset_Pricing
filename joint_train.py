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
    # step 1: basic settings for training process
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
    torch.manual_seed(0)
    start_step = 0
    algorithm_dict = None

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # step 2: set the hyparameters
    hparams = {}
    hparams = set_hparams(hparams)
    hparams['pic_size'] = args.pic_size
    hparams['window_size'] = args.window_size
    hparams['overlap'] = args.window_size - 1
    
    # for saving the model
    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))
    
    k_fold = 5
    k_interval = 1.0/k_fold
    # set the list of assets' names
    # step 3: training the model
    for i in range(k_fold):
        if args.algorithm in vars(algorithms):
            # initiate the model
            algorithm = vars(algorithms)[args.algorithm](
                hparams)
        else:
            raise NotImplementedError
        for index, name in enumerate(asset_list):
            # initiate the dataset
            if args.dataset in vars(datasets):
                # path: market + "_" + "Trading/Vol"
                dataset = vars(datasets)[args.dataset](args.data_dir,
                    hparams, name)
            else:
                raise NotImplementedError
            temp_train_dataset, temp_test_dataset = dataset.tr_te_split(start = i * k_interval, end = (i + 1) * k_interval)
            if index == 0:
                dataset.reset_data(temp_train_dataset)
                train_dataset = deepcopy(dataset)
                dataset.reset_data(temp_test_dataset)
                test_dataset = deepcopy(dataset)
                
            else:
                dataset.reset_data(temp_train_dataset)
                temp_train_dataset = deepcopy(dataset)
                dataset.reset_data(temp_test_dataset)
                temp_test_dataset = deepcopy(dataset)
                
                train_dataset.concat(temp_train_dataset)
                
                test_dataset.concat(temp_test_dataset)
        train_dataset.report_data()
        test_dataset.report_data()
        n_epoch = 50
        results = {}
        for epoch in range(n_epoch):
            train_loaders = DataLoader(train_dataset, batch_size = hparams['batch_size'], shuffle=True)
            n_step_train = len(train_dataset)
            step_per_epoch = int(n_step_train / hparams['batch_size'])
            train_minibatches_iterator = iter(train_loaders)
            for n_batch in range(step_per_epoch):
                if args.dataset == 'imgVolOptDataset':
                    x, x_v, y = next(train_minibatches_iterator)
                    x = torch.cat([x, x_v], dim = 1)
                
                
                elif args.dataset == 'imgOptDataset':
                    x, y = next(train_minibatches_iterator)
                
                else:
                    x, y = next(train_minibatches_iterator)
                # return a table
                results_step = algorithm.update(x,y)
                results_step['epoch'] = epoch
                
                for key in results_step.keys():
                    if epoch == 0:
                        results[key] = [results_step[key]]
                    else:
                        results[key].append(results_step[key])
        
            if epoch % 5 == 0:
                test_loaders = DataLoader(test_dataset, batch_size = hparams['batch_size'], shuffle=True)
                test_minibatches_iterator = iter(test_loaders)
                save_checkpoint(f'model_epoch{epoch}.pkl')
                # evaluate the model
                n_step_test = len(test_dataset)
                step_per_epoch = int(n_step_test / hparams['batch_size'])
                acc = evaluator(test_minibatches_iterator, step_per_epoch, algorithm, args, hparams)
                print(acc)
                print(results_step)
        df_results = pd.DataFrame(results)
        df_results.to_csv("./results/" + args.algorithm + "_" + args.dataset + "_" + str(hparams['window_size']) + "_" + str(hparams['price_ratio']) + "_" + str(hparams['p_sect']) + "_fold" + str(i) + '.csv')