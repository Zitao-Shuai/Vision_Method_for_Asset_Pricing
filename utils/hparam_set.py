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
def set_hparams(hparams):
    '''
    hparams: dict
    Used to set hyperparameters for each models.
    '''
    hparams['batch_size'] = 128
    hparams['lr'] = 1e-5
    hparams['start_date'] = '2005-12-31'
    hparams['end_date'] = '2018-12-31'
    hparams['frequency'] = '1d'
    hparams['asset_code'] = '510050.XSHE' # local of no use
    hparams['source'] = 'local' # local of no use
    
    
    hparams['overlap'] = 19 # no use now
    hparams['window_size'] =20
    
    
    hparams['p_sect'] = 180
    hparams['price_ratio'] = 0.6
    hparams['pic_size'] = 120 
    return hparams