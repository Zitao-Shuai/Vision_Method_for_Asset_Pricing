import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from copy import deepcopy
import copy
import numpy as np
from collections import OrderedDict
ALGORITHMS = [
     'SimpleCNN'
    ,'AR'
    ,'DNN'
    ,'AECNN'
]
def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
class SimpleCNN(nn.Module):
    '''
    This network is used to extract the figure signals.
    window_size     input size      Model
    5d              60 * 60         M1
    20d             120 * 120       M2
    60d             240 * 240       M3  
    '''
    def __init__(self, hparams):
        super(SimpleCNN, self).__init__()
        if hparams['pic_size'] == 120:
            # for 20-days; size: 120
            self.network = nn.Sequential(
                torch.nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 2, padding = 1), # 60 * 60
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 2, padding = 1), # 30 * 30
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1), # 15 * 15
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 3, padding = 1), # 5 * 5
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                )
        elif hparams['pic_size'] == 240:
            # for 60-days; size: 240
            self.network = nn.Sequential(
                torch.nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 2, padding = 1), # 120 * 120
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 2, padding = 1), # 60 * 60
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                
                torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 2, padding = 1), # 30 * 30
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),

                torch.nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 2, padding = 1), # 15 * 15
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                
                torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 3, padding = 1), # 5 * 5
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                )
        else:
                # for 5-day; size = 60 
            self.network = nn.Sequential(
                torch.nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 2, padding = 1), # 30 * 30
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                
                torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 2, padding = 1), # 15 * 15
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                
                torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 3, padding = 1), # 5 * 5
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                )
        self.fc = torch.nn.Linear(32 * 5 * 5, 1)
        m = 0.95
        self.optimizer = torch.optim.SGD(
             self.network.parameters()
            ,lr=hparams["lr"]
            ,momentum = m
        )
        self.optimizer_fc = torch.optim.SGD(
             self.fc.parameters()
            ,lr=hparams["lr"]
            ,momentum = m
        )
        
    def update(self, x, y):
        bs = x.shape[0]
        emb = self.network(x).view((bs,-1))
        y_hat = self.fc(emb)

        loss = ((y - y_hat)**2).mean()
        self.optimizer.zero_grad()
        self.optimizer_fc.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer_fc.step()
        result = {"loss": loss.item()}
        return result
    def predict(self,x):
        bs = x.shape[0]
        emb = self.network(x).view((bs,-1))
        y_hat = self.fc(emb)
        return y_hat
class DNN(nn.Module):
    '''
    This network is used to extract the figure signals.
    Input: batch_size * 5
    feature: open; high; low; close; volume
    '''
    def __init__(self, hparams):
        super(DNN, self).__init__()
        self.network = nn.Sequential(
            torch.nn.Linear(5, 128),
            nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
                
            )
       
        m = 0.95
        self.optimizer = torch.optim.SGD(
             self.network.parameters()
            ,lr=hparams["lr"]
            ,momentum = m
        )
        
        
    def update(self, x, y):
        
        y_hat = self.network(x)
        
        loss = ((y - y_hat)**2).mean()
        self.optimizer.zero_grad()
        
        loss.backward()
        self.optimizer.step()
        
        result = {"loss": loss.item()}
        return result
    def predict(self,x):
        
        y_hat = self.network(x)
        return y_hat
class AECNN(nn.Module):
    '''
    This network is used to extract the figure signals.
    Based on our SimpleCNN, we add a decoder to construct a AutoEncoder structrue.
    window_size     input size      Model
    5d              60 * 60         M1
    20d             120 * 120       M2
    60d             240 * 240       M3  
    
    '''
    def __init__(self, hparams):
        super(AECNN, self).__init__()
        self.hparams = hparams
        if hparams['pic_size'] == 120:
            # for 20-days; size: 120
            self.network = nn.Sequential(
                torch.nn.Conv2d(in_channels = 2, out_channels = 8, kernel_size = 3, stride = 2, padding = 1), # 60 * 60
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels = 8, out_channels =16, kernel_size = 3, stride = 2, padding = 1), # 30 * 30
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1), # 15 * 15
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 3, padding = 1), # 5 * 5
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                )
            self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=3,
                               padding=1,output_padding=2), 
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1,output_padding=1), # 30
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1,output_padding=1), # 60
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1,output_padding=1), # 120
            )
            in_features = 64 * 5 * 5
        elif hparams['pic_size'] == 240:
            # for 60-days; size: 240
            self.network = nn.Sequential(
                torch.nn.Conv2d(in_channels = 2, out_channels = 8, kernel_size = 3, stride = 2, padding = 1), # 120 * 120
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 2, padding = 1), # 60 * 60
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1), # 30 * 30
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1), # 15 * 15
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 3, padding = 1), # 5 * 5
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                )
            self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=3,
                               padding=1,output_padding=2), 
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1,output_padding=1), # 30
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1,output_padding=1), # 60
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1,output_padding=1), # 120
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1,output_padding=1), # 240
            )
            in_features = 128 * 5 * 5
        else:
                # for 5-day; size = 60 
            self.network = nn.Sequential(
                torch.nn.Conv2d(in_channels = 2, out_channels = 8, kernel_size = 3, stride = 2, padding = 1), # 30 * 30
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 2, padding = 1), # 15 * 15
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 3, padding = 1), # 5 * 5
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                )

            self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=3,
                               padding=1,output_padding=2), 
            nn.BatchNorm2d(16),
            nn.ReLU(), 
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1,output_padding=1), # 30
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1,output_padding=1), # 60
            )
            in_features = 32 * 5 * 5
        
        
        self.fc = nn.Linear(in_features, 1)
        self.optimizer_encoder = torch.optim.Adam(
             self.network.parameters()
            ,lr=self.hparams["lr"]
        )
        self.optimizer_fc = torch.optim.Adam(
             self.fc.parameters()
            ,lr=self.hparams["lr"]
        )
        self.optimizer_decoder = torch.optim.Adam(
             self.decoder.parameters()
            ,lr=self.hparams["lr"]
        )
        
    def update(self, x, y):
        bs = x.shape[0]
        emb = self.network(x)
        
        recon_loss = ((self.decoder(emb) - x)**2).mean()
        emb = emb.view((bs,-1))
        
        y_hat = self.fc(emb)
        
        loss = ((y - y_hat)**2).mean() + recon_loss
        self.optimizer_encoder.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_fc.zero_grad()
        loss.backward()
        self.optimizer_encoder.step()
        self.optimizer_decoder.step()
        self.optimizer_fc.step()
        result = {"loss": loss.item()}
        return result
    def predict(self,x):
        bs = x.shape[0]
        emb = self.network(x).view((bs,-1))
        y_hat = self.fc(emb)
        return y_hat
    def reset_opt(self):
        # reset the optimizer to SGD
        m = 0.95
        self.optimizer_encoder = torch.optim.SGD(
             self.network.parameters()
            ,lr=self.hparams["lr"]
            ,momentum = m
        )
        self.optimizer_fc = torch.optim.SGD(
             self.fc.parameters()
            ,lr=self.hparams["lr"]
            ,momentum = m
        )
        self.optimizer_decoder = torch.optim.SGD(
             self.decoder.parameters()
            ,lr=self.hparams["lr"]
            ,momentum = m
        )
class AR:
    '''
    This model is used to run ordinary auto-regression.
    '''
    def __init__(self, hparams):
        self.lag_term = hparams['lag_term']
        self.label = 'rtn'
        self.factor_list = [self.label]
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
    def set_dataset(self, dataset):
        dataset = dataset[self.factor_list]
        for i in range(self.lag_term):
            dataset[self.label + '_lag_'+str(i + 1)] = dataset[self.label].shift(i + 1)
        dataset = dataset.dropna(how = "any")
        return dataset
    def update(self,x,y):
        self.model.fit(x, y)
    def predict(self,x):
        y_hat = self.model.predict(x)
        return y_hat