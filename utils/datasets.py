import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from torch.utils.data import Dataset
import datetime

DATASETS = [
     'assetDataset'
    ,'imgDataset'
    ,'imgVolOptDataset'
    ,'imgOptDataset'
]
CN_list = [
    '50SH'
]
def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]
def load_data(path, factor_list, hparams, market = 'SPX', start_date='2015-12-31', end_date = '2020-12-31', frequency='1d', asset_code = "510050.XSHE", source = "ricequant"):
    '''
    Function: 
        Load the data for later use.
        Mode:
            local: from the local memory. path + market + "_" + "Trading/Vol" 
            other APIs.
    Input：
        (str)path: the path of the data
        (list)factor_list: the columns that should be selected.
        (dict)hparams: the dictionary of the hyperparameters.
        (str)source: the mode of loading data.
        others: the settings of loading data from APIs.
    Output:
        (DataFrame)data_raw: return the processed dataframe
    '''
    
    if source == "wind":
        # Not implemented, since wind is too expensive.
        pass
    elif source == "local":
        # path: market + "_" + "Trading/Vol"
        data_raw = pd.read_excel(path)
       
        data_raw['date'] = pd.to_datetime(data_raw['date'])
        
        flag_inverse = 0
        
        if data_raw.loc[0,'date']>data_raw.loc[1,'date']:
            flag_inverse = 1
        data_raw = data_raw.set_index(['date'])
        data_raw = data_raw[factor_list]
        
        data_raw['volume'] = data_raw.apply(lambda x: float(str(x['volume']).replace(",","")),axis = 1)
        data_raw['open'] = data_raw.apply(lambda x: float(str(x['open']).replace(",","")),axis = 1)
        data_raw['high'] = data_raw.apply(lambda x: float(str(x['high']).replace(",","")),axis = 1)
        data_raw['low'] = data_raw.apply(lambda x: float(str(x['low']).replace(",","")),axis = 1)
        data_raw['close'] = data_raw.apply(lambda x: float(str(x['close']).replace(",","")),axis = 1)
        data_raw['lag1'] = data_raw['close'].shift(-1)
        data_raw = data_raw.dropna(how="any")
        data_raw['rtn'] = 252 * (data_raw['lag1'] - data_raw['close']) / data_raw['close'] # t+1 return
        data_raw['HV1'] = data_raw['rtn'].rolling(20).std()
        data_raw['HV1_lag1'] = data_raw['HV1'].shift(1)
        data_raw['d_HV1'] = (data_raw['HV1'] - data_raw['HV1_lag1']) / data_raw['HV1_lag1']
        
        data_raw = data_raw.dropna(how="any")
        if flag_inverse == 0:
            data_raw = data_raw.iloc[::-1] 
        data_raw = data_raw.loc[datetime.datetime.strptime(end_date, "%Y-%m-%d"):]
        temp = data_raw.loc[datetime.datetime.strptime(start_date, "%Y-%m-%d"):]
        
        true_index = temp.index[hparams['window_size']-2]
        data_raw = data_raw.loc[:true_index]
        data_raw = data_raw.iloc[::-1] 
        
        return data_raw
    else:
        from rqdatac import get_price,init,options
        import rqdatac
        rqdatac.init('xxx','xxx')
        data_raw = get_price(asset_code, frequency = frequency,start_date = start_date, end_date = end_date).reset_index()
        del data_raw['order_book_id']
        data_raw['date'] = pd.to_datetime(data_raw['date'])
        flag_inverse = 0
        if data_raw.loc[0,'date']>data_raw.loc[0,'date']:
            flag_inverse = 1
        data_raw = data_raw.set_index(['date'])
        data_raw = data_raw[factor_list]
        data_raw['volume'] = data_raw.apply(lambda x: np.log(float(str(x['volume']).replace(",",""))),axis = 1)
        data_raw['open'] = data_raw.apply(lambda x: np.log(float(str(x['open']).replace(",",""))),axis = 1)
        data_raw['high'] = data_raw.apply(lambda x: np.log(float(str(x['high']).replace(",",""))),axis = 1)
        data_raw['low'] = data_raw.apply(lambda x: np.log(float(str(x['low']).replace(",",""))),axis = 1)
        data_raw['close'] = data_raw.apply(lambda x: np.log(float(str(x['close']).replace(",",""))),axis = 1)
        
        data_raw['lag1'] = data_raw['close'].shift(-1)
        data_raw = data_raw.dropna(how="any")
        data_raw['rtn'] = (data_raw['lag1'] - data_raw['close']) / data_raw['close'] # future return
        data_raw['HV1'] = data_raw['rtn'].rolling(20).std()
        data_raw['HV1_lag1'] = data_raw['HV1'].shift(1)
        data_raw['d_HV1'] = (data_raw['HV1'] - data_raw['HV1_lag1']) / data_raw['HV1_lag1']
        data_raw = data_raw.dropna(how="any")
        if flag_inverse == 1:
            data_raw = data_raw.iloc[::-1]
        data_raw = data_raw.loc[datetime.datetime.strptime(end_date, "%Y-%m-%d"):]
        temp = data_raw.loc[datetime.datetime.strptime(start_date, "%Y-%m-%d"):]
        true_index = temp.index[hparams['window_size']-2]
        data_raw = data_raw.loc[:true_index]
        
        
        return data_raw

class assetDataset(Dataset):
    def __init__(self, path, hparams, market = 'SPX'):
        '''
        Function: 
            The base class of the assetDataset. The returned data record is structural data.
            The self.dataframe is used to provide the time-index, for uniformity, we don't use the data stored in it.
            The data: series(self.srs), label(self.label), dataframe(self.dataframe) 
        Input：
            (str)path: the path of the data.
            (dict)hparams: the dictionary of the hyperparameters.
            (str)market: the name of the asset.
        '''
        start_date = hparams['start_date']
        end_date = hparams['end_date']
        frequency = hparams['frequency']
        asset_code = hparams['asset_code']
        source = hparams['source']
        self.hparams = hparams
        
        self.factor_list = [
            'open'
            ,'close'
            ,'high'
            ,'low'
            ,'volume'
            ,'limit_up'
            ,'limit_down'
        ] # raw
        
        if market in CN_list:
            self.factor_list.remove('limit_up')
            self.factor_list.remove('limit_down')
 
        # path: market + "_" + "Trading/Vol"
        path = path + market + "_Trading.xlsx"
        print("path:",path)
        self.dataframe = load_data(   path, self.factor_list, hparams, market
                                    , start_date = start_date
                                    , end_date = end_date
                                    , frequency = frequency
                                    , asset_code = asset_code
                                    , source = source)
        self.label = self.dataframe['rtn'].tolist()
        self.srs = self.dataframe[self.factor_list].values.tolist()
        for i in range(len(self.srs)):
            self.srs[i] = torch.tensor(self.srs[i])
        print(self.srs[0].shape)
    def tr_te_split(self, start, end):
        '''
        Function: 
            Split the dataset in a manual setting way. (Mainly for K-fold)
        Input：
            (float)start: the path of the data.
            (float)end: the dictionary of the hyperparameters.
            The end should larger than the start.
        '''
        # start: ratio, end: ratio. for the testing data
        train_split = {'dataframe': pd.concat([self.dataframe.iloc[0:int(start * self.dataframe.shape[0]),:],self.dataframe.iloc[int(end * self.dataframe.shape[0]):,:]],axis = 0)}
        train_split['label'] = self.label[0: int(start * len(self.label))] + self.label[int(end * len(self.label)):]
        train_split['srs'] = self.srs[0: int(start * len(self.srs))] + self.srs[int(end * len(self.srs)):]
        test_split = {'dataframe':self.dataframe.iloc[int(start * self.dataframe.shape[0]):int(end * self.dataframe.shape[0]),:]}
        test_split['label'] = self.label[int(start * len(self.label)):int(end * len(self.label))]
        test_split['srs'] = self.srs[int(start * len(self.srs)):int(end * len(self.srs))]
        return train_split, test_split
    def reset_data(self, data):
        '''
        Function:
            Reset the data of the object.
        Input:
            (dict)data: a dictionary of the data of the corresponding class.
        '''
        self.dataframe = data['dataframe']
        self.srs = data['srs']
        self.label = data['label']
    def concat(self, dataset):
        '''
        Function:
            concat the data from two objects
        Input:
            (assetDataset)dataset: another assetDataset object.
        '''
        
        self.srs = self.srs + dataset.srs
        self.label = self.label + dataset.label
        self.dataframe = pd.concat([self.dataframe, dataset.dataframe],axis = 0) # maybe the index will overlap
    def report_data(self):
        print(self.dataframe.index)
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self,idx):
        
        label = self.label[idx]
        data = self.srs[idx]
        return data, label
def Gen_Image(data, p_sect, hparams, interval = 20, overlap = 19, ratio = 0.8, is_save = 0, item = "SPX"):
    '''
    Function: 
        Generate pictures from the given dataframe.
        The details of the method are shown in my working paper.
    Input：
        (DataFrame)data: the dataframe we used to generate pictures.
        (int)interval: time interval, daily. It is initially set 20 which represent a month's trading days.
        (str)item: the name of the item
        (float)ratio: the ratio of the size of the graph of the prices
        (dict)hparams: the dictionary of the hyperparameters.
        (int)is_save: 1: save locally. 0: add into list.
    Output:
        (list)img_list: return the list of processed pictures.
    Generate the pixel image.
    Note:
    1.Each day has a corresponding 3 units
    2.The upper line of the image: the maximum of the high price among all of the datapoints.
    3.The lower line of the image: the minimum of the low price among all of the datapoints.  
    4.high-low price: bar
    5.open-close: left-right side
    '''
    # for price
    max_price = data[['high','close','open','close']].max().max()
    min_price = data[['high','close','open','close']].min().min()
    delta_p = (max_price - min_price)
    # for volume
    max_volume = data['volume'].max()         
    min_volume = data['volume'].min()
    delta_v = (max_volume - min_volume)
    # for graph size
    wide = interval * 3
    height = p_sect
    count = 0
    index = 0
    img_list = []
    
    while index < data.shape[0]:
        if (count % interval) == 0:
            
            img = np.zeros((wide, height))
            count = 0
            index = index - overlap
        date = data.index[index]
        # pixel the points
        # picture for prices
        y_open = int((ratio * (data.loc[date, 'open'] - min_price) * height / delta_p + (1 - ratio) * height))
        if y_open >= height:
            print("y_open",y_open)
            y_open = height-1
        img[(index % interval) * 3, y_open] = 1
        y_close = int((ratio * (data.loc[date, 'close'] - min_price) * height / delta_p + (1 - ratio) * height))
        img[(index % interval) * 3 + 2, y_close] = 1
        if int(data.loc[date, 'high'] * height / delta_p) >= int(data.loc[date, 'low'] * height / delta_p):
            for i in range(1+int((data.loc[date, 'high'] * height - data.loc[date, 'low'] * height) / delta_p)):
                y_HL = i + int((ratio * (data.loc[date, 'low'] - min_price) * height / delta_p + (1 - ratio) * height))
                if y_HL >= height:
                    y_HL = height - 1
                    
                img[(index % interval) * 3 + 1, y_HL] = 1
        
        # piture for volume
        for i in range(int((1 - ratio) * (data.loc[date, 'volume'] - min_volume) * height / delta_v)):
            y_v = i
            img[(index % interval) * 3 + 1, y_v] = 1
        
        
        if count % interval == interval-1:
            # transform to tensor or save
            if is_save == 1:
                img = img.T
                im = Image.fromarray(img.astype(np.uint8))
                transform_ts = torchvision.transforms.Compose([  
                    transforms.ToTensor()])
                img = transform_ts(im)
                resize = transforms.Resize([hparams['pic_size'],hparams['pic_size']])
                img = resize(img)
                im.save("./Image/"+str(item)+"_"+str(int(index))+"_"+str(interval)+"_micro.png")
            else:
                img = img.T
                im = Image.fromarray(img.astype(np.uint8))
                transform_ts = torchvision.transforms.Compose([  
                    transforms.ToTensor()])
                img = transform_ts(im)
                resize = transforms.Resize([hparams['pic_size'],hparams['pic_size']])
                img = resize(img)
                img_list.append(img)
        count = count + 1
        index = index + 1
    return img_list
class imgDataset(assetDataset):
    def __init__(self, path, hparams, market):
        '''
        Function:
            Inheritated from Class assetDataset. The returned data record is the processed picture.
            The data: images(self.img), label(self.label), dataframe(self.dataframe) 
        '''
        super(imgDataset, self).__init__(path, hparams, market)
        self.overlap = hparams['overlap'] # no use now
        self.window_size = hparams['window_size']
        self.p_sect = hparams['p_sect']
        self.price_ratio = hparams['price_ratio']
        
    
        self.img = Gen_Image(self.dataframe, self.p_sect, self.hparams, self.window_size, self.hparams['overlap'], self.price_ratio)
        print(len(self.img))
        label_flag = 'rtn'
        self.label = self.dataframe[label_flag].to_list()
    def tr_te_split(self, start, end):
        train_split = {"img": self.img[0: int(start * len(self.img))] + self.img[int(end * len(self.img)):]}
        train_split['label'] = self.label[0: int(start * len(self.label))] + self.label[int(end * len(self.label)):]
        train_split['dataframe'] = pd.concat([self.dataframe.iloc[0:int(start * self.dataframe.shape[0]),:],self.dataframe.iloc[int(end * self.dataframe.shape[0]):,:]],axis = 0)
        
        test_split = {"img": self.img[ int(start * len(self.img)):int(end * len(self.img))]}
        test_split['label'] = self.label[int(start * len(self.label)):int(end * len(self.label))]
        test_split['dataframe'] = self.dataframe.iloc[int(start * self.dataframe.shape[0]):int(end * self.dataframe.shape[0]),:]
        return train_split, test_split
    def report_data(self):
        print(self.dataframe.index)
        print("num of data:",len(self.img))
    def reset_data(self, data):
        self.img = data['img']
        self.label = data['label']
        self.dataframe = data['dataframe']
    def concat(self, dataset):
        self.img = self.img + dataset.img
        self.label = self.label + dataset.label
    def __len__(self):
        return len(self.label)
    def __getitem__(self,idx):
        data = self.img[idx]
        label = self.label[idx]
        return data, label
class imgOptDataset(imgDataset):
    def __init__(self, path, hparams, market, label_hv = 'd_HV1'):
        '''
        The read file should contain both information about volatility and the target asset.
        '''
        super(imgOptDataset, self).__init__(path, hparams, market)
        self.label = self.dataframe[label_hv]
        
def Gen_Vol_Image(data, p_sect, hparams, interval = 20, overlap = 19, ratio = 1, ratio_HV5 = 0.4, ratio_HV20 = 0.3, ratio_HV60 = 0.3, is_save = 0, item = "CN_opt"):
    '''
     Function: 
        Generate pictures from the given dataframe.
        The details of the method are shown in my working paper.
    Input：
        (DataFrame)data: the dataframe we used to generate pictures.
        (int)interval: time interval, daily. It is initially set 20 which represent a month's trading days.
        (str)item: the name of the item
        (float)ratio: the ratio of the size of the graph of the prices
        (dict)hparams: the dictionary of the hyperparameters.
        (int)is_save: 1: save locally. 0: add into list.
    Output:
        (list)img_list: return the list of processed pictures.
    Generate the pixel image.
    Note:
    We consider the 5-d 20-d 60-d volatilities.
    The first layer is 5-d;
    The second layer is 20-d;
    The third layer is 60-d.
    '''
    
    wide = interval * 3
    height = p_sect
    max_HV5 = data['HV5'].max()         
    min_HV5 = data['HV5'].min()
    delta_HV5 = (max_HV5 - min_HV5)
    max_HV20 = data['HV20'].max()         
    min_HV20 = data['HV20'].min()
    delta_HV20 = (max_HV20 - min_HV20)
    max_HV60 = data['HV60'].max()         
    min_HV60 = data['HV60'].min()
    delta_HV60 = (max_HV60 - min_HV60)
    count = 0
    index = 0
    img_list = []
    while index < data.shape[0]:
        
        if (count % interval) == 0:
            img = np.zeros((wide, height))
            count = 0
            index = index - overlap
        date = data.index[index]
        y_hv5 = int((ratio_HV5 * (data.loc[date, 'HV5'] - min_HV5) * height / delta_HV5  + (1 - ratio_HV5) * height))
        if y_hv5 >= height:
            y_hv5 = height-1
        img[(index % interval) * 3, y_hv5] = 1
        img[(index % interval) * 3 + 1, y_hv5] = 1
        img[(index % interval) * 3 + 2, y_hv5] = 1
        y_hv20 = int((ratio_HV20 * (data.loc[date, 'HV20'] - min_HV20) * height / delta_HV20 + (1 - ratio_HV5 - ratio_HV20) * height))
        img[(index % interval) * 3, y_hv20] = 1
        img[(index % interval) * 3 + 1, y_hv20] = 1
        img[(index % interval) * 3 + 2, y_hv20] = 1
        y_hv60 = int((ratio_HV60 * (data.loc[date, 'HV60'] - min_HV60) * height / delta_HV60 + (1 - ratio_HV5 - ratio_HV20 - ratio_HV60) * height))
        img[(index % interval) * 3, y_hv60] = 1
        img[(index % interval) * 3 + 1, y_hv60] = 1
        img[(index % interval) * 3 + 2, y_hv60] = 1

        if count % interval == interval-1:
            if is_save == 1:
                img = img.T
                im = Image.fromarray(img.astype(np.uint8))
                transform_ts = torchvision.transforms.Compose([  
                    transforms.ToTensor()])
                img = transform_ts(im)
                resize = transforms.Resize([hparams['pic_size'],hparams['pic_size']])
                im.save(".\\Image\\"+str(item)+"_"+str(int(index))+"_" +str(interval)+"_HV.png")
            else:
                img = img.T
                im = Image.fromarray(img.astype(np.uint8))
                transform_ts = torchvision.transforms.Compose([  
                    transforms.ToTensor()])
                img = transform_ts(im)
                resize = transforms.Resize([hparams['pic_size'],hparams['pic_size']])
                img = resize(img)
                img_list.append(img)
        count = count + 1
        index = index + 1
    return img_list
class imgVolOptDataset(imgOptDataset):
    def __init__(self, path, hparams, market, label_hv = 'HV20', if_ratio = -1):
        '''
        Function:
            Inheritated from Class imgOptDataset. The returned data record is the processed pictures.
            The data: images(self.img), label(self.label), dataframe(self.dataframe), images of the VIX series(self.vol) 
        Input:
            (str)label_hv: to select the label of the target predicting item.
        '''
        super(imgVolOptDataset, self).__init__(path, hparams, market)
        self.img = Gen_Image(self.dataframe, self.p_sect, self.hparams, self.window_size, self.hparams['overlap'], self.price_ratio)
        label_flag = 'rtn'
        self.label = self.dataframe[label_flag].to_list()

        vol_path = path + market + "_Vol.xlsx"
        df = pd.read_excel(vol_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        df = df.loc[: datetime.datetime.strptime(hparams['end_date'], "%Y-%m-%d")]
        temp = df.loc[:datetime.datetime.strptime(hparams['start_date'], "%Y-%m-%d"):]
        true_index = temp.index[-hparams['window_size']+1]
        df = df.loc[true_index:]

        self.vol = Gen_Vol_Image(df, self.p_sect, self.hparams, self.window_size, self.hparams['overlap'], self.price_ratio)
        
        if (not len(self.label) == len(self.img)) or (not len(self.label) == len(self.vol)):
            raise ValueError("error!", len(self.label), len(self.img), len(self.vol))
    def concat(self, dataset):
        self.img = self.img + dataset.img
        self.label = self.label + dataset.label
        self.vol = self.vol + dataset.vol
    def report_data(self):
        print(self.dataframe.index)
    def tr_te_split(self, start, end):
        train_split = {"img": self.img[0: int(start * len(self.img))] + self.img[int(end * len(self.img)):]}
        train_split['label'] = self.label[0: int(start * len(self.label))] + self.label[int(end * len(self.label)):]
        train_split['vol'] = self.vol[0: int(start * len(self.vol))] + self.label[int(end * len(self.vol)):]
        train_split['dataframe'] = pd.concat([self.dataframe.iloc[0:int(start * self.dataframe.shape[0]),:],self.dataframe.iloc[int(end * self.dataframe.shape[0]):,:]],axis = 0)
        

        test_split = {"img": self.img[ int(start * len(self.img)):int(end * len(self.img))]}
        test_split['label'] = self.label[int(start * len(self.label)):int(end * len(self.label))]
        test_split['vol'] = self.vol[int(start * len(self.vol)):int(end * len(self.vol))]
        test_split['dataframe'] = self.dataframe.iloc[int(start * self.dataframe.shape[0]):int(end * self.dataframe.shape[0]),:]
        return train_split, test_split
    def reset_data(self, data):
        self.img = data['img']
        self.label = data['label']
        self.vol = data['vol']
        self.dataframe = data['dataframe']
    def __getitem__(self,idx):
        data = self.img[idx]
        data_vol = self.vol[idx]
        label = self.label[idx]
        return data, data_vol, label
        
