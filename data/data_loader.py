import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta


import ast

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    '''from scratch!!'''
    def __read_data__(self):
        print("in here?")
        self.scaler = StandardScaler()

        # Load tensor files from directory
        path = 'tensorData/'
        pt_files = sorted([f for f in os.listdir(path) if f.endswith('.pt')])
        tensor_list = []

        for file in pt_files:
            tensor_path = os.path.join(path, file)
            tensor = torch.load(tensor_path)  # Expecting shape (340, 220)

            if tensor.shape != (340, 220):
                raise ValueError(f"Unexpected shape {tensor.shape} in {file}")

            tensor_list.append(tensor)

        # Stack tensors into shape (num_samples, 340, 220)
        combined_tensor = torch.stack(tensor_list, dim=0)  # torch.Tensor
        num_samples = combined_tensor.shape[0]

        # Split boundaries
        num_train = int(num_samples * 0.8)
        num_test = int(num_samples * 0.1)
        num_vali = num_samples - num_train - num_test

        border1s = [0, num_train - self.seq_len, num_samples - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, num_samples]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        print('comined anything?', combined_tensor[border1:border2])
        self.data_x = combined_tensor[border1:border2]
        print(f"shape? 3? {self.data_x.shape}")
        if self.inverse:
            self.data_y = combined_tensor[border1:border2]
        else:
            self.data_y = combined_tensor[border1:border2]
        # self.data_stamp = np.arange(border1, border2)
        start_date = datetime(2024, 1, 1)  # Start date is the first day of 2024

        # Generate the corresponding dates for the days in the range
        dates = [start_date + timedelta(days=int(day_num)-1) for day_num in np.arange(border1, border2)]

        # Convert the dates into a pandas dataframe (this mimics df_stamp['date'])
        df_stamp = pd.DataFrame({'date': dates})

        # Now call your time_features function (assuming it's similar to time_features.py in the original repo)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        # data_stamp should now contain the same output as before, but using day numbers instead of the actual dates.
        self.data_stamp = data_stamp


##########################    
    """def something(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        tensor_list = []
        path = 'tensorData/'
        pt_files = sorted([f for f in os.listdir(path) if f.endswith('.pt')])
        for file in pt_files:
            tensor_path = os.path.join(path, file)
            tensor = torch.load(tensor_path)  # Assumes each file has shape (340, 220)
            
            if tensor.shape != (340, 220):
                raise ValueError(f"Unexpected shape {tensor.shape} in {file}")
            
            tensor_list.append(tensor)

        # Stack all tensors into one big tensor of shape (num_data_points, 340, 220)
        combined_tensor = torch.stack(tensor_list, dim=0)
        print("HELLO!", combined_tensor.shape)
        '''
        good for 2D rows
        # print(f"before: {df_raw[:2]}")
        # df_raw['padded_coordinates'] = df_raw['padded_coordinates'].apply(
        #     lambda x: np.array([list(map(float, row.split())) for row in x.replace('[', '').replace(']', '').split('\n')], dtype=np.float32)
        # )
        '''
        #  using for the change of the datapoints from string to tensors. :D
        df_raw['tensor'] = df_raw['tensor'].apply(lambda x: torch.tensor(ast.literal_eval(x), dtype=torch.float32))
        df_raw['tensor'] = df_raw['tensor'].apply(lambda x: x.numpy())


        # df_raw['padded_coordinates'] = df_raw['padded_coordinates'].apply(
        #     lambda x: torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x
        # )
        # df_raw['padded_coordinates'] = df_raw['padded_coordinates'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
        '''df_raw['padded_coordinates'] = df_raw['padded_coordinates'].apply(
            lambda x: np.array([list(map(float, row.split())) for row in x.replace('[', '').replace(']', '').split('\n')], dtype=np.float32)
        )
        print(f"type after the change: {type(df_raw['padded_coordinates'].iloc[0])}")
        for i, value in enumerate(df_raw['padded_coordinates']):
            if not isinstance(value, np.ndarray):
                print(f"Non-numpy array at index {i}: {type(value)}")
        x = np.array(x, dtype=np.float32)
        if x.ndim != expected_ndim or x.shape != (expected_shape,):
            raise ValueError(f"Unexpected shape: {x.shape}")
        # df_raw.columns: ['date', ...(other features), target feature]
        df_raw.columns = ['date', 'padded_coordinates']'''
        # df_raw.columns = ['date','min_lat', 'max_lat', 'min_lon', 'max_lon', 'avg_brightness', 'avg_confidence']
        # df_raw.columns = ['date','confidence', 'brightness', 'coordinates']
        # df_raw.columns = ['date','padded_coordinates']
        df_raw.columns = ['date','tensor']
        # df_raw.columns = ['date','confidence', 'brightness', 'padded_coordinates']
        # df_raw.columns = ['date', 'lat1', 'lon1', 'lat2', 'lon2', 'lat3', 'lon3', 'lat4', 'lon4', 'lat5', 'lon5', 'lat6', 'lon6', 'lat7', 'lon7', 'lat8', 'lon8', 'lat9', 'lon9', 'lat10', 'lon10', 'lat11', 'lon11', 'lat12', 'lon12', 'lat13', 'lon13', 'lat14', 'lon14', 'lat15', 'lon15', 'lat16', 'lon16', 'lat17', 'lon17', 'lat18', 'lon18', 'lat19', 'lon19', 'lat20', 'lon20', 'lat21', 'lon21', 'lat22', 'lon22', 'lat23', 'lon23', 'lat24', 'lon24', 'lat25', 'lon25', 'lat26', 'lon26', 'lat27', 'lon27', 'lat28', 'lon28', 'lat29', 'lon29', 'lat30', 'lon30', 'lat31', 'lon31', 'lat32', 'lon32', 'lat33', 'lon33', 'lat34', 'lon34', 'lat35', 'lon35', 'lat36', 'lon36', 'lat37', 'lon37', 'lat38', 'lon38', 'lat39', 'lon39', 'lat40', 'lon40', 'lat41', 'lon41', 'lat42', 'lon42', 'lat43', 'lon43', 'lat44', 'lon44', 'lat45', 'lon45', 'lat46', 'lon46', 'lat47', 'lon47', 'lat48', 'lon48', 'lat49', 'lon49', 'lat50', 'lon50', 'lat51', 'lon51', 'lat52', 'lon52', 'lat53', 'lon53', 'lat54', 'lon54', 'lat55', 'lon55', 'lat56', 'lon56', 'lat57', 'lon57', 'lat58', 'lon58', 'lat59', 'lon59', 'lat60', 'lon60', 'lat61', 'lon61', 'lat62', 'lon62', 'lat63', 'lon63', 'lat64', 'lon64', 'lat65', 'lon65', 'lat66', 'lon66', 'lat67', 'lon67', 'lat68', 'lon68', 'lat69', 'lon69', 'lat70', 'lon70', 'lat71', 'lon71', 'lat72', 'lon72', 'lat73', 'lon73', 'lat74', 'lon74', 'lat75', 'lon75', 'lat76', 'lon76', 'lat77', 'lon77', 'lat78', 'lon78', 'lat79', 'lon79', 'lat80', 'lon80', 'lat81', 'lon81', 'lat82', 'lon82', 'lat83', 'lon83', 'lat84', 'lon84', 'lat85', 'lon85', 'lat86', 'lon86', 'lat87', 'lon87', 'lat88', 'lon88', 'lat89', 'lon89', 'lat90', 'lon90', 'lat91', 'lon91', 'lat92', 'lon92', 'lat93', 'lon93', 'lat94', 'lon94', 'lat95', 'lon95', 'lat96', 'lon96', 'lat97', 'lon97', 'lat98', 'lon98', 'lat99', 'lon99', 'lat100', 'lon100', 'brightness', 'confidence']
        # df_raw.columns = ['date', 'lat1', 'lon1', 'lat2', 'lon2', 'lat3', 'lon3', 'lat4', 'lon4', 'lat5', 'lon5', 'lat6', 'lon6', 'lat7', 'lon7', 'lat8', 'lon8', 'lat9', 'lon9', 'lat10', 'lon10', 'lat11', 'lon11', 'lat12', 'lon12', 'lat13', 'lon13', 'lat14', 'lon14', 'lat15', 'lon15', 'lat16', 'lon16', 'lat17', 'lon17', 'lat18', 'lon18', 'lat19', 'lon19', 'lat20', 'lon20', 'brightness', 'confidence']
        # cols = list(df_raw.columns); 
        print(f"after: {df_raw[:2]}")
        # print(f"something?? : {df_raw[:2]}")
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove('date')
            print("in here?")
        df_raw = df_raw[['date']+cols]

        df_features = df_raw[['date'] + cols]
        # print(f"somewhat after: {df_raw[:2]}")
        num_train = int(len(df_raw)*0.8)
        num_test = int(len(df_raw)*0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[self.target]

        # print(f"before scale: {df_data[:2]}")
        # if self.scale:
        #     train_data = df_data[border1s[0]:border2s[0]]
        #     self.scaler.fit(train_data.values)
        #     data = self.scaler.transform(df_data.values)
        #     print("in scale")
        # else:
        data = df_data.values
        print(f"is this a tensor? {type(data)}")
        print(f"is this a tensor? {data.shape}")
        
        print(f"it is? {data[0][0][0]}")
        print(f"DUDE PLEASE: {type(data[0][0][0])}")

        # print(f"after scale: {df_data[:2]}")
        # print(f"data now: {data[:2]}")
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        print("stamps: ", data_stamp)

        # print(f"before scale: {df_data[:2]}")
        self.data_x = data[border1:border2]
        print(f"shape? 3? {self.data_x.shape}")
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp"""

        # print(f"data head: {self.data_x[:5]}")
        # print(f"DO I GET SOMETHING: {self.data_x[1]}")


    def __getitem__(self, index):
        # print('are you in here?')
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        # print(f'brother wat? {seq_x.shape}')
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # print("seq_x:", seq_x.shape, seq_x.dtype)
        # print("seq_x: ", seq_x)
        # print("seq_y:", seq_y.shape, seq_y.dtype)

        '''ONLY USED FOR MY USECASE????'''
        # seq_x = torch.stack([
        #     torch.tensor(item.item(), dtype=torch.float32)
        #     if isinstance(item, np.ndarray) and item.dtype == object
        #     else torch.tensor(item, dtype=torch.float32)
        #     for item in seq_x
        # ], dim=0)

        # print(f"{type(seq_x[0])}")
        # print(f"{type(seq_y[0])}")
        # print(f"{type(seq_x_mark[0])}")
        # print(f"{type(seq_y_mark[0])}")

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    # def __getitem__(self, index):
        
    #     s_begin = index
    #     s_end = s_begin + self.seq_len
    #     r_begin = s_end - self.label_len 
    #     r_end = r_begin + self.label_len + self.pred_len

    #     seq_x = self.data_x[s_begin:s_end]
    #     if self.inverse:
    #         seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], axis=0)
    #     else:
    #         seq_y = self.data_y[r_begin:r_end]

    #     seq_x_mark = self.data_stamp[s_begin:s_end]
    #     seq_y_mark = self.data_stamp[r_begin:r_end]

    #     for i, item in enumerate(seq_x):
    #         print(f"seq_x[{i}] shape: {item.shape}, type: {type(item)}")

    #     # Use stack instead of array to force shape alignment
    #     seq_x = np.stack(seq_x).astype(np.float32)
    #     seq_y = np.stack(seq_y).astype(np.float32)
    #     seq_x_mark = np.stack(seq_x_mark).astype(np.float32)
    #     seq_y_mark = np.stack(seq_y_mark).astype(np.float32)

    #     return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        print("hrlp", len(self.data_x))
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        # df_raw.columns = ['date', 'lat1', 'lon1', 'lat2', 'lon2', 'lat3', 'lon3', 'lat4', 'lon4', 'lat5', 'lon5', 'lat6', 'lon6', 'lat7', 'lon7', 'lat8', 'lon8', 'lat9', 'lon9', 'lat10', 'lon10', 'lat11', 'lon11', 'lat12', 'lon12', 'lat13', 'lon13', 'lat14', 'lon14', 'lat15', 'lon15', 'lat16', 'lon16', 'lat17', 'lon17', 'lat18', 'lon18', 'lat19', 'lon19', 'lat20', 'lon20', 'brightness', 'confidence']

        # df_raw['padded_coordinates'] = df_raw['padded_coordinates'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # df_raw.columns = ['date','min_lat', 'max_lat', 'min_lon', 'max_lon', 'avg_brightness', 'avg_confidence']
        # df_raw.columns = ['date', 'lat1', 'lon1', 'lat2', 'lon2', 'lat3', 'lon3', 'lat4', 'lon4', 'lat5', 'lon5', 'lat6', 'lon6', 'lat7', 'lon7', 'lat8', 'lon8', 'lat9', 'lon9', 'lat10', 'lon10', 'lat11', 'lon11', 'lat12', 'lon12', 'lat13', 'lon13', 'lat14', 'lon14', 'lat15', 'lon15', 'lat16', 'lon16', 'lat17', 'lon17', 'lat18', 'lon18', 'lat19', 'lon19', 'lat20', 'lon20', 'lat21', 'lon21', 'lat22', 'lon22', 'lat23', 'lon23', 'lat24', 'lon24', 'lat25', 'lon25', 'lat26', 'lon26', 'lat27', 'lon27', 'lat28', 'lon28', 'lat29', 'lon29', 'lat30', 'lon30', 'lat31', 'lon31', 'lat32', 'lon32', 'lat33', 'lon33', 'lat34', 'lon34', 'lat35', 'lon35', 'lat36', 'lon36', 'lat37', 'lon37', 'lat38', 'lon38', 'lat39', 'lon39', 'lat40', 'lon40', 'lat41', 'lon41', 'lat42', 'lon42', 'lat43', 'lon43', 'lat44', 'lon44', 'lat45', 'lon45', 'lat46', 'lon46', 'lat47', 'lon47', 'lat48', 'lon48', 'lat49', 'lon49', 'lat50', 'lon50', 'lat51', 'lon51', 'lat52', 'lon52', 'lat53', 'lon53', 'lat54', 'lon54', 'lat55', 'lon55', 'lat56', 'lon56', 'lat57', 'lon57', 'lat58', 'lon58', 'lat59', 'lon59', 'lat60', 'lon60', 'lat61', 'lon61', 'lat62', 'lon62', 'lat63', 'lon63', 'lat64', 'lon64', 'lat65', 'lon65', 'lat66', 'lon66', 'lat67', 'lon67', 'lat68', 'lon68', 'lat69', 'lon69', 'lat70', 'lon70', 'lat71', 'lon71', 'lat72', 'lon72', 'lat73', 'lon73', 'lat74', 'lon74', 'lat75', 'lon75', 'lat76', 'lon76', 'lat77', 'lon77', 'lat78', 'lon78', 'lat79', 'lon79', 'lat80', 'lon80', 'lat81', 'lon81', 'lat82', 'lon82', 'lat83', 'lon83', 'lat84', 'lon84', 'lat85', 'lon85', 'lat86', 'lon86', 'lat87', 'lon87', 'lat88', 'lon88', 'lat89', 'lon89', 'lat90', 'lon90', 'lat91', 'lon91', 'lat92', 'lon92', 'lat93', 'lon93', 'lat94', 'lon94', 'lat95', 'lon95', 'lat96', 'lon96', 'lat97', 'lon97', 'lat98', 'lon98', 'lat99', 'lon99', 'lat100', 'lon100', 'brightness', 'confidence']
        # df_raw.columns = ['date', 'padded_coordinates']
        df_raw.columns = ['date','tensor']

        # df_raw.columns = ['date','confidence', 'brightness', 'padded_coordinates']
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove('date')
        
        df_raw = df_raw[['date']+cols]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        print(f"{self.data_x[:1]}")

    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
