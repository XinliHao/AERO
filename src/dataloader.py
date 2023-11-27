# -*-coding:utf-8 -*-

# @Time ：2023/6/11 16:52
# @Author:xinli hao
# @Email:xinli_hao@ruc.edu.cn
import torch
import numpy as np
import os


class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def normalize(a, min_a=None, max_a=None):
    if min_a is None:
        min_a = np.min(a, axis=0)
    time = a - min_a
    return time

# def load_pos_graph(dataset,data_folder):
#     # load predefined graph
#     adj_pre = torch.load(f'{data_folder}/{dataset}/{dataset}_train_adj.pt').astype(float)
#     return adj_pre

def split_dataset(dataset,data_folder,model):
    folder = os.path.join(data_folder, dataset)
    if not os.path.exists(folder):
        print(folder)
        raise Exception('Processed Data not found.')
    
    loader = []
    for file in ['train', 'test', 'labels']:
        finalpath = os.path.join(folder, f'{dataset}_{file}.npy')
        print("load data from ", finalpath)
        loader.append(np.load(finalpath))
    
    loader[0][:, 0] = normalize(loader[0][:, 0])
    loader[1][:, 0] = normalize(loader[1][:, 0])
    return loader[0].astype(np.float32),loader[0].astype(np.float32),loader[1].astype(np.float32),loader[2]


def convert_to_windows(data,w_size):
    # data[2000,25]
    windows = []
    # 步长为1
    step = 1
    # for i, g in enumerate(data):
    for i in range(w_size, data.shape[0], step):
        if i >= w_size:
            w = data[i - w_size:i]
        else:
            continue
            # repeat的参数是对应维度的复制个数,表示data[0]的第一个维度重复w_size-i次，第二个维度不变
            # # 所以实现的结果是，从第一个点开始，就从前面提取n_window长度的窗口，如果长度不够，就重复第一个点直到补齐
            # w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        windows.append(w)
        # windows.append(w if 'TranAD' in args.model or 'Attention' in args.model or 'GDN' in args.model or 'AERO' or 'OnlyTemporal' or 'OnlyConcurrent'in args.model else w.view(-1))
    return np.stack(windows)
