# -*- coding:utf-8 -*-
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(file_name):
    """
    :return: dataframe
    """
    path = os.path.dirname(os.path.realpath(__file__)) + '/data/' + file_name
    df = pd.read_csv(path, encoding='gbk')
    df.fillna(df.mean(), inplace=True)

    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


# Multiple outputs data processing.
def nn_seq_mo(seq_len, B, num):
    data = load_data('data.csv')

    train = data[:int(len(data) * 0.6)]
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):len(data)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(dataset, batch_size, step_size):
        load = dataset[dataset.columns[1]]
        load = (load - n) / (m - n)
        load = load.tolist()
        dataset = dataset.values.tolist()
        seq = []
        for i in range(0, len(dataset) - seq_len - num, step_size):
            train_seq = []
            train_label = []

            for j in range(i, i + seq_len):
                x = [load[j]]
                for c in range(2, 8):
                    x.append(dataset[j][c])
                train_seq.append(x)

            for j in range(i + seq_len, i + seq_len + num):
                train_label.append(load[j])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=6, drop_last=True)
        # 2080Ti的num_worker数量为6

        return seq

    Dtr = process(train, B, step_size=1)
    Val = process(val, B, step_size=1)
    Dte = process(test, B,step_size=num)

    return Dtr, Val, Dte, m, n


# Single step scrolling data processing.
def nn_seq_sss(seq_len, B):
    data = load_data('data.csv')

    train = data[:int(len(data) * 0.6)]
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):len(data)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(dataset, batch_size):
        load = dataset[dataset.columns[1]]
        load = (load - n) / (m - n)
        load = load.tolist()
        dataset = dataset.values.tolist()
        seq = []
        for i in range(len(dataset) - seq_len):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                x = [load[j]]
                for c in range(2, 8):
                    x.append(dataset[j][c])
                train_seq.append(x)
            train_label.append(load[i + seq_len])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=6, drop_last=True)
        # 2080Ti的num_worker数量为6

        return seq

    Dtr = process(train, B)
    Val = process(val, B)
    Dte = process(test, B)

    return Dtr, Val, Dte, m, n


# Multiple models single step data processing.
def nn_seq_mmss(seq_len, B, pred_step_size):
    data = load_data('data.csv')

    train = data[:int(len(data) * 0.6)]
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):len(data)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(dataset, batch_size, step_size):
        load = dataset[dataset.columns[1]]
        load = (load - n) / (m - n)
        dataset = dataset.values.tolist()
        load = load.tolist()
        #
        seqs = [[] for i in range(pred_step_size)]
        for i in range(0, len(dataset) - seq_len - pred_step_size, step_size):
            train_seq = []
            for j in range(i, i + seq_len):
                x = [load[j]]
                for c in range(2, 8):
                    x.append(dataset[j][c])
                train_seq.append(x)
            for j, ind in zip(range(i + seq_len, i + seq_len + pred_step_size), range(pred_step_size)):
                #
                train_label = [load[j]]
                seq = torch.FloatTensor(train_seq)
                train_label = torch.FloatTensor(train_label).view(-1)
                seqs[ind].append((seq, train_label))
        #
        res = []
        for seq in seqs:
            seq = MyDataset(seq)
            seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=6, drop_last=True)
            res.append(seq)
            # 2080Ti的num_worker数量为6
        return res

    Dtrs = process(train, B, step_size=1)
    Vals = process(val, B, step_size=1)
    Dtes = process(test, B, step_size=pred_step_size)

    return Dtrs, Vals, Dtes, m, n


# Multi task learning
def nn_seq_mtl(file, seq_len, B, pred_step_size):  # 改成批量输入
    # df = load_data(file)  # mtl_data_2.csv 但仍然报错"_io.TextIOWrapper"
    df = load_data('data.csv')
    # df = load_data('mtl_data_copy.csv')  # mtl_data_2.csv
    # df = pd.read_csv(file, encoding='gbk').astype('float32', errors='ignore')
    df.fillna(df.mean(), inplace=True)  # 缺失值填充
    data = pd.concat([df] * 64)
    # print('data:', data)

    # split
    train = data[:int(len(data) * 0.6)].copy()  # int()为整形，data[:0.6]指取前60%
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)].copy()
    test = data[int(len(data) * 0.8):len(data)].copy()
    # test = data[int(len(data) * 0.8):int(len(data) * 0.8)+243+seq_len]  想令test取前243个
    # 但会报错ValueError: Shapes of x (243,) and y (150,) are incompatible

    # normalization
    train.drop([train.columns[0]], axis=1, inplace=True)
    val.drop([val.columns[0]], axis=1, inplace=True)
    test.drop([test.columns[0]], axis=1, inplace=True)
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train.values)
    val = scaler.transform(val.values)
    test = scaler.transform(test.values)

    def process(dataset, batch_size, step_size):
        dataset = dataset.tolist()
        seq = []
        for i in range(0, len(dataset) - seq_len - pred_step_size, step_size):
            train_seq = []
            for j in range(i, i + seq_len):
                x = []
                for c in range(len(dataset[0])):  # 前24个时刻的所有变量
                    x.append(dataset[j][c])
                train_seq.append(x)
            # 下几个时刻的所有变量
            train_labels = []
            for j in range(len(dataset[0])):
                train_label = []
                for k in range(i + seq_len, i + seq_len + pred_step_size):
                    train_label.append(dataset[k][j])
                train_labels.append(train_label)
            # tensor
            train_seq = torch.FloatTensor(train_seq)
            train_labels = torch.FloatTensor(train_labels)
            seq.append((train_seq, train_labels))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=6, drop_last=True)
        # 2080Ti的num_worker数量为6

        return seq

    Dtr = process(train, B, step_size=1)
    Val = process(val, B, step_size=1)
    Dte = process(test, B, step_size=pred_step_size)

    return Dtr, Val, Dte, scaler


def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    # MAPE 百分比均方差
    # loss = np.mean(np.abs((x - y) / x))
    # loss = np.mean(np.abs((x - y) / y))  分母改用了y，避开x中存在的0值

    # sMAPE 平均绝对百分比误差
    loss = np.mean(np.abs((x - y) / (0.5 * (x + y))))
    
    # MAE 均方差
    # i = len(x)
    # loss = sum(np.abs(x - y))/i

    return loss


def csv_to_split(file):
    data = pd.read_csv(file, sep=r'\\s+', engine='python')  # skiprows=[0,1]
    df = pd.DataFrame(data[data.columns[0]].str.split(' ', n=12))  # 设置参数expand=True可使其分列而不是1列
    for i in df.columns:
        if df[i].count() == 0:
            df.drop(labels=i, axis=1, inplace=True)
    df.rename(columns={list(df)[0]: 'col1_new_name'}, inplace=True)  # 重命名列名
    df_split13 = df['col1_new_name'].apply(lambda x: pd.Series(x))  # 拆开列表1列变13列
    df_split13 = df_split13.replace(r'^\s*$', np.nan, regex=True)  # 用NaN代替空格
    split_data = df_split13.T.apply(lambda x: pd.Series(x.dropna().values)).T  # 去除空值向左移
    split_data.columns = ['Elements', 'x', 'y', 'z']
    num = len(split_data['Elements'])
    return split_data, num
