# -*- coding:utf-8 -*-

import os
from tqdm import tqdm
import glob
import sys
import numpy as np
import pandas as pd

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]  # '/mnt/Pycharm_project3'
dataPath = os.path.split(rootPath)[0]  # ‘/mnt’ 跳转到云服务器中的Dataset文件夹
file_path = os.path.join(dataPath, 'Dataset/LSTM/elements')  # 跳转到云服务器中的Dataset文件夹
sys.path.append(rootPath)

from centroid import find_centroid, element_coding
from data_process import nn_seq_mtl
from data_process import csv_to_split
from softmax_all import softmax_all, df_indextransform, heatmap
from args import multi_task_args_parser
from model_train import mtl_train
from model_test import mtl_test
# from model_test import pearsonr

# path = os.path.abspath(os.path.dirname(os.getcwd()))
# LSTM_PATH = path + '/models/mtl.pkl'
LSTM_PATH = rootPath + '/models/mtl.pkl'

files = glob.glob(os.path.join(file_path, "*.csv"))  # 构建文件列表
print(files)
# project3的文件下 区别于云服务器大Dataset
test = os.path.join(rootPath, 'data/test.csv')
print('test的路径是：', test)
data = os.path.join(rootPath, 'data/data.csv')
print('data的路径是：', data)
elements_all = os.path.join(rootPath, 'data/elements_all.csv')
Atomic_relative_mass = os.path.join(rootPath, 'data/Atomic_relative_mass.csv')

df_atomic = pd.read_csv(Atomic_relative_mass, keep_default_na=False, encoding='utf-8',
names=['序号','元素', 'elements', '相对原子质量'])
# df_atomic = pd.read_csv(r'Atomic_relative_mass.csv', keep_default_na=False, encoding='utf-8',
# names=['序号','元素', 'elements', '相对原子质量'])
# print(df_atomic)
list2 = list(df_atomic['相对原子质量'].to_list())
# list2 = list(set(list2))  # 去掉重复项

list3 = np.append(list2, 'weight')
list4 = list(df_atomic['elements'].to_list())

# list4 = list(set(list4))  # 得到图的横纵标题
if os.path.isfile(rootPath + '/all_matrix.csv'): # check if it exists
    all_matrix = pd.read_csv(rootPath + '/all_matrix.csv')
    all_matrix.drop([all_matrix.columns[0]], axis=1, inplace=True)  # 不然会重复一列index
    # print('all_matrix:', all_matrix)
    all_matrix.index = list2
else:
    all_matrix = pd.DataFrame(np.zeros(shape=(len(list2), len(list2) + 1), dtype=float),
                          index=list2, columns=list3)  # 最右边留一列权重数

# if __name__ == '__main__':
#     args = multi_task_args_parser()
#     Dtr, Val, Dte, scaler = nn_seq_mtl(seq_len=args.seq_len, B=args.batch_size, pred_step_size=args.pred_step_size)
#     mtl_train(args, Dtr, Val, LSTM_PATH)
#     mtl_test(args, Dte, scaler, LSTM_PATH)
#     # pearsonr(y,pred) 引用不了y和pred

if __name__ == '__main__':
    args = multi_task_args_parser()

    list_bad = []  # 用来存准确率较低的晶体名称
    for file in tqdm(files):
        print('loading file:', file)
        df, num = csv_to_split(file)
        # print('df:', df)
        # df.insert(0, 'id', np.arange(len(df)))  # 第一列前插入一列id，从1~n
        df_1, test = find_centroid(df, test)
        file_data = element_coding(data, test, elements_all)
        Dtr, Val, Dte, scaler = nn_seq_mtl(file=data, seq_len=args.seq_len, B=args.batch_size, pred_step_size=args.pred_step_size)
        mtl_train(args, Dtr, Val, LSTM_PATH)
        mtl_test(args, Dte, scaler, LSTM_PATH)
        file_input = rootPath + '/algorithms/elements.csv'
        all_matrix, acc = softmax_all(file_input, all_matrix, num)

        if acc < 0.65:
            list_bad.append([file, acc])
            print('准确率低于0.65的晶体文件名和对应的准确率，list_bad:', list_bad)

        all_matrix.to_csv('all_matrix.csv', index=True)  # 每次都保存

    # 实际每次保存依然会降低精度，且旧矩阵（分数）与新矩阵（仅分子）数量级不同，所以选择不除以权重
    # 如果选择使用下面这段代码，则需要在上面判断if os.path.isfile(rootPath + '/all_matrix.csv')中增加每行乘weight
    # for i in range(len(list2)):
    #     if all_matrix.iloc[i, -1] != 0:
    #         all_matrix.iloc[i, :-1] = all_matrix.iloc[i, :-1] / all_matrix.iloc[i, -1]
    #     else:
    #         all_matrix.iloc[i, :-1] = 0

    all_matrix.to_csv('all_matrix.csv', index=True)  # 尝试不是每次都保存

    diag_vals = all_matrix.iloc[:, :-1].values.diagonal()  # 获取对角线上的元素集合成array
    weights = all_matrix['weight']  # 获取最后一列
    # acc = (diag_vals * weights).sum() / weights.sum()  # 计算的是带分母的矩阵准确率
    acc_all = diag_vals.sum() / weights.sum()  # 计算的是不带分母的矩阵准确率
    print('all_matrix所有元素的准确率为：', acc_all)
    print('准确率低于0.5的晶体文件名和对应的准确率，list_bad:', list_bad)
    all_matrix_output = df_indextransform(all_matrix, list4)
    heatmap(all_matrix_output, list4)
    print('-----------------------------------------Done!-----------------------------------------')