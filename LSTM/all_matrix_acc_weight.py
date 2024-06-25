# -*- coding:utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np

"""
控制开关，case1是远程服务器，case2是本地，num是文件名序列号
"""
# case1 = True
case1 = False
case2 = True
# case2 = False

num = 24


# 将DataFrame每行都除以对应的weight值，注意跳过weight=0的行
def convert(filename):
    df = pd.read_csv(filename)
    # 对每行的weight进行判断，若不为0，则将该行的每个元素除以其对应的weight值
    df.iloc[:, :-1] = df.apply(lambda row: row.iloc[:-1] / row['weight'] if row['weight'] != 0 else row.iloc[:-1],
                                   axis=1)
    print('all_matrix经过转换后为', df)
    return df


"""
如果文件在远程服务器上，则运行这段
"""
if case1:
    filePath = os.path.abspath(os.path.dirname(__file__))  # '/mnt/Pycharm_project3'
    file = os.path.join(filePath, 'all_matrix{}.csv'.format(num))  # 跳转到同目录中的all_matrix文件
    # 保存到同样的位置并命名为all_matrix_convert.csv
    output_file = os.path.join(filePath, 'all_matrix_convert{}.csv'.format(num))

    if os.path.exists(file):
        print("文件存在")
        df = convert(file)
        df.to_csv(output_file, index=False)
    else:
        print("文件不存在")


"""
如果文件在本地，则运行这段
"""
if case2:
    file = r'D:\学习\研二\小论文point cloud transformer\代码运行日志\lstm\all_matrix{}.csv'.format(num)
    # 保存到同样的位置并命名为all_matrix_convert.csv
    output_file = r'D:\学习\研二\小论文point cloud transformer\代码运行日志\lstm\all_matrix{}_convert.csv'.format(num)

    if os.path.exists(file):
        print("文件存在")
        df = convert(file)
        df.to_csv(output_file, index=False)
    else:
        print("文件不存在")

