# -*- coding:utf-8 -*-
"""
测试heatmap_small函数，将总的矩阵分解成多个小的矩阵
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
curPath = os.path.abspath(os.path.dirname(__file__))
file = curPath + '/algorithms/elements.csv'
file_out = curPath + '/algorithms/Atomic_relative_mass.csv'
df_atomic = pd.read_csv(file_out, keep_default_na=False, encoding='utf-8',
                        names=['序号', '元素', 'elements', '相对原子质量'])
# df_atomic = pd.read_csv(r'Atomic_relative_mass.csv', keep_default_na=False, encoding='utf-8',
# names=['序号','元素', 'elements', '相对原子质量'])
print(df_atomic)
list2 = df_atomic['相对原子质量'].to_list()
list2 = list(set(list2))  # 去掉重复项
print('list2:', list2)
list3 = np.append(list2, 'weight')
list4 = df_atomic['elements'].to_list()
# list4 = list(set(list4))  # 得到图的横纵标题
list4 = list(OrderedDict.fromkeys(list4))  # 将列表中的元素作为OrderedDict的键来去重，并保持顺序不变
print('list4:', list4)
all_matrix = pd.read_csv(r'D:\学习\研二\小论文point cloud transformer\代码运行日志\lstm\all_matrix5.1.csv', index_col=0)


# 把行列标签换成元素
def df_indextransform(all_matrix, list_label):
    # all_matrix = all_matrix.rename(index={list_label}, columns={list_label})
    # all_matrix_output = all_matrix.drop(['weight'], axis=1)
    print('all_matrix:', all_matrix)
    all_matrix_output = all_matrix
    all_matrix_output.columns = list_label + ['weight']
    all_matrix_output.index = list_label
    # all_matrix_output = all_matrix_output.values  # 这行代码会把行列标签替换成0-102

    return all_matrix_output


def heatmap_small(all_matrix):
    # 读取csv文件
    df = pd.DataFrame(all_matrix)
    print('all_matrix:', df)

    # 创建五个空的dataframe
    elements1 = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']
    df1 = pd.DataFrame(np.zeros((len(elements1), len(elements1)+1)), columns=elements1 + ['weight'], index=elements1)
    elements2 = ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']
    df2 = pd.DataFrame(np.zeros((len(elements2), len(elements2)+1)), columns=elements2 + ['weight'], index=pd.Index(elements2))
    elements3 = ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                'Kr']
    df3 = pd.DataFrame(np.zeros((len(elements3), len(elements3)+1)), columns=elements3 + ['weight'], index=pd.Index(elements3))
    elements4 = ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                'Xe']
    df4 = pd.DataFrame(np.zeros((len(elements4), len(elements4)+1)), columns=elements4 + ['weight'], index=pd.Index(elements4))
    elements5 = ['Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']
    df5 = pd.DataFrame(np.zeros((len(elements5), len(elements5)+1)), columns=elements5 + ['weight'], index=pd.Index(elements5))


    print('df.index', df.index)
    for i in df.index:
        for j in df.columns:
            if i in elements1:
                if j in elements1:
                    df1.loc[i, j] = df.loc[i, j]
            elif i in elements2:
                if j in elements2:
                    df2.loc[i, j] = df.loc[i, j]
            elif i in elements3:
                if j in elements3:
                    df3.loc[i, j] = df.loc[i, j]
            elif i in elements4:
                if j in elements4:
                    df4.loc[i, j] = df.loc[i, j]
            elif i in elements5:
                if j in elements5:
                    df5.loc[i, j] = df.loc[i, j]

        #  将该行数据复制到对应的小dataframe中
        # for col_label in df.columns[:-1]:
        #     period_df.at[row_label, col_label] = df.at[row_label, col_label]

        # period_df.at[row_label, 'weight'] = df.at[row_label, 'weight']

    # 将DataFrame保存为csv文件
    file_Path = r'D:\学习\研二\小论文point cloud transformer\代码运行日志\lstm\all_matrix小图'
    df1.to_csv(file_Path+'/matrix1.csv', index=True)
    df2.to_csv(file_Path+'/matrix2.csv', index=True)
    df3.to_csv(file_Path+'/matrix3.csv', index=True)
    df4.to_csv(file_Path+'/matrix4.csv', index=True)
    df5.to_csv(file_Path+'/matrix5.csv', index=True)

    print('df1:', df1)
    accuracy(df1)
    print('df2:', df2)
    accuracy(df2)
    print('df3:', df3)
    accuracy(df3)
    print('df4:', df4)
    accuracy(df4)
    print('df5:', df5)
    accuracy(df5)

    heatmap(df1, elements1)
    heatmap(df2, elements2)
    heatmap(df3, elements3)
    heatmap(df4, elements4)
    heatmap(df5, elements5)

    return df1, df2, df3, df4, df5


def accuracy(all_matrix):
    result = 0
    diag = []
    for i in range(len(all_matrix)):  # len(all_matrix) - 1
        diag_vals = all_matrix.iloc[i, i]  # [i, i + 1]
        # print('diag_vals', diag_vals)
        weight_val = all_matrix.iloc[i, -1]  # [i + 1, -1]
        result += diag_vals * weight_val
        diag.append(diag_vals)
    weights = all_matrix['weight']
    acc = result / weights.sum()
    print('该矩阵的准确率为：', acc)
    print('diag:', diag)
    return acc, diag


def heatmap(all_matrix, list_element):
    """如果上面softmax_all输出保留了最后一列weight，则需要在这删除该列，虽然是acc_matrix，但实际是all_matrix。"""
    all_matrix = all_matrix.drop(['weight'], axis=1)
    pic_acc_matrix = all_matrix
    heat_map = np.random.rand(len(list_element), len(list_element))  # 设置长宽
    fig, ax = plt.subplots(figsize=(9, 9))
    data = np.array(all_matrix)
    # sns.heatmap(data, annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="YlOrRd")
    # columns是横坐标，index是纵坐标
    # sns.heatmap(pd.DataFrame(np.round(pic_acc_matrix, 2), columns=list_element, index=list_element),
    #             annot=False, vmax=1, vmin=0, xticklabels=list4, yticklabels=list4, linewidths=0,
    #             square=True, cmap="YlGnBu")
    sns.heatmap(pd.DataFrame(np.round(pic_acc_matrix, 2)),
                annot=False, vmax=1, vmin=0, xticklabels=list_element,
                yticklabels=list_element, linewidths=0,
                square=True, cmap="YlGnBu")
    # ax.set_title('元素拟合热力图', fontsize=18)
    ax.set_ylabel('Pred', fontsize=18)
    ax.set_xlabel('True', fontsize=18)  # 横变成y轴，跟矩阵原始的布局情况是一样
    # ax.set_yticklabels([list_lebal], fontsize=18, rotation=360, horizontalalignment='right')
    # ax.set_xticklabels([list_lebal], fontsize=18, horizontalalignment='right')
    fig.savefig(curPath + '/heatmap.jpg', dpi=400, bbox_inches='tight')
    plt.show()
    return


# all_matrix_output = df_indextransform(all_matrix, list4)
# print("原始表格的索引：", all_matrix.index)
# heatmap_small(all_matrix_output)

'''
下面的代码用来在all_matrix文件的最右边加上一列，元素是对角线的准确率,用来画散点图
'''
acc, diag = accuracy(all_matrix)
all_matrix.loc[:, 'new_column'] = diag

all_matrix.to_csv(r'D:\学习\研二\小论文point cloud transformer\代码运行日志\lstm\all_matrix5.1.csv')
