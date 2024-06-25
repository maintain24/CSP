# -*- coding:utf-8 -*-
'''
softmax.py文件对单个晶体进行分类评估，softmax_all.py对大量晶体全元素进行分类评估
'''

import torch
import torchvision
import numpy as np
import pandas as pd
import math
import csv
import os
import seaborn as sns
import matplotlib.pyplot as plt

curPath = os.path.abspath(os.path.dirname(__file__))
file = curPath + '/algorithms/elements.csv'
file_out = curPath + '/algorithms/Atomic_relative_mass.csv'
df_atomic = pd.read_csv(file_out, keep_default_na=False, encoding='utf-8',
                        names=['序号', '元素', 'elements', '相对原子质量'])
# df_atomic = pd.read_csv(r'Atomic_relative_mass.csv', keep_default_na=False, encoding='utf-8',
# names=['序号','元素', 'elements', '相对原子质量'])

list2 = df_atomic['相对原子质量'].to_list()
list2 = list(set(list2))  # 去掉重复项

list3 = np.append(list2, 'weight')
list4 = df_atomic['elements'].to_list()
list4 = list(set(list4))  # 得到图的横纵标题
all_matrix = pd.DataFrame(np.zeros(shape=(len(list2), len(list2) + 1), dtype=float),
                          index=list2, columns=list3)  # 最右边留一列权重数


def softmax_all(file, all_matrix, n):  # n是晶体原子数
    df = np.loadtxt(file, delimiter=',', unpack=True)  # 第一行是y，第二行是pred，unpack=True令矩阵转置
    list_label = list(set(df[:, 0]))  # 除去重复
    outputs = np.zeros(shape=(n, len(list_label)), dtype=float)
    true, pred = df[int(len(df) * 0.2):, 0], df[int(len(df) * 0.2):, -1]  # 总2850个，滑窗至(300,300+243)时准确率较好

    def preprocessing(pred, n):
        for i in range(n):
            for j in range(len(list_label)):
                outputs[i][j] = np.abs(min(list_label) / (np.abs(pred[i] - list_label[j])))
                # outputs[i][j] = 1/(np.abs(pred[i] - list_lebal[j]))  # 二者输出概率矩阵数量级差不多
        # print('经过预处理的矩阵outputs：', outputs)
        return outputs

    def softmax(X):
        X_exp = np.exp(X)
        for row in range(X_exp.shape[0]):
            X_exp[row] /= np.sum(X_exp[row])
        return X_exp

    def accuracy(outputs, true, list_label):
        acc_matrix = np.zeros(shape=(len(list_label), len(list_label)), dtype=float)
        acc_num_list = []
        for i in range(len(list_label)):
            acc_num = 0
            for j in range(n):
                if true[j] == list_label[i]:
                    k = np.argmax(outputs[j, :])
                    acc_matrix[i, k] += 1
                    acc_num += 1
            # print('该行的匹配个数为：', acc_num)
            acc_num_list.append(acc_num)
            # acc_matrix[i, :] = acc_matrix[i, :] / acc_num  # 小矩阵除以权重，但精度损失太大
            acc_matrix[i, :] = acc_matrix[i, :]  # 修改为不除分母weight，避免变成分数，保留精度
        print('准确率矩阵acc_matrix为：', acc_matrix)
        # 计算准确率矩阵的准确率
        diag_vals = np.diag(acc_matrix)
        acc = diag_vals.sum() / sum(acc_num_list)  # array的各元素相加和list的各元素相加用法不一样
        return acc_matrix, acc_num_list, acc

    def all_accuracy_matrix(acc_matrix, all_matrix, acc_num_list):  # [76, 20, 8, 100, 12, 27]
        list1 = list_label  # [1.00794, 12.016999600964724, 14.006699563686787, 15.99939928501296, 26.981537860934257, 30.973762]
        this_type_str = type(all_matrix)
        if this_type_str is np.ndarray:
            all_matrix = pd.DataFrame(all_matrix, index=list2, columns=list3)
        acc_matrix = pd.DataFrame(acc_matrix)
        # all_matrix = pd.DataFrame(np.zeros(shape=(len(list2), len(list2)+1), dtype=float),
        #                            index=list2,columns=list3)  # 最右边留一列权重数

        for i in range(len(list1)):
            # all_matrix.iloc[k, -1] += acc_num_list[i]  # 每行的权重数 [76, 20, 8, 101, 12, 26]
            # print('行名：', '{}'.format(list1[i]))
            for k in range(len(list2)):
                if np.abs(float(list1[i]) - float(list2[k])) < 0.000001:
                # if np.abs(float(list1[i])) == np.abs(float(list2[k])):
                #     print('np.abs(float(list1[i])):', np.abs(float(list1[i])))  # 用于检查大小矩阵两者是否匹配
                #     print('np.abs(float(list2[k])):', np.abs(float(list2[k])))

                    for j in range(len(list1)):
                        for l in range(len(list2)):
                            if np.abs(float(list1[j]) - float(list2[l])) < 0.000001:
                            # if np.abs(float(list1[j])) == np.abs(float(list2[l])):
                                # all_matrix.iloc[k, l] = (acc_matrix.iloc[i, j] * acc_num_list[i]
                                #                          + all_matrix.iloc[k, l]*all_matrix.iloc[k, -1]) / \
                                #                         (all_matrix.iloc[k, -1] + acc_num_list[i])
                                all_matrix.iloc[k, l] += acc_matrix.iloc[i, j]
                                # 修改为累加，不产生分数，保留了精度
                            else:
                                all_matrix.iloc[k, l] = all_matrix.iloc[k, l]
                    all_matrix.iloc[k, -1] += acc_num_list[i]  # 对应的标签行，在最后一列进行权数累加
                else:
                    all_matrix.iloc[k, -1] = all_matrix.iloc[k, -1]
                    for j in range(len(list1)):
                        for l in range(len(list2)):
                            all_matrix.iloc[k, l] = all_matrix.iloc[k, l]

        # print('all_matrix:', all_matrix)
        return all_matrix # 之后list1要改

    outputs2 = preprocessing(pred, n)
    outputs3 = softmax(outputs2)
    # print('经过softmax处理后的矩阵outputs：', outputs3)
    # print('outputs矩阵每行的概率之和为：', np.sum(outputs2, axis=1))
    acc_matrix, acc_num_list, acc = accuracy(outputs3, true, list_label)

    # 对于acc值小于0.5的准确率矩阵，采取记录+跳过措施
    if acc < 0.65:
        all_matrix = all_matrix
    else:
        all_matrix = all_accuracy_matrix(acc_matrix, all_matrix, acc_num_list)
    # all_matrix = all_matrix.drop(['weight'], axis=1).values
    # heatmap(all_matrix, list4)
    return all_matrix, acc


# softmax_all(file, n=243)

def df_indextransform(all_matrix, list_label):
    # all_matrix = all_matrix.rename(index={list_label}, columns={list_label})
    all_matrix_output = all_matrix.drop(['weight'], axis=1)
    all_matrix_output.columns = list_label
    all_matrix_output.index = list_label
    all_matrix_output = all_matrix_output.values

    return all_matrix_output


def heatmap(acc_matrix, list_element):
    """如果上面softmax_all输出保留了最后一列weight，则需要在这删除该列，虽然是acc_matrix，但实际是all_matrix。"""
    pic_acc_matrix = acc_matrix
    heat_map = np.random.rand(len(list_element), len(list_element))  # 设置长宽
    fig, ax = plt.subplots(figsize=(36, 36))
    data = np.array(acc_matrix)
    # sns.heatmap(data, annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="YlOrRd")
    # columns是横坐标，index是纵坐标
    # sns.heatmap(pd.DataFrame(np.round(pic_acc_matrix, 2), columns=list_element, index=list_element),
    #             annot=False, vmax=1, vmin=0, xticklabels=list4, yticklabels=list4, linewidths=0,
    #             square=True, cmap="YlGnBu")
    sns.heatmap(pd.DataFrame(np.round(pic_acc_matrix, 2)),
                annot=False, vmax=1, vmin=0, xticklabels=list4, yticklabels=list4, linewidths=0,
                square=True, cmap="YlGnBu")
    # ax.set_title('元素拟合热力图', fontsize=18)
    ax.set_ylabel('Pred', fontsize=18)
    ax.set_xlabel('True', fontsize=18)  # 横变成y轴，跟矩阵原始的布局情况是一样
    # ax.set_yticklabels([list_lebal], fontsize=18, rotation=360, horizontalalignment='right')
    # ax.set_xticklabels([list_lebal], fontsize=18, horizontalalignment='right')
    fig.savefig(curPath + '/heatmap.jpg', dpi=400, bbox_inches='tight')
    plt.show()
    return


