import torch
import torchvision
import numpy as np
import pandas as pd
import math
import csv
import seaborn as sns
import matplotlib.pyplot as plt

# df = pd.read_excel(r"new.csv")   # pd.read_excel().set_index('R2')
df = np.loadtxt(r"elements.csv", delimiter=',', unpack=True)  # 第一行是y，第二行是pred，unpack=True令矩阵转置
list_lebal = list(set(df[:, 0]))  # 除去重复
print('df:', df)
print('list_lebal:', list_lebal)
print('矩阵的行数：', df.shape[0])
print('矩阵的行向量长度：', df.shape[1])
n = 243
# 创建矩阵
outputs = np.zeros(shape=(n, len(list_lebal)), dtype=float)
true, pred = df[300:, 0], df[300:, -1]  # 总2850个，滑窗至(300,300+243)时准确率较好
# outputs = torch.ones(size=(n, n))
# true, pred = torch.tensor(df[:, 0]), torch.tensor(df[:, -1])
print('ture:', true, 'pred:', pred)


def preprocessing(pred, list_lebal):
    for i in range(n):
        for j in range(len(list_lebal)):
            outputs[i][j] = np.abs(min(list_lebal) / (np.abs(pred[i] - list_lebal[j])))
            # outputs[i][j] = 1/(np.abs(pred[i] - list_lebal[j]))  # 二者输出概率矩阵数量级差不多
    print('经过预处理的矩阵outputs：', outputs)
    return outputs


def softmax(X):
    X_exp = np.exp(X)
    # X_exp = torch.exp(X)
    # partition = X_exp.sum(dim=0, keepdim=True)  # 计算结果有无穷数inf
    # partition = X_exp.sum(axis=0, keepdim=True)  # 计算结果中仍然有无穷书inf
    # partition = np.sum(X_exp, axis=1)
    for row in range(X_exp.shape[0]):
        X_exp[row] /= np.sum(X_exp[row])
    return X_exp


preprocessing(pred, list_lebal)
outputs = softmax(outputs)
print('经过softmax处理后的矩阵outputs：', outputs)
print('outputs矩阵每行的概率之和为：', np.sum(outputs, axis=1))

# def net(X):
#     return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
'''
x.view(-1,n)表示不确定reshape成几行，但能确定是n列。
例如16个元素reshape成x.view(-1, 2)便是(8, 2)，x.view(-1, 4)便是(4, 4)。
torch.mm(a, b) 是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵。
'''


def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


def evaluate_accuracy(outputs, list_lebal):
    acc_sum = 0.0
    index = []
    acc_matrix = np.zeros(shape=(len(list_lebal), len(list_lebal)), dtype=float)
    for i in range(len(list_lebal)):
        k = 0
        for j in range(n):
            if true[j] == list_lebal[i]:
                # print('true中与list_lebal对应的索引为：', index.append(list_lebal.index(i)))
                acc_matrix[i, :] += outputs[j, :]
                k = k + 1
        acc_matrix[i, :] = acc_matrix[i, :] / k
    print('概率矩阵为：', acc_matrix)
    return


# evaluate_accuracy(outputs, list_lebal)  # 直接求平均概率密度，得到的结果各元素都很小很不理想


def accuracy(outputs, true, list_lebal):
    acc_matrix = np.zeros(shape=(len(list_lebal), len(list_lebal)), dtype=float)
    for i in range(len(list_lebal)):
        acc_num = 0
        for j in range(n):
            if true[j] == list_lebal[i]:
                k = np.argmax(outputs[j, :])
                acc_matrix[i, k] += 1
                acc_num += 1
        print('该行的匹配个数为：', acc_num)
        acc_matrix[i, :] = acc_matrix[i, :] / acc_num
    print('准确率矩阵为：', acc_matrix)
    return acc_matrix


acc_matrix = accuracy(outputs, true, list_lebal)


# def evaluate_accuracy(data_iter, outputs):
#     acc_sum, n = 0.0, 0
#     for X, y in data_iter:
#         # acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()  # .item()提高浮点数精度
#         acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
#         n += y.shape[0]
#     return acc_sum / n


def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def element_to_list():
    list0 = []
    # list1 = np.array(list_lebal)
    # list1 = list1[:, np.newaxis]
    list1 = list_lebal
    # list1 = [int(i) for i in list_lebal]
    print(list1)
    with open(r'Atomic_relative_mass.csv', 'r', encoding='utf8') as fp2:
        list2 = [i for i in csv.reader(fp2)]
        print('list2为：', list2)
    for i in range(len(list1)):
        for j in range(len(list2)):
            if np.abs(float(list1[i]) - float(list2[j][3])) < 0.000001:
                # 浮点数不能直接转整形int()，而且由于计算机存储逻辑，浮点数之间不要用==比较。例如0.5+0.5==1也会报错
                list0.append(list2[j][2])
    print('list_element为：', list0)
    return list0


list_element = element_to_list()


def heatmap(acc_matrix, list_element):
    heat_map = np.random.rand(len(list_lebal), len(list_lebal))  # 设置长宽
    fig, ax = plt.subplots(figsize=(9, 9))
    data = np.array(acc_matrix)
    # sns.heatmap(data, annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="YlOrRd")
    # columns是横坐标，index是纵坐标
    sns.heatmap(pd.DataFrame(np.round(acc_matrix, 2), columns=[list_element], index=[list_element]),
                annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True,
                square=True, cmap="YlGnBu")
    # ax.set_title('元素拟合热力图', fontsize=18)
    ax.set_ylabel('Pred', fontsize=18)
    ax.set_xlabel('True', fontsize=18)  # 横变成y轴，跟矩阵原始的布局情况是一样
    # ax.set_yticklabels([list_lebal], fontsize=18, rotation=360, horizontalalignment='right')
    # ax.set_xticklabels([list_lebal], fontsize=18, horizontalalignment='right')

    plt.show()
    return


heatmap(acc_matrix, list_element)
