# -*- coding:utf-8 -*-
import os
import pandas as pd

# 改后缀
path = r"D:\学习\研二\小论文point cloud transformer\数据集\低频率（weight=0）元素数据集补充\数据集csv-PmReTmXe"

files = os.listdir(path)
print(files)

for file in files:
    NewName = file.replace(".xyz", ".csv")
    os.renames(path + "\\" + file, path + "\\" + NewName)   # 重命名，"\\" 反斜杠需要转义


'''
求矩阵的准确率

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
all_matrix = pd.read_csv(rootPath + '/all_matrix2.csv')
print('all_matrix.iloc[:, :-1]', all_matrix.iloc[:, :-1])
# diag_vals = all_matrix.iloc[:, :-1].values.diagonal()  # 获取对角线上的元素集合成array

result = 0
for i in range(len(all_matrix)-1):
    diag_vals = all_matrix.iloc[i, i+1]
    print('diag_vals', diag_vals)
    weight_val = all_matrix.iloc[i+1, -1]
    result += diag_vals * weight_val

weights = all_matrix['weight']  # 获取最后一列
# print('(diag_vals * weights).sum()', (diag_vals * weights).sum())
print('(diag_vals * weights).sum()', result)
print('diag_vals', diag_vals)
print('weights.sum()', weights.sum())
# acc = (diag_vals * weights).sum()/weights.sum()
acc = result/weights.sum()
print('所有元素的准确率为：', acc)
'''