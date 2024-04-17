# -*- coding:utf-8 -*-
"""
画出初始晶体结构csv文件和预测晶体结构obj文件的t-SNE可视化图
"""
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def spilt(file):
    data = pd.read_csv(file, sep=r'\\s+', engine='python')  # skiprows=[0,1]
    df = pd.DataFrame(data[data.columns[0]].str.split(' ', n=12))  # 设置参数expand=True可使其分列而不是1列
    for i in df.columns:
        if df[i].count() == 0:
            df.drop(labels=i, axis=1, inplace=True)
    df.rename(columns={list(df)[0]: 'col1_new_name'}, inplace=True)  # 重命名列名
    df_split13 = df['col1_new_name'].apply(lambda x: pd.Series(x))  # 拆开列表1列变13列
    df_split13 = df_split13.replace(r'^\s*$', np.nan, regex=True)  # 用NaN代替空格
    split_data = df_split13.T.apply(lambda x: pd.Series(x.dropna().values)).T  # 去除空值向左移
    split_data.columns = ['elements', 'x', 'y', 'z']
    # 掐头去尾
    noelements_data = split_data.drop('elements', axis=1)
    notitle_data = noelements_data.iloc[1:, 0:]  # [1:, 0:]
    df1 = notitle_data.astype('float32', errors='ignore')
    return df1

# 读取csv文件
file_csv = r'D:\学习\研二\小论文point cloud transformer\数据集\各元素数据集（未分割）\1000000.csv'
# df1 = pd.read_csv(file_csv, header=None)
df1 = spilt(file_csv)
X1 = df1.iloc[1:, 1:].values
print('X1:', X1)

# 读取obj文件
file_obj = r'D:\学习\研一\点云处理后数据集\obj\修改坐标带颜色的obj\0,255,0.obj'
df2 = pd.read_csv(file_obj, header=None, delim_whitespace=True)
X2 = df2.iloc[:, 1:].values
print('X2:', X2)

# 初始化t-SNE模型
tsne = TSNE(n_components=3, perplexity=0.5, learning_rate=200)

# 转换正确坐标散点
X1_tsne = tsne.fit_transform(X1)

# 转换预测坐标散点
X2_tsne = tsne.fit_transform(X2)

# 绘制正确坐标散点
plt.scatter(X1_tsne[:, 0], X1_tsne[:, 1], color='blue', label='Correct')

# 绘制预测坐标散点
plt.scatter(X2_tsne[:, 0], X2_tsne[:, 1], color='red', label='Prediction')

# 添加图例和标题
plt.legend()
plt.title('Comparison of Correct and Prediction Scatter Plots')

# 显示图形
plt.show()