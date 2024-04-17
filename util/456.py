# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

"""对特征张量降维可视化（2维）"""
# 指定文件路径
# file_path = r'D:\学习\研究生毕业论文\大论文\第四章补充\特征变化\x1.csv'
file_path = r'/mnt/Dataset/PT2_1/result/x1.csv'

# 读取CSV文件，假设文件中没有列名和索引
data = pd.read_csv(file_path, header=None)

# 应用PCA进行降维到2个主成分用于可视化
pca_visualization = PCA(n_components=2)
data_transformed_visualization = pca_visualization.fit_transform(data)

# 应用PCA进行降维到4个主成分用于信息丢失量的计算
pca = PCA(n_components=4)
data_transformed = pca.fit_transform(data)
data_reconstructed = pca.inverse_transform(data_transformed)

# 计算均方误差
mse = mean_squared_error(data, data_reconstructed)

# 计算方差解释比
explained_variance = np.sum(pca.explained_variance_ratio_)

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(data_transformed_visualization[:, 0], data_transformed_visualization[:, 1], alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Result (2 Components)')
plt.grid(True)

# 保存图像，分辨率设置为400 dpi
plt.savefig('PCA_Visualization.png', dpi=400)
plt.show()

print(f"MSE: {mse:.4f}")
print(f"Explained Variance Ratio: {explained_variance:.4f}")