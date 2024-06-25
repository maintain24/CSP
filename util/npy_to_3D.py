# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

# 文件路径
filename = r'D:\software\PythonProject\Python\point-transformer6\dataset\Area_2_conferenceRoom_1.npy'

# 加载.npy文件
data = np.load(filename)

print(data)
print(data.shape)

# 确保数据至少有7列
if data.shape[1] < 7:
    print("数据集不包含足够的列来提取坐标和第7列的颜色标签。")
    exit()

# 提取坐标和第7列的颜色标签
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
categories = data[:, 6].astype(int)  # 确保类别标签是整数类型

# 选择一个适合期刊论文的颜色映射，这里使用tab20
cmap = plt.get_cmap('tab20', len(np.unique(categories)))

# 定义不同的角度组合
angles = [(30, 30), (60, 45), (90, 60)]

# 创建一个颜色条目列表，用于scatter函数中的颜色参数
category_colors = [cmap(i) for i in range(len(np.unique(categories)))]

# 为每个角度创建一个单独的图形，并设置高分辨率
for i, angle in enumerate(angles):
    plt.figure(figsize=(10, 8), dpi=300)

    ax = plt.axes(projection='3d')

    # 根据第7列的值自动分配颜色
    # 使用ListedColormap确保颜色映射是循环的
    scatter = ax.scatter(x, y, z, c=categories, s=5, marker='o', cmap=ListedColormap(category_colors))

    # 设置轴标签
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # 设置视图角度
    ax.view_init(elev=angle[0], azim=angle[1])

    plt.title(f'View from Elevation {angle[0]} Azimuth {angle[1]}')

    # 保存图形
    plt.savefig(f'D:\\学习\\研三\\小论文第一篇（英文）\\CES\\审稿意见\\补充实验结果\\训练测试数据集 可视化\\train_view_{i + 1}.png')

    # 显示图形
    plt.show()

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=ListedColormap(category_colors), norm=plt.Normalize(vmin=np.min(categories), vmax=np.max(categories)))
    plt.colorbar(sm, ax=ax, pad=0.1)