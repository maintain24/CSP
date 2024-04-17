# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import torch
import math
import scipy.linalg as linalg
import glob
import os
import random
import copy
import tqdm

'''
对csv文件中的坐标数据批量读取，并生成大量随机翻转的npy文件用以训练
'''

# file_path = r'D:\学习\研一\点云处理后数据集\csv'
file_path = r'D:\software\PythonProject\Python\point-transformer2\dataset\csv'
files = glob.glob(os.path.join(file_path, "*.csv"))        # 构建文件列表
files.sort()                                               # 为文件排序
print('文件列表files：', files)
object_number = files.__len__()
print("object number :", object_number)


# 专门针对晶体cif转成的csv文件进行格式分割
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
    return split_data


# 将所有点集平移，使中心点位于原点
def center_points(df):
    center = df.mean()
    df['x'] -= center['x']
    df['y'] -= center['y']
    df['z'] -= center['z']
    return df


# 缩小点集到（±5，±5，±5）的空间中
def shrink_volume(df):
    # 计算边界框大小
    min_x, max_x = df['x'].min(), df['x'].max()
    min_y, max_y = df['y'].min(), df['y'].max()
    min_z, max_z = df['z'].min(), df['z'].max()
    box_size = max(max_x - min_x, max_y - min_y, max_z - min_z)
    # 缩放边界框大小
    scale = 10 / box_size
    df['x'] *= scale
    df['y'] *= scale
    df['z'] *= scale
    return df


def safe_float_convert(x):
    try:
        float(x)
        return True # numeric, success!
    except ValueError:
        return False # not numeric
    except TypeError:
        return False # null type
# mask = df[1:].map(safe_float_convert)  # 使用map()将新函数按元素的方式映射到数据框的B列，并创建一个布尔掩码
# df = df.loc[mask]


# 旋转矩阵 欧拉角
def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix


# 空间三维坐标系旋转计算
# 分别是x,y和z轴,也可以自定义旋转轴
axis_x, axis_y, axis_z = [1, 0, 0], [0, 1, 0], [0, 0, 1]
rand_axis = [0, 0, 1]
# 旋转角度
yaw = math.pi/180
# 返回旋转矩阵
rot_matrix = rotate_mat(rand_axis, yaw)
print(rot_matrix)


# data = pd.read_csv(r"D:\software\PythonProject\Python\pointnet2\data\stanford_indoor3d\Area_5_conferenceRoom_1.csv")
# 读取单个文件,将它们从原始未分割csv文件转换成分割好、坐标颜色label的格式齐全文件
def main(file, iteration):
    # data = pd.read_csv(file, sep=r'\s{2,}', engine='python')  # 使用正则表达式模式\s{2,}作为分隔符。此正则表达式匹配 2 个或更多空白字符。
    data = pd.read_csv(file, sep=r'\\s+', engine='python')  # skiprows=[0,1]
    # data = df[0].apply(lambda x: ' '.join(x.split()))  # Replacing uneven spaces with single space
    print('data:', data)
    df = pd.DataFrame(data[data.columns[0]].str.split(' ', n=12))  # 设置参数expand=True可使其分列而不是1列
    for i in df.columns:
        if df[i].count() == 0:
            df.drop(labels=i, axis=1, inplace=True)
    # split_data = df.apply(lambda x: np.NaN if str(x).isspace() else x)  # 将空值替换为NaN值
    # split_data = pd.DataFrame(df[:, :].values.tolist())  # array[]转dataframe
    df.rename(columns={list(df)[0]: 'col1_new_name'}, inplace=True)  # 重命名列名
    print('split_data[3,0]:', df['col1_new_name'][3][0])  # 第一列第四行的列表第1个元素

    df_split13 = df['col1_new_name'].apply(lambda x: pd.Series(x))  # 拆开列表1列变13列
    # split_data = df_split13.apply(pd.to_numeric)  # 把'str'类型数据转为'Float'型
    # split_data[col] = data[col].apply(pd.to_numeric, errors='coerce')  # 把'str'类型数据转为'Float'型
    # print('空格数据类型是：', split_data[3][2].dtype.dtypes)  # AttributeError: 'float' object has no attribute 'dtype'
    df_split13 = df_split13.replace(r'^\s*$', np.nan, regex=True)  # 用NaN代替空格
    # split_data[:][:] = split_data[:][:].apply(lambda x: np.NaN if str(x).isspace() else x)  # 用NaN代替空格
    # out = split_data.T.apply(lambda x: sorted(x, key=pd.isnull)).T  # 去除空值向左移
    split_data = df_split13.T.apply(lambda x: pd.Series(x.dropna().values)).T  # 去除空值向左移
    print('列名：', split_data.columns.values.tolist())  # 列名称
    # out.rename(columns={'0': 'elements', '1': 'x', '2': 'y', '3': 'z'}, inplace=True)
    split_data.columns = ['elements', 'x', 'y', 'z']
    print('列名：', split_data.columns.values.tolist())  # 列名称
    print('split_data:', split_data)

    # 掐头去尾
    noelements_data = split_data.drop('elements', axis=1)
    notitle_data = noelements_data.iloc[1:, 0:]  # [1:, 0:]
    print('notitle_data', notitle_data)

    # 修改x、y、z值
    df1 = notitle_data.astype('float32', errors='ignore')
    yaw = ((iteration - 1) * 1/12 * math.pi) / 180  # 旋转
    df1 = df1.dot(rot_matrix)  # 旋转
    df1.columns = ['x', 'y', 'z']
    df1 = center_points(df1)
    df1 = shrink_volume(df1)
    # df1['x'] = df1['x'] + 12 * (iteration - 1)
    print('坐标旋转df1_rotate:', df1)

    # 添加r、g、b及label列
    df2 = df1
    color = list(np.random.choice(range(256), size=3))
    print('rgb三色值color:', color)
    df2 = df2.assign(r=color[0], g=color[1], b=color[2])
    df2 = df2.assign(l=iteration-1)  # label列
    cut_data = df2
    print('cut_data:', cut_data)
    return cut_data


if __name__ == '__main__':
    case1 = True
    # case1 = False
    case2 = True
    # case2 = False
    # case3 = True
    case3 = False
    '''
    情况1：生成大的混乱数据集，在xyz三轴随机输入n类DataFrame
    '''
    if case1:
        # 先定义字典，df{i}对应file，给所有file处理数据再赋label
        dfs = {}
        for i in range(object_number):
            df_name = f"df{i}"
            filename = files[i]
            dfs[df_name] = main(files[i], i+1)

        # 打乱字典键的顺序，合并数据,堆叠坐标
        for k in range(object_number):
            keys = list(dfs.keys())
            random.shuffle(keys)
            all_data = pd.DataFrame(columns=['x', 'y', 'z'])
            # iteration = 1
            for l in range(5):  # Z轴
                for i in range(object_number):  # x轴
                    j = 0
                    for df_name in random.sample(keys, len(keys)):
                        cut_data = dfs[df_name]
                        cut_data['x'] = cut_data['x'] + 10.5 * i
                        cut_data['y'] = cut_data['y'] + 10.5 * j
                        cut_data['z'] = cut_data['z'] + 10.5 * l
                        all_data = all_data.append(cut_data, ignore_index=True)
                        cut_data['x'] = cut_data['x'] - 10.5 * i
                        cut_data['y'] = cut_data['y'] - 10.5 * j
                        cut_data['z'] = cut_data['z'] - 10.5 * l
                        j = j + 1
            # all_data = all_data.sample(frac=1, replace=False)  # 打乱所有行，frac=1来抽取全部行，并设置replace=False来保证不重复抽取

            # all_data = all_data.drop('elements', axis=1)
            print(f'------------------------------------第{k+1}个文件--------------------------------------')
            # all_data.to_csv(f'D:\software\PythonProject\Python\point-transformer2\dataset\out{k+1}.csv',
            #                 index=0)  # 将dataframe输出到csv文件中，不设置表头header
            print('all_data:', all_data)
            # df.to_csv(r'D:\software\PythonProject\Python\point-transformer2\dataset\out.csv',header=None)

            # 原代码，设置数据默认浮点Tensor
            # torch.set_default_tensor_type(torch.FloatTensor)
            # np.save(r"D:\software\PythonProject\Python\pointnet2\data\stanford_indoor3d\Area_5_conferenceRoom_1", df)

            # 设置pytorch中默认的浮点类型
            # torch.set_default_tensor_type(torch.FloatTensor)  # 设置torch.cuda.FloatTensor可将数据迁移至cuda
            # 保存训练集npy文件
            np.save(f"D:\software\PythonProject\Python\point-transformer2\dataset\Area_2_conferenceRoom_{k+1}.npy", all_data)
            # 保存测试集npy文件
            # np.save(f"D:\software\PythonProject\Python\point-transformer2\dataset\Area_5_conferenceRoom_{k+1}.npy", all_data)

    '''
    情况2：生成一列的不同类型数据集，用作测试集(没打乱顺序)
    '''
    if case2:
        for i in range(5):
            all_data = pd.DataFrame(columns=['x', 'y', 'z'])
            iteration = 1
            for file in files:
                cut_data = main(file, iteration)
                cut_data['x'] = cut_data['x'] + 10.5 * (iteration - 1)
                all_data = all_data.append(cut_data, ignore_index=True)
                iteration += 1

            print(f'------------------------------------第{i+1}个文件--------------------------------------')
            print('all_data:', all_data)
            np.save(f"D:\software\PythonProject\Python\point-transformer2\dataset\Area_5_conferenceRoom_{i}.npy",
                    all_data)

    '''
    情况3：生成大的数据集，但取消了shuffle，在xy轴随机输入n类DataFrame
    '''
    if case3:
        iteration = 1
        for file in files:
            df = main(file, iteration)
            all_data = pd.DataFrame(columns=['x', 'y', 'z'])
            for i in range(object_number):  # x轴
                df['x'] = df['x'] + 10.5
                for j in range(object_number):  # y轴
                    cut_data = copy.deepcopy(df)
                    cut_data['y'] = cut_data['y'] + 10.5 * j
                    all_data = all_data.append(cut_data, ignore_index=True)
            print(f'------------------------------------第{iteration}个文件--------------------------------------')
            print('all_data:', all_data)
            np.save(f"D:\software\PythonProject\Python\point-transformer2\dataset\Area_1_conferenceRoom_{iteration}.npy",
                    all_data)
            iteration += 1
