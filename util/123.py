# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import csv

# 创建空的dataframe
df = pd.DataFrame()
xyz_file = r'D:\学习\研二\小论文point cloud transformer\代码运行日志\PT2\coordinate.csv'
coord = np.random.rand(0, 3)
p0 = np.random.rand(0, 3)
p1 = np.random.rand(0, 3)
p2 = np.random.rand(0, 3)
p3 = np.random.rand(0, 3)
p4 = np.random.rand(0, 3)
p5 = np.random.rand(0, 3)
# 循环产生数据
for i in range(10):
    # 产生7种numpy数据
    coord_np = np.random.rand(i + 1, 3)
    p0_np = np.random.rand(i + 2, 3)
    p1_np = np.random.rand(i + 3, 3)
    p2_np = np.random.rand(i + 4, 3)
    p3_np = np.random.rand(i + 5, 3)
    p4_np = np.random.rand(i + 6, 3)
    p5_np = np.random.rand(i + 7, 3)

    # 将数据填充到临时dataframe中
    # tmp_df = pd.DataFrame()
    # tmp_df = pd.concat([tmp_df, pd.DataFrame(coord_np)], axis=1)
    # tmp_df = pd.concat([tmp_df, pd.DataFrame(p0_np)], axis=1)
    # tmp_df = pd.concat([tmp_df, pd.DataFrame(p1_np)], axis=1)
    # tmp_df = pd.concat([tmp_df, pd.DataFrame(p2_np)], axis=1)
    # tmp_df = pd.concat([tmp_df, pd.DataFrame(p3_np)], axis=1)
    # tmp_df = pd.concat([tmp_df, pd.DataFrame(p4_np)], axis=1)
    # tmp_df = pd.concat([tmp_df, pd.DataFrame(p5_np)], axis=1)

    # # 将临时dataframe添加到大dataframe中
    # df = pd.concat([df, tmp_df], axis=1)

    # 方法2
    # with open(xyz_file, mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     for row in coord_np:
    #         writer.writerow([row[0], row[1], row[2]])
    #     for row in p0_np:
    #         writer.writerow(['', '', '', row[0], row[1], row[2], ''])
    #     for row in p1_np:
    #         writer.writerow(row[6:9])
    #     for row in p2_np:
    #         writer.writerow(row[9:12])
    #     for row in p3_np:
    #         writer.writerow(row[12:15])
    #     for row in p4_np:
    #         writer.writerow(row[15:18])
    #     for row in p5_np:
    #         writer.writerow(row[18:21])

    # 方法3
    # data = np.random.rand(1, 21)
    #
    # # 选择要保存的列
    # coord_np_cols = data[:, 0:3]
    # p0_np_cols = data[:, 3:6]
    # p1_np_cols = data[:, 6:9]
    # p2_np_cols = data[:, 9:12]
    # p3_np_cols = data[:, 12:15]
    # p4_np_cols = data[:, 15:18]
    # p5_np_cols = data[:, 18:21]
    #
    # # 将所选列转换为DataFrame对象
    # df_coord = pd.DataFrame(coord_np_cols)
    # df_p0 = pd.DataFrame(p0_np_cols)
    # df_p1 = pd.DataFrame(p1_np_cols)
    # df_p2 = pd.DataFrame(p2_np_cols)
    # df_p3 = pd.DataFrame(p3_np_cols)
    # df_p4 = pd.DataFrame(p4_np_cols)
    # df_p5 = pd.DataFrame(p5_np_cols)
    #
    # # 打开csv文件并将DataFrame对象追加到文件中
    # with open(xyz_file, mode='a', newline='') as file:
    #     df_coord.to_csv(file, header=False, index=False, mode='a', sep=',', float_format='%.6f')
    #     df_p0.to_csv(file, header=False, index=False, mode='a', sep=',', float_format='%.6f')
    #     df_p1.to_csv(file, header=False, index=False, mode='a', sep=',', float_format='%.6f')
    #     df_p2.to_csv(file, header=False, index=False, mode='a', sep=',', float_format='%.6f')
    #     df_p3.to_csv(file, header=False, index=False, mode='a', sep=',', float_format='%.6f')
    #     df_p4.to_csv(file, header=False, index=False, mode='a', sep=',', float_format='%.6f')
    #     df_p5.to_csv(file, header=False, index=False, mode='a', sep=',', float_format='%.6f')

    #方法4
    coord = np.append(coord, coord_np, axis=0)
    p0 = np.append(p0, p0_np, axis=0)
    p1 = np.append(p1, p1_np, axis=0)
    p2 = np.append(p2, p2_np, axis=0)
    p3 = np.append(p3, p3_np, axis=0)
    p4 = np.append(p4, p4_np, axis=0)
    p5 = np.append(p5, p5_np, axis=0)
    # print('coord>>>>>>>>>>>>>', coord)
    # print('p0>>>>>>>>>>>>>', p0)
    # print('p1>>>>>>>>>>>>>', p1)
    # print('p2>>>>>>>>>>>>>', p2)
# np.savetxt(xyz_file, np.concatenate((coord, p0, p1, p2, p3, p4, p5), axis=1),
#            delimiter=',', fmt='%.6f')  # 保存预测坐标数据集
# 将每个ndarray数据转化为DataFrame格式
df1 = pd.DataFrame(coord)
df2 = pd.DataFrame(p0)
df3 = pd.DataFrame(p1)
df4 = pd.DataFrame(p2)
df5 = pd.DataFrame(p3)
df6 = pd.DataFrame(p4)
df7 = pd.DataFrame(p5)

# 使用concat函数按列拼接这7个DataFrame
df = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=1)

# 将拼接后的DataFrame保存到csv文件中
df.to_csv(xyz_file, index=False, header=False)
# # 输出df的形状和前5行数据
# print(df.shape)
# print(df)
# df.to_csv(xyz_file, index=False)