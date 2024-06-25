import os
import pandas as pd
import numpy as np
import math
import glob
import random
import copy
from ase import io
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import OneHotEncoder


def find_nearest_samples(x: float, csv_path: str, num_samples: int = 15):
    df = pd.read_csv(csv_path)
    bandgap_column = 'outputs.pbe.bandgap'
    df['diff'] = (df[bandgap_column] - x).abs()
    df_sorted = df.sort_values(by='diff')
    nearest_samples = df_sorted.iloc[:2 * num_samples + 1]

    for index, row in nearest_samples.iterrows():
        print(f"qmof_id: {row['qmof_id']}, bandgap: {row[bandgap_column]}")

    return nearest_samples['qmof_id'].tolist(), nearest_samples[bandgap_column].tolist()


def convert_cif_to_xyz(cif_directory: str, output_directory: str, file_ids: list):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for file_id in file_ids:
        cif_file = os.path.join(cif_directory, f"{file_id}.cif")
        xyz_file = os.path.join(output_directory, f"{file_id}.xyz")

        if os.path.exists(cif_file):
            structure = io.read(cif_file)
            io.write(xyz_file, structure)
            print(f"Converted {cif_file} to {xyz_file}")
        else:
            print(f"File {cif_file} does not exist")


def center_points(df):
    df_centered = df - df.mean()
    return df_centered


def shrink_volume(df):
    scale_factor = 0.5
    df_scaled = df * scale_factor
    return df_scaled


def calculate_nearest_distances(df: pd.DataFrame) -> pd.DataFrame:
    # 计算所有原子坐标之间的距离矩阵
    dist_matrix = distance_matrix(df[['x', 'y', 'z']], df[['x', 'y', 'z']])

    # 初始化一个空列表来存储每个原子的最近三个距离
    nearest_distances = []

    # 遍历每个原子的距离行
    for i, row in enumerate(dist_matrix):
        # 排除与自身的距离（通常是0），然后排序剩余的距离
        sorted_distances = sorted(row[row != 0])  # 排除自身的距离

        # 检查是否至少有三个其他原子
        if len(sorted_distances) < 3:
            raise ValueError(f"原子 {i} 没有足够的其他原子来计算最近的三个距离。")

        # 添加最近的三个距离
        nearest_distances.append(sorted_distances[:3])

    # 将结果转换为DataFrame
    nearest_distance_df = pd.DataFrame(nearest_distances, columns=['dist1', 'dist2', 'dist3'])
    print(f"Nearest distance DataFrame shape: {nearest_distance_df.shape}")
    # print(f"Nearest distance DataFrame with NaN values:\n{nearest_distance_df[nearest_distance_df.isna().any(axis=1)]}")
    return nearest_distance_df


def main(file: str, iteration: int, bandgap: float, output_csv_directory: str):
    data = pd.read_csv(file, sep=r'\\s+', engine='python', header=None)
    print(f"Data read from {file}:")
    print(data.head())  # 输出前几行进行调试

    # 删除前两行
    data = data.iloc[2:]
    print("Data after removing the first two lines:")
    print(data.head())

    # 对第一列进行拆分
    df = data[0].str.split(expand=True)  # 使用 expand=True 参数进行分列
    print("Data after splitting:")
    print('df.shape', df.shape)
    print(df.head())  # 输出前几行进行调试

    # 重命名列名
    df.columns = ['elements', 'x', 'y', 'z'] + [f'col{i}' for i in range(4, df.shape[1])]
    df = df[['elements', 'x', 'y', 'z']]

    print("Data after final renaming of columns:")
    print(df.head())  # 输出前几行进行调试

    # 首先，将 'elements' 列与数值列分开处理
    elements_col = df[['elements']]
    # 选择数值列进行后续的数值操作
    numeric_df = df[['x', 'y', 'z']]
    # 转换坐标为浮点数
    numeric_df = numeric_df.astype(float)

    # 旋转矩阵计算
    yaw = ((iteration - 1) * 1 / 12 * math.pi) / 180
    rot_matrix = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                           [math.sin(yaw), math.cos(yaw), 0],
                           [0, 0, 1]])

    numeric_df[['x', 'y', 'z']] = numeric_df[['x', 'y', 'z']].dot(rot_matrix)
    numeric_df = center_points(numeric_df)
    df_scaled = shrink_volume(numeric_df)

    # 将 'elements' 列添加回处理后的 DataFrame，并重置索引，避免空两行错位
    df = pd.concat([elements_col, df_scaled], axis=1).reset_index(drop=True)
    print(f"df shape: {df.shape}")

    # 计算最近的三个距离并添加到数据框中
    distance_df = calculate_nearest_distances(df[['x', 'y', 'z']])
    print(f"Nearest distance DataFrame shape: {distance_df.shape}")
    # df = df.join(distance_df)
    # 确保索引一致后拼接
    df = pd.concat([df, distance_df], axis=1)

    # 添加标签列
    df = df.assign(label=bandgap)

    # 输出到CSV文件
    output_file = os.path.join(output_csv_directory, os.path.basename(file).replace('.xyz', '_cut.csv'))
    df.to_csv(output_file, index=False)

    print(f"Saved cut data to {output_file}")
    return df


def process_samples(x: float, csv_path: str, cif_directory: str, output_directory: str, output_csv_directory: str):
    nearest_ids, bandgap_values = find_nearest_samples(x, csv_path)
    convert_cif_to_xyz(cif_directory, output_directory, nearest_ids)

    if not os.path.exists(output_csv_directory):
        os.makedirs(output_csv_directory)

    for idx, file_id in enumerate(nearest_ids):
        xyz_file = os.path.join(output_directory, f"{file_id}.xyz")
        if os.path.exists(xyz_file):
            cut_data = main(xyz_file, 1, bandgap_values[idx], output_csv_directory)
        else:
            print(f"Converted file {xyz_file} does not exist")


# process_samples(1.5, r'D:\\学习\\研三\\晶体性能预测\\qmof_database\\qmof_database\\qmof.csv',
#                 r'D:\学习\研三\晶体性能预测\qmof_database\qmof_database\relaxed_structures',
#                 r'D:\\学习\\研三\\晶体性能预测\\converted_xyz',
#                 r'D:\\学习\\研三\\晶体性能预测\\cut_data_csv')
#
# # 运行元素编码处理函数
# process_csv_files(r'D:\学习\研三\晶体性能预测\cut_data_csv',
#                   r'D:\学习\研三\晶体性能预测\processed_data_csv')

# 定义处理CSV文件的函数
def process_csv_files(input_folder: str, output_folder: str):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有CSV文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)

            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 提取元素信息
            elements = df.iloc[:, 0]

            # 独热编码元素信息
            encoder = OneHotEncoder(sparse_output=False)
            encoded_elements = encoder.fit_transform(elements.values.reshape(-1, 1))
            element_df = pd.DataFrame(encoded_elements, columns=encoder.get_feature_names_out(['element']))

            # 重排数据
            xyz = df.iloc[:, 1:4]
            dist = df.iloc[:, 4:7]
            performance_label = df.iloc[:, 7]
            rearranged_df = pd.concat([xyz, dist, element_df, performance_label], axis=1)

            # 保存处理后的数据
            output_file_path = os.path.join(output_folder, filename)
            rearranged_df.to_csv(output_file_path, index=False)
            print(f'Processed {filename} and saved to {output_file_path}')

            # 保存编码器以备解码使用
            encoder_path = os.path.join(output_folder, 'encoder.pkl')
            pd.to_pickle(encoder, encoder_path)


def process_csv_files_2(input_folder: str, output_folder: str):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 读取元素信息文件
    elements_info_path = r"D:\\software\\PythonProject\\Python\\point-transformer6\\dataset\\elements\\elements.xlsx"
    elements_info_df = pd.read_excel(elements_info_path)
    element_to_mass = dict(zip(elements_info_df.iloc[:, 2], elements_info_df.iloc[:, 3]))

    # 遍历输入文件夹中的所有CSV文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)

            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 提取元素信息并转换为原子相对质量
            elements = df.iloc[:, 0]  # 假设第一列是元素名称
            performance_label = df.iloc[:, -1]  # 假设最后一列是性能标签
            df['element_mass'] = elements.map(lambda x: element_to_mass.get(x, np.nan))  # 使用字典映射元素到质量

            # 重排数据，这里假设除了元素列之外的列是坐标和性能标签
            xyz = df.iloc[:, 1:7]  # 假设第2到第7列是坐标、距离feature

            rearranged_df = pd.concat([xyz, df['element_mass'], performance_label], axis=1)

            # 保存处理后的数据
            output_file_path = os.path.join(output_folder, filename)
            rearranged_df.to_csv(output_file_path, index=False)
            print(f'Processed {filename} and saved to {output_file_path}')


# 定义解码元素的函数
def decode_elements(encoded_elements: pd.DataFrame, encoder: OneHotEncoder) -> pd.Series:
    decoded_elements = encoder.inverse_transform(encoded_elements)
    return pd.Series(decoded_elements.flatten(), name='element')


# 示例解码过程
def example_decode():
    # 加载编码器
    encoder_path = os.path.join(output_folder, 'encoder.pkl')
    encoder = pd.read_pickle(encoder_path)

    # 加载已编码的数据（示例）
    encoded_data_path = os.path.join(output_folder, 'example_encoded_data.csv')
    encoded_df = pd.read_csv(encoded_data_path)

    # 提取编码后的元素特征列
    encoded_elements = encoded_df.iloc[:, 6:-1]

    # 解码元素
    decoded_elements = decode_elements(encoded_elements, encoder)
    print(decoded_elements.head())


# 运行示例解码函数
# example_decode()

if __name__ == '__main__':

    base_csv_path = r'D:\学习\研三\晶体性能预测\qmof_database\qmof_database\qmof.csv'
    base_cif_directory = r'D:\学习\研三\晶体性能预测\qmof_database\qmof_database\relaxed_structures'
    base_output_xyz_directory = r'D:\学习\研三\晶体性能预测\converted_xyz'
    base_output_csv_directory = r'D:\学习\研三\晶体性能预测\cut_data_csv'
    processed_data_directory = r'D:\学习\研三\晶体性能预测\processed_data_csv'

    # 定义目标性能值：
    target_bandgap = 1.5
    process_samples(target_bandgap, base_csv_path, base_cif_directory, base_output_xyz_directory,
                    base_output_csv_directory)
    process_csv_files_2(r'D:\学习\研三\晶体性能预测\cut_data_csv',
                      r'D:\学习\研三\晶体性能预测\processed_data_csv')


    files = glob.glob(os.path.join(processed_data_directory, "*.csv"))  # 构建文件列表
    object_number = files.__len__()

    # case1 = True
    case1 = False
    case2 = True
    # case2 = False
    # case3 = True
    case3 = False
    '''
    情况1：生成大的混乱数据集，在xyz三轴随机输入n类DataFrame
    '''
    if case1:
        # 接下来，我们将使用处理后的 CSV 文件来生成数据集
        # 先定义字典，df{i}对应file，给所有file处理数据再赋label
        dfs = {}
        # for i in range(object_number):
        #     df_name = f"df{i}"
        #     filename = files[i]
        #     dfs[df_name] = pd.read_csv(files[i])
        for i in range(object_number):
            df_name = f"df{i}"
            filename = files[i]
            df = pd.read_csv(filename)
            df['idx'] = i
            dfs[df_name] = df
            print(df)

        # 打乱字典键的顺序，合并数据,堆叠坐标
        for k in range(3):  # range(object_number)
            keys = list(dfs.keys())
            random.shuffle(keys)
            all_data = pd.DataFrame(columns=['x', 'y', 'z'])
            # iteration = 1
            for l in range(3):  # Z轴
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
            np.save(f"D:\software\PythonProject\Python\point-transformer6\dataset\Area_2_conferenceRoom_{k+1}.npy", all_data)
            # 保存测试集npy文件
            # np.save(f"D:\software\PythonProject\Python\point-transformer6\dataset\Area_5_conferenceRoom_{k+1}.npy", all_data)

    '''
    情况2：生成一列的不同类型数据集，用作测试集(没打乱顺序)
    '''
    if case2:
        # 定义字典，用于存储每个 DataFrame
        dfs = {}

        for i in range(object_number):
            df_name = f"df{i}"
            filename = files[i]
            df = pd.read_csv(filename)
            df['idx'] = i  # 添加 'idx' 列，值为 i
            dfs[df_name] = df

        # 生成不同类型数据集，用作测试集
        for i in range(5):  # 假设有5种不同的数据类型
            all_data = pd.DataFrame(columns=['x', 'y', 'z'])
            iteration = 1
            for df_name in dfs:  # 使用字典 dfs 中的 DataFrames
                df = dfs[df_name]
                df['x'] = df['x'] + 10.5 * (iteration - 1)  # 调整 x 轴坐标
                all_data = all_data.append(df, ignore_index=True)  # 追加到 all_data
                iteration += 1

            # 打印信息
            print(f'------------------------------------第{i + 1}个数据类型--------------------------------------')
            print('all_data:', all_data)

            # 保存数据集
            np.save(f"D:\\software\\PythonProject\\Python\\point-transformer6\\dataset\\Area_5_conferenceRoom_{i}.npy",
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
            np.save(f"D:\software\PythonProject\Python\point-transformer6\dataset\Area_1_conferenceRoom_{iteration}.npy",
                    all_data)
            iteration += 1
