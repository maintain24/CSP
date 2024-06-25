import pandas as pd
import os

# 定义文件路径
csv_file_path = r'D:\学习\研三\小论文第一篇（英文）\CES\审稿意见\补充实验结果\20240621（30epoch）\coordinate.csv'
csv_file_path_2 = r'D:\学习\研三\小论文第一篇（英文）\CES\审稿意见\补充实验结果\20240621（30epoch）\coordinate_2.csv'
excel_file_path = r'D:\学习\研一\元素相对原子质量.xlsx'
output_folder_path = r'D:\学习\研三\小论文第一篇（英文）\CES\审稿意见\补充实验结果\20240621（30epoch）\xyz'
output_folder_path_2 = r'D:\学习\研三\小论文第一篇（英文）\CES\审稿意见\补充实验结果\20240621（30epoch）\xyz_2'

# 读取Excel文件
element_df = pd.read_excel(excel_file_path, usecols=[2, 3], header=None)
# element_dict = dict(zip(element_df[3], element_df[2]))
element_df.columns = ['symbol', 'mass']
# 将质量列转换为浮点数
element_df['mass'] = element_df['mass'].astype(float)
element_dict = dict(zip(element_df['mass'], element_df['symbol']))
print(element_dict)

# 最近邻查找函数
def find_closest_element(mass):
    # 如果输入的质量是NaN，返回一个特定的值或者跳过
    if pd.isna(mass):
        return 'NaN'  # 或者其他适当的默认值或行为
    # 排除字典中的NaN键
    clean_dict = {k: v for k, v in element_dict.items() if not pd.isna(k)}
    # 使用clean_dict来找到最接近的质量
    closest_mass = min(clean_dict.keys(), key=lambda k: abs(k - mass))
    # 返回最接近质量对应的元素符号
    return clean_dict[closest_mass]


"""选择转化哪个文件"""
# coordinate = True
coordinate = False
coordinate_2 = True
# coordinate_2 = False


if coordinate:
    # 读取CSV文件
    df = pd.read_csv(csv_file_path, header=None)
    # 按第13列标签进行降序排序
    df_sorted = df.sort_values(by=12, ascending=False)

    # 创建输出文件夹
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 按第13列标签分组
    groups = df_sorted.groupby(12)

    # 生成.xyz文件
    for idx, group in groups:
        atom_count = len(group)
        performance_value = group.iloc[0, 10]  # 取该组第一个样本的性能值

        xyz_file_content = f"{atom_count}\n{performance_value}\n"

        for _, row in group.iterrows():
            x, y, z = row[0], row[1], row[2]
            element_code = row[9]
            # 找到最接近的相对原子质量值的元素符号
            element_symbol = find_closest_element(element_code)
            # element_symbol = element_dict.get(element_code, 'Unknown')
            xyz_file_content += f"{element_symbol} {x} {y} {z}\n"

        xyz_file_name = os.path.join(output_folder_path, f"{idx}.xyz")
        with open(xyz_file_name, 'w') as file:
            file.write(xyz_file_content)


if coordinate_2:
    # 读取CSV文件
    df = pd.read_csv(csv_file_path_2, header=None)
    # 按第17列标签进行降序排序
    df_sorted = df.sort_values(by=16, ascending=False)
    print('df_sorted第17列', df_sorted.iloc[:, 16])

    # 创建输出文件夹
    if not os.path.exists(output_folder_path_2):
        os.makedirs(output_folder_path_2)

    # 按第17列标签分组
    groups = df_sorted.groupby(16)

    # 生成.xyz文件
    for idx, group in groups:
        atom_count = len(group)
        performance_value = group.iloc[0, 18]  # 取该组第一个样本的性能值

        xyz_file_content = f"{atom_count}\n{performance_value}\n"

        for _, row in group.iterrows():
            x, y, z = row[12], row[13], row[14]
            element_code = row[17]
            # 找到最接近的相对原子质量值的元素符号
            element_symbol = find_closest_element(element_code)
            # element_symbol = element_dict.get(element_code, 'Unknown')
            xyz_file_content += f"{element_symbol} {x} {y} {z}\n"

        xyz_file_name = os.path.join(output_folder_path_2, f"{idx}.xyz")
        with open(xyz_file_name, 'w') as file:
            file.write(xyz_file_content)

