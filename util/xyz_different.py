# -*- coding:utf-8 -*-
import os
import numpy as np

def read_xyz(file_path: str) -> np.ndarray:
    """
    读取 .xyz 文件并返回原子坐标的 numpy 数组
    :param file_path: .xyz 文件路径
    :return: 原子坐标的 numpy 数组
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()[2:]  # 跳过前两行
        coordinates = []
        for line in lines:
            parts = line.split()
            coords = list(map(float, parts[1:4]))
            coordinates.append(coords)
        return np.array(coordinates)

def compare_structures(file1: str, file2: str) -> None:
    """
    比较两个 .xyz 文件中的原子坐标并打印不同之处
    :param file1: 第一个 .xyz 文件路径
    :param file2: 第二个 .xyz 文件路径
    """
    coords1 = read_xyz(file1)
    coords2 = read_xyz(file2)

    if coords1.shape != coords2.shape:
        print("两个结构的原子数量不同，无法比较。")
        return

    differences = np.abs(coords1 - coords2)
    threshold = 1e-3  # 设置一个阈值来忽略微小的差异

    for i, diff in enumerate(differences):
        if np.any(diff > threshold):
            print(f"原子 {i+1} 的坐标不同：")
            print(f"文件1: {coords1[i]}")
            print(f"文件2: {coords2[i]}")
        # else:
        #     print('mei you bu tong')

if __name__ == "__main__":
    output_folder_path = r'D:\学习\研三\小论文第一篇（英文）\CES\审稿意见\补充实验结果\20240621（30epoch）\xyz'
    output_folder_path_2 = r'D:\学习\研三\小论文第一篇（英文）\CES\审稿意见\补充实验结果\20240621（30epoch）\xyz_2'

    file1 = os.path.join(output_folder_path, '6.0.xyz')
    file2 = os.path.join(output_folder_path_2, '6.xyz')

    compare_structures(file1, file2)
