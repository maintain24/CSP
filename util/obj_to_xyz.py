# -*- coding:utf-8 -*-
'''
将obj文件转换成xyz格式文件，然后删除第一列的顶点‘v’替换成占位符‘X’，删除后面的rgb值，在第一行添加原子个数，第二行添加文件名
'''

import os

input_file_path = r'D:\学习\研一\点云处理后数据集\obj\修改坐标带颜色的obj\002.obj'
output_file_path = r'D:\学习\研一\点云处理后数据集\obj\修改坐标带颜色的obj\002.xyz'
output_file_name = os.path.basename(output_file_path)

with open(input_file_path, 'r') as f:
    lines = f.readlines()

num_points = 0
output_lines = []

for line in lines:
    if line.startswith('v'):
        num_points += 1
        coords = line.split()[1:4]
        output_lines.append('X ' + ' '.join(coords) + '\n')

output_lines.insert(0, str(num_points) + '\n' + output_file_name + '\n')

with open(output_file_path, 'w') as f:
    f.writelines(output_lines)

