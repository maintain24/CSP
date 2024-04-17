# -*- coding:utf-8 -*-
import csv

# 打开文件并读取数据
with open(r"D:/学习/研二/小论文point cloud transformer/代码运行日志/PT2/result/20230412.2result.txt") as f:
    lines = f.readlines()

a, b, c = [], [], []
for line in lines:
    num1, num2, num3 = list(map(float, line.split(" ")))
    a.append(num1)
    b.append(num2)
    c.append(num3)

# 将数据保存到csv文件中
with open(r"D:/学习/研二/小论文point cloud transformer/代码运行日志/PT2/result/20230412.2result.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['a', 'b', 'c'])
    for i in range(len(a)):
        writer.writerow([a[i], b[i], c[i]])

