# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

'''
画图epoch和loss的折线图
'''
f = open(r"D:/学习/研二/小论文point cloud transformer/代码运行日志/PT2/result/20230116.8result.txt")
line = f.readline()
a = []
b = []
c = []
d = []

while line:
    num1, num2, num3 = list(map(float, line.split(" ")))
    a.append(num1)
    b.append(num2)
    c.append(num3)
    line = f.readline()
f.close()


num_each_epoch = [x for x in a if x == 1]
print('每个epoch的个数为：', len(num_each_epoch))  # 97

for i in range(round(max(a))):  # 0-14,i=0
    iter = 1
    for j in a:
        if j == i+1:  # a[0]=1
            num4 = (j-1)+iter/len(num_each_epoch)
            iter = iter + 1
            d.append(num4)
        else: pass

print(d)
print(len(a))
print(len(d))

a = np.array(d)
b = np.array(b)
c = np.array(c)

plt.figure(figsize=(10,5),dpi=450)
plt.rcParams['figure.figsize']=(5.5,5)
# 画epoch与Acc
plt.subplot(2, 1, 1)
plt.plot(a, c, 'y-')
plt.tight_layout
my_x_ticks = np.arange(0, 21, 1)  # 原始数据有13个点，故此处为设置从0开始，间隔为1
plt.xticks(my_x_ticks)
plt.title('Crystal Structure Formation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# 画epoch与Loss
plt.subplot(2, 1, 2)
plt.plot(a, b, 'b-')
my_x_ticks = np.arange(0, 21, 1)  # 原始数据有13个点，故此处为设置从0开始，间隔为1
plt.xticks(my_x_ticks)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

