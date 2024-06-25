# -*- coding:utf-8 -*-
"""
找中心点，将晶体结构从中心点出发三维变一维进行编码
"""
import math
import pandas as pd
import csv
import codecs

# d = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# f = [(1, 2, 1), (4, 6, 2), (5, 7, 3), (2, 8, 4), (21, 4, 5), (23, 4, 6), (3, 23, 7), (2, 56, 8), (32, 6, 9), (2, 3, 10)]
# print(f)

# def delete_first_lines(self, filename, count):   # 删除文件前一行
#     fin = open(filename, 'r')
#     a = fin.readlines()
#     fout = open(filename, 'w')
#     b = ''.join(a[count:])
#     fout.write(b)

# df = pd.read_csv(r'D:\software\PythonProject\Python\centroid lstm\1000000.csv')  # 例如‘C:\Users\Administrator\Desktop\data.csv'  （其中r表示不转义）


def find_centroid(file, output):
    # df = pd.read_csv(file)
    df = file.drop(index=0)  # 删除第一行[1001238,Picture,1,NaN]
    # f = df.drop(['Elements'][0], axis=1)
    f = df.drop('Elements', axis=1).astype('float32', errors='ignore')
    f = f.values
    # Y = df[['Elements'][0]]
    Y = df['Elements'].astype('float32', errors='ignore')
    Y = Y.values
    # print('f:', f)
    # print('Y:', Y)
    list_length = len(Y)

    for i in range(list_length):
        for j in range(list_length):
            d = 0
            d = d + math.sqrt((f[i][0]-f[j][0])**2+(f[i][1]-f[j][1])**2+(f[i][2]-f[j][2])**2)
    s = d
    m = 0
    # distance = []
    # for i in range(10):
        # print("点(",f[i][0],",",f[i][1],", ",f[i][2],")","到各点的距离和是:", d[i])
    #     distance.append(d[i])
    # print(distance)
    # distance.sort(reverse=False)
    # print(distance)
    if d < s:
        s = d
        m = i
    # print("中心点是", "(", f[m][0], ",", f[m][1], ",", f[m][2], ")")

    distance = []
    for k in range(list_length):
        D = math.sqrt((f[m][0]-f[k][0])**2+(f[m][1]-f[k][1])**2+(f[m][2]-f[k][2])**2)
        distance.append(D)
        distance.sort(reverse=False)
    # print(distance)

    with codecs.open(output, 'w+', 'utf-8') as csvfile:  # './test.csv'
        # 指定 csv 文件的头部显示项
        filednames = ['Element', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=filednames)
        writer.writeheader()
        for i in range(0, list_length):
            try:
                writer.writerow({'Element':Y[i], 'Value':distance[i]})
            except UnicodeEncodeError:
                print("编码错误, 该数据无法写到文件中, 直接忽略该数据")
    return file, output


def element_coding(output, test, elements):
    f = open(output, 'w+', encoding='utf-8', newline='')  # 最好写上newline=‘’
    csv_writer = csv.writer(f)
    csv_writer.writerow(['id', 'Elements', 'Value'])

    with open(test, 'r', encoding='utf8') as fp1:
        # 使用列表推导式，将读取到的数据装进列表
        list1 = [i for i in csv.reader(fp1)]  # csv.reader 读取到的数据是list类型
        # print('list1:', list1)
    with open(elements, 'r', encoding='utf8') as fp2:
        list2 = [i for i in csv.reader(fp2)]
        # print('list2:', list2)
    for i in list1:
        for j in list2:
            if i[0] == j[0]:  # 匹配到ID，就将内容写入到data.csv中
                csv_writer.writerow([list1.index(i), j[1], i[1]])  #   [i[0],j[1]]。也可以加上‘,j[2]’
                # print('i[0], j[0], j[1]:', i[0], j[0], j[1])

    f.close()
    return f


# files = r'D:\software\PythonProject\Python\centroid lstm\1000000.csv'
# find_centroid(files, 'test.csv')
# element_coding('data.csv', 'test.csv', 'elements_all.csv')


