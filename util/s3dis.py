import os

import numpy as np
import time
import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util import sa_create
from util.data_util import data_prepare

'''
class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if split == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]
        valid_data_list = []
        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(data_root, item + '.npy')
                # data = np.load(data_path, allow_pickle=True)  # xyzrgbl, N*7 ,增加allow_pickle=True可以读取npy文件中的pkl数据

                # 检查文件名是否以点开头
                if os.path.basename(data_path).startswith('.'):
                    print(f"Deleting hidden file: {data_path}")
                    try:
                        os.remove(data_path)  # 删除文件
                    except OSError as e:
                        print(f"Error deleting file {data_path}: {e}")
                    continue  # 跳过该文件

                try:
                    data = np.load(data_path, allow_pickle=True)
                    valid_data_list.append(item)
                    # 处理数据
                except OSError as e:
                    print(f"Error loading file {data_path}: {e}")

                sa_create("shm://{}".format(item), data)
        self.data_list = valid_data_list  # 更新有效的数据列表
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        # time.sleep(5)  # 使得多进程读取数据不冲突，上一个读取完保存才到下一个
        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        coord, feat, label = data[:, 0:3], data[:, 3:7], data[:, 8]
        coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop
'''

class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if split == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]
        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(data_root, item + '.npy')
                data = np.load(data_path)  # xyzrgbl, N*7 ,增加allow_pickle=True可以读取npy文件中的pkl数据
                sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        # time.sleep(5)  # 使得多进程读取数据不冲突，上一个读取完保存才到下一个
        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        # if data.shape[1] != 8:
        #     print(f"Data at index {data_idx} has incorrect shape: {data.shape}")
        # 打印数据维度以检查是否正确
        # print(f"Data at index {data_idx} has shape: {data.shape}")
        coord, feat, label = data[:, 0:3], data[:, 3:8], data[:, 8].astype(np.float64)
        coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        # 打印原始标签以检查是否正确读取
        # print(f"Original label at index {data_idx}: {label}")
        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop
