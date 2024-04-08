import os
import numpy as np
import torch
import torch.nn as nn


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        if train:
            self.path = self.data_dir + "train/"
        else:
            self.path = self.data_dir + "test/"

        lst_data = os.listdir(self.path + "img/")

        file_list = [f.replace(".npy", "") for f in lst_data] 
        file_list.sort()

        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        inputs = np.load(self.path + "img/" + self.file_list[index] + ".npy")
        label = np.load(self.path + "label/" + self.file_list[index] + ".npy")

        # normalize
        inputs = inputs / 255.0
        label = label / 255.0
        inputs = inputs.astype(np.float32)
        label = label[:, :, 0].astype(np.float32)

        # 인풋 데이터 차원이 2이면, 채널 축을 추가해줘야한다. 파이토치 인풋은 (batch, 채널, 행, 열)
        if inputs.ndim == 2:
            inputs = inputs[:, :, np.newaxis]
        if label.ndim == 2:
            label = label[:, :, np.newaxis]

        data = {'input': inputs, 'label': label}

        if self.transform:
            data = self.transform(data)  # transform에 할당된 class 들이 호출되면서 __call__ 함수 실행
            
        data['file_name'] = self.path + "img/" + self.file_list[index] + ".npy"

        return data

# Argumentation - Tensor Transformation 구현
class ToTensor(object):
    def __call__(self, data):
        input, label = data['input'], data['label']

        input = input.transpose((2, 0, 1)).astype(np.float32)
        label = label.transpose((2, 0, 1)).astype(np.float32)

        data = {'input': torch.from_numpy(input), 'label': torch.from_numpy(label)}

        return data

# Argumentation - Normalization 구현
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input, label = data['input'], data['label']

        #Input에 대해서만
        input = (input - self.mean) / self.std

        data = {'input': input, 'label': label}

        return data

# Argumentation - RandomFlip 구현
class RandomFlip(object):
    def __call__(self, data):
        input, label = data['input'], data['label']

        if np.random.rand() > 0.5:
            input = np.fliplr(input)
            label = np.fliplr(label)

        if np.random.rand() > 0.5:
            input = np.flipud(input)
            label = np.flipud(label)

        data = {'input': input, 'label': label}

        return data