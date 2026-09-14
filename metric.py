import numpy as np
import time
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
from modules.model import PETCGDNN, MCLDNN, DAE, CLDNN2, SupConMCLDNN, feat_bottleneck, LinearClassifier, LSTMModel, GRUModel, MCLDNN_SVD, orthogonal_loss, ReverseLayerF
from data.dataset_sample_inc import splitRML2016A, splitDomainRML2016A, loadRML2016A, loadDomainRML2016A, loadDomainRML2016A_SVD
from utils.util import plot_confusion_matrix, FocalLoss, setup_seed
from sklearn.metrics import confusion_matrix
import os
from utils.scheduler import PolynomialLR
import tqdm
import matplotlib.pyplot as plt
from torch import optim
from modules.buffer import Buffer
import pandas as pd
import torch.nn.functional as F
import random
import re
from decimal import Decimal
import csv

def single_class(data: np.ndarray):
    # 四列数据
    res = []
    for i in range(data.shape[1]):
        max_i = np.max(data[:, i])
        min_i = np.min(data[:, i])
        cha = round(round((max_i - min_i) / 2, 3)+0.001, 2)
        print(f'{round(min_i+cha, 2):.2f} ± {cha:.2f}')
        # 返回列表
        res.append(f'{round(min_i+cha, 2):.2f} ± {cha:.2f}')
    return res

def data_shift(str, lennum=4):
    # 使用正则表达式提取数字和浮点数
    numbers = re.findall(r':\s*(-?\d+(\.\d+)?)', str)

    # 将字符串转换为浮点数
    numbers = [float(num[0]) for num in numbers]

    # 创建NumPy数组
    array = np.array(numbers)
    arr = np.zeros((1, lennum), dtype=float)
    for i in range(array.shape[0]):
        if i % lennum == 0:
            arr = np.concatenate([arr, array[i: i+lennum].reshape(1, -1)], axis=0)
    print(arr[1:])
    return arr[1:]

def read_csv(csv_path):
    data = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        # 创建 CSV 读取器
        csv_reader = csv.reader(csvfile)
        
        # 读取所有行
        for row in csv_reader:
            # 每行是一个字符串列表
            data.extend(row)
    return "\n".join(data)

def metric_csv(csv_path, lennum = 7):
    str1 = read_csv(csv_path)
#     str1 = '''
# '''
    str2 = '''
'''
    str3 = '''
'''
    a1 = data_shift(str1, lennum=lennum)
    a2 = data_shift(str2, lennum=lennum)
    a3 = data_shift(str3, lennum=lennum)
    data = np.vstack((a1, a2, a3))
    # print(data)
    # 列表文件
    res = single_class(data)
    return res
    

if __name__ == '__main__':
    lennum = 7
    root_path = r''
    metric_csv(root_path)
