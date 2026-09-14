import random
import time 
import sys
sys.path.append('../XincL_local')
sys.path.append('../XinCL')
# # from utils.util import setup_seed
from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
import pickle
import h5py
import glob
from scipy.io import loadmat

def time_shift(signal, shift=10):
    return np.roll(signal, shift, axis=1)

def frequency_shift(signal, shift=10):
    signal_fft = np.fft.fft(signal, axis=1)
    shifted_fft = np.roll(signal_fft, shift, axis=1)
    shifted_signal = np.fft.ifft(shifted_fft, axis=1)
    return np.real(shifted_signal)

def add_gaussian_noise(signal, noise_level=0.05):
    """
    添加高斯白噪声
    :param noise_level: 噪声强度的标准差 (可根据你的信号幅值进行动态调整)
    """
    # 生成与信号同维度的高斯分布噪声
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

def amplitude_scaling(signal, scale=0.5):
    """
    幅度随机缩放
    :param scale: 缩放因子，0.5 表示幅度减半
    """
    return signal * scale

def time_masking(signal, mask_ratio=0.1):
    """
    时间遮蔽，随机将一段连续的信号置零
    :param mask_ratio: 遮蔽长度占总序列长度的比例
    """
    masked_signal = signal.copy()
    seq_len = signal.shape[1]  # 假设 axis=1 是时间维度
    mask_len = int(seq_len * mask_ratio)
    
    # 随机选择遮蔽的起始点
    start = np.random.randint(0, seq_len - mask_len)
    
    # 将该区间的数据清零
    masked_signal[:, start:start+mask_len] = 0
    return masked_signal



def phase_rotation(signal_i, signal_q, angle=180):
    """
    相位旋转 (需要同时传入 I 路和 Q 路数据)
    :param angle: 旋转角度
    """
    # 随机生成旋转角度并转为弧度
    angle_rad = np.deg2rad(angle)
    
    # 旋转矩阵计算: I' = I*cos(a) - Q*sin(a), Q' = I*sin(a) + Q*cos(a)
    i_rotated = signal_i * np.cos(angle_rad) - signal_q * np.sin(angle_rad)
    q_rotated = signal_i * np.sin(angle_rad) + signal_q * np.cos(angle_rad)
    
    return i_rotated, q_rotated


def apply_augmentation(x, t_shift, f_shift, noise, scale, mask, phase):

    # time shift
    x = time_shift(x, shift=t_shift)

    # frequency shift
    x = frequency_shift(x, shift=f_shift)

    # amplitude scaling
    x = amplitude_scaling(x, scale=scale)

    # noise
    if noise > 0:
        x = add_gaussian_noise(x, noise_level=noise)

    # time masking
    if mask > 0:
        x = time_masking(x, mask_ratio=mask)

    # phase rotation（关键修复）
    if phase > 0:
        I = x[:, 0, :]   # 或 x[...,0]
        Q = x[:, 1, :]
        I, Q = phase_rotation(I, Q, angle=phase)
        x = np.stack([I, Q], axis=1)

    return x

# 随即划分保存数据
def splitRML2016A(path, save_path, snr_split=False, index=0):
    Xd = pickle.load(open(path, 'rb'), encoding='latin')
    np.random.seed(2016)  
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X_train = []
    lbl_train = []
    X_inc_train = []
    lbl_inc_train = []
    X_test = []
    lbl_test = []
    X_inc_test = []
    lbl_inc_test = []

    # SNR:[-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    test_SNR = []
    test_SNR.append(snrs[0])
    test_SNR.append(snrs[-1])
    if index < 0:
        index = len(snrs) + index
    
    if snr_split == True:
        snr = snrs[index]
        for mod in mods:
            # 对每个信噪比的数据均匀切分
            temp = Xd[(mod, snr)]
            label = (mod, snr)
            # 对预处理好的数据进行打包，并按2:2:1:2划分数据
            n_examples = temp.shape[0]
            n_train = n_examples * 0.5
            n_test = n_examples * 0.2
            n_inc_train = n_examples * 0.1
            n_inc_test = n_examples * 0.2
            train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
            inc_train_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)),
                                             size=int(n_inc_train), replace=False)
            inc_test_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx) - set(inc_train_idx)),
                                            size=int(n_inc_test), replace=False)
            test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(inc_train_idx) - set(inc_test_idx))

            X_train.append(temp[train_idx])
            X_test.append(temp[test_idx])
            X_inc_train.append(temp[inc_train_idx])
            X_inc_test.append(temp[inc_test_idx])

            for i in range(int(n_train)):
                lbl_train.append(label)
            for i in range(int(n_test)):
                lbl_test.append(label)
            for i in range(int(n_inc_train)):
                lbl_inc_train.append(label)
            for i in range(int(n_inc_test)):
                lbl_inc_test.append(label)

    else:
        for mod in mods:
            for snr in snrs:
                # 对每个模式下每个信噪比的数据均匀切分
                temp = Xd[(mod, snr)]
                label = (mod, snr)
                # 对预处理好的数据进行打包，并按2:2:1:2划分数据
                n_examples = temp.shape[0]
                n_train = n_examples * 0.5
                n_test = n_examples * 0.2
                n_inc_train = n_examples * 0.1
                n_inc_test = n_examples * 0.2
                train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
                inc_train_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)),
                                                 size=int(n_inc_train), replace=False)
                inc_test_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx) - set(inc_train_idx)),
                                                size=int(n_inc_test), replace=False)
                test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(inc_train_idx) - set(inc_test_idx))

                X_train.append(temp[train_idx])
                X_test.append(temp[test_idx])
                X_inc_train.append(temp[inc_train_idx])
                X_inc_test.append(temp[inc_test_idx])

                for i in range(int(n_train)):
                    lbl_train.append(label)
                for i in range(int(n_test)):
                    lbl_test.append(label)
                for i in range(int(n_inc_train)):
                    lbl_inc_train.append(label)
                for i in range(int(n_inc_test)):
                    lbl_inc_test.append(label)
                
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    X_inc_train = np.vstack(X_inc_train)
    X_inc_test = np.vstack(X_inc_test)

    # 得到训练数据和测试数据的类型标签
    Y_train = np.array(list(map(lambda x: mods.index(lbl_train[x][0]), np.arange(0, len(lbl_train)))))
    Y_test = np.array(list(map(lambda x: mods.index(lbl_test[x][0]), np.arange(0, len(lbl_test)))))
    Y_inc_train = np.array(list(map(lambda x: mods.index(lbl_inc_train[x][0]), np.arange(0, len(lbl_inc_train)))))
    Y_inc_test = np.array(list(map(lambda x: mods.index(lbl_inc_test[x][0]), np.arange(0, len(lbl_inc_test)))))
    
    # 得到训练数据和测试数据的信噪比标签
    snr_train = np.array(list(map(lambda x: snrs.index(lbl_train[x][1]), np.arange(0, len(lbl_train)))))
    snr_test = np.array(list(map(lambda x: snrs.index(lbl_test[x][1]), np.arange(0, len(lbl_test)))))
    snr_inc_train = np.array(list(map(lambda x: snrs.index(lbl_inc_train[x][1]), np.arange(0, len(lbl_inc_train)))))
    snr_inc_test = np.array(list(map(lambda x: snrs.index(lbl_inc_test[x][1]), np.arange(0, len(lbl_inc_test)))))

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, 'X_train' if snr_split == False else f'snr_{str(snrs[index])}_X_train'), X_train)
    np.save(os.path.join(save_path, 'Y_train' if snr_split == False else f'snr_{str(snrs[index])}_Y_train'), Y_train)
    np.save(os.path.join(save_path, 'snr_train' if snr_split == False else f'snr_{str(snrs[index])}_snr_train'), snr_train)
    np.save(os.path.join(save_path, 'X_test' if snr_split == False else f'snr_{str(snrs[index])}_X_test'), X_test)
    np.save(os.path.join(save_path, 'Y_test' if snr_split == False else f'snr_{str(snrs[index])}_Y_test'), Y_test)
    np.save(os.path.join(save_path, 'snr_test' if snr_split == False else f'snr_{str(snrs[index])}_snr_test'), snr_test)
    np.save(os.path.join(save_path, 'X_inc_train' if snr_split == False else f'snr_{str(snrs[index])}_X_inc_train'), X_inc_train)
    np.save(os.path.join(save_path, 'Y_inc_train' if snr_split == False else f'snr_{str(snrs[index])}_Y_inc_train'), Y_inc_train)
    np.save(os.path.join(save_path, 'snr_inc_train' if snr_split == False else f'snr_{str(snrs[index])}_snr_inc_train'), snr_inc_train)
    np.save(os.path.join(save_path, 'X_inc_test' if snr_split == False else f'snr_{str(snrs[index])}_X_inc_test'), X_inc_test)
    np.save(os.path.join(save_path, 'Y_inc_test' if snr_split == False else f'snr_{str(snrs[index])}_Y_inc_test'), Y_inc_test)
    np.save(os.path.join(save_path, 'snr_inc_test' if snr_split == False else f'snr_{str(snrs[index])}_snr_inc_test'), snr_inc_test)

# 加载随机分割保存的分割数据文件
def loadRML2016A(path, snr_split=False, snrs=None, index=0):
    if index < 0:
        index = len(snrs) + index
    train_dataset = Getdata_RML2016A(
        np.load(os.path.join(path, 'X_train.npy' if snr_split == False else f'snr_{str(snrs[index])}_X_train.npy')),
        np.load(os.path.join(path, 'Y_train.npy' if snr_split == False else f'snr_{str(snrs[index])}_Y_train.npy')),
        np.load(os.path.join(path, 'snr_train.npy' if snr_split == False else f'snr_{str(snrs[index])}_snr_train.npy')))
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    # val_dataset = Getdata_RML2016A(
    #     np.load(os.path.join(path, 'X_val.npy' if snr_split == False else f'snr_{str(snrs[index])}_X_val.npy')),
    #     np.load(os.path.join(path, 'Y_val.npy' if snr_split == False else f'snr_{str(snrs[index])}_Y_val.npy')),
    #     np.load(os.path.join(path, 'snr_val.npy' if snr_split == False else f'snr_{str(snrs[index])}_snr_val.npy')))
    # val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)
    test_dataset = Getdata_RML2016A(
        np.load(os.path.join(path, 'X_test.npy' if snr_split == False else f'snr_{str(snrs[index])}_X_test.npy')),
        np.load(os.path.join(path, 'Y_test.npy' if snr_split == False else f'snr_{str(snrs[index])}_Y_test.npy')),
        np.load(os.path.join(path, 'snr_test.npy' if snr_split == False else f'snr_{str(snrs[index])}_snr_test.npy')))
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)

    inc_train_dataset = Getdata_RML2016A(
        np.load(os.path.join(path, 'X_inc_train.npy' if snr_split == False else f'snr_{str(snrs[index])}_X_inc_train.npy')),
        np.load(os.path.join(path, 'Y_inc_train.npy' if snr_split == False else f'snr_{str(snrs[index])}_Y_inc_train.npy')),
        np.load(os.path.join(path, 'snr_inc_train.npy' if snr_split == False else f'snr_{str(snrs[index])}_snr_inc_train.npy')))
    inc_train_loader = DataLoader(inc_train_dataset, batch_size=512, shuffle=True)
    # inc_val_dataset = Getdata_RML2016A(
    #     np.load(os.path.join(path, 'X_inc_val.npy' if snr_split == False else f'snr_{str(snrs[index])}_X_inc_val.npy')),
    #     np.load(os.path.join(path, 'Y_inc_val.npy' if snr_split == False else f'snr_{str(snrs[index])}_Y_inc_val.npy')),
    #     np.load(os.path.join(path, 'snr_inc_val.npy' if snr_split == False else f'snr_{str(snrs[index])}_snr_inc_val.npy')))
    # inc_val_loader = DataLoader(inc_val_dataset, batch_size=512, shuffle=True)
    inc_test_dataset = Getdata_RML2016A(
        np.load(os.path.join(path, 'X_inc_test.npy' if snr_split == False else f'snr_{str(snrs[index])}_X_inc_test.npy')),
        np.load(os.path.join(path, 'Y_inc_test.npy' if snr_split == False else f'snr_{str(snrs[index])}_Y_inc_test.npy')),
        np.load(os.path.join(path, 'snr_inc_test.npy' if snr_split == False else f'snr_{str(snrs[index])}_snr_inc_test.npy')))
    inc_test_loader = DataLoader(inc_test_dataset, batch_size=512, shuffle=True)
    return train_dataset, test_dataset, inc_train_dataset, inc_test_dataset, train_loader, test_loader, inc_train_loader, inc_test_loader

# 按域来划分保存数据
# ori_ratio: 原始数据集所占总数据比例，也即所占信噪比比例
# ori_snr: 原始数据集是高信噪比'high'还是低信噪比'low'
def splitDomainRML2016A(path, save_path, domain_num=5, ori_snr='high'):
    Xd = pickle.load(open(path, 'rb'), encoding='latin')
    np.random.seed(3407)
    v, p = 8, 3
    snr_list, mod_list = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    if ori_snr == 'high':
        snr_list.reverse()
    print(snr_list)
    # ori_start = random.randint(0, len(snr_list)) // p * p
    # while ori_start + v > len(snr_list):
    #     ori_start = (ori_start // p - 1) * p
    # print(ori_start)
    # time.sleep(5)
    # ori_end = ori_start + v
    # snr_per_doamin = (len(snr_list)-v) / (domain_num-1)

    ori_start = 0
    ori_end = ori_start + v


    # print(snr_list[ori_start], snr_list[ori_end-1])
    # print(snr_list[inc_start], snr_list[inc_end-1])
    #
    # print(snr_list, type(snr_list))
    X_train, lbl_train= [], []
    X_val, lbl_val= [], []
    X_test, lbl_test = [], []
    # 原始数据
    for i in range(ori_start, ori_end):
        snr = snr_list[i]
        for mod in mod_list:
            # 对每个信噪比的数据均匀切分
            temp = Xd[(mod, snr)]
            label = (mod, snr)
            # 对预处理好的数据进行打包，并按6:2:2划分数据
            n_examples = temp.shape[0]
            n_train = n_examples * 0.6
            n_val = n_examples * 0.2
            train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
            val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val), replace=False)
            test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

            X_train.append(temp[train_idx])
            X_val.append(temp[val_idx])
            X_test.append(temp[test_idx])

            for i in range(int(n_train)):
                lbl_train.append(label)
            for i in range(int(n_train), int(n_examples - n_val)):
                lbl_val.append(label)
            for i in range(int(n_train + n_val), int(n_examples)):
                lbl_test.append(label)

    X_train = np.vstack(X_train)
    X_val = np.vstack(X_val)
    X_test = np.vstack(X_test)

    snr_train = np.array(list(map(lambda x: snr_list.index(lbl_train[x][1]), np.arange(0, len(lbl_train)))))
    snr_val = np.array(list(map(lambda x: snr_list.index(lbl_val[x][1]), np.arange(0, len(lbl_val)))))
    snr_test = np.array(list(map(lambda x: snr_list.index(lbl_test[x][1]), np.arange(0, len(lbl_test)))))
    # 得到训练数据和测试数据的类型标签
    Y_train = np.array(list(map(lambda x: mod_list.index(lbl_train[x][0]), np.arange(0, len(lbl_train)))))
    Y_val = np.array(list(map(lambda x: mod_list.index(lbl_val[x][0]), np.arange(0, len(lbl_val)))))
    Y_test = np.array(list(map(lambda x: mod_list.index(lbl_test[x][0]), np.arange(0, len(lbl_test)))))
    # save_path = os.path.join(save_path, 'First8_domain_num_' + str(domain_num)+'_2')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, 'X_train'), X_train)
    np.save(os.path.join(save_path, 'Y_train'), Y_train)
    np.save(os.path.join(save_path, 'snr_train'), snr_train)
    np.save(os.path.join(save_path, 'X_val'), X_val)
    np.save(os.path.join(save_path, 'Y_val'), Y_val)
    np.save(os.path.join(save_path, 'snr_val'), snr_val)
    np.save(os.path.join(save_path, 'X_test'), X_test)
    np.save(os.path.join(save_path, 'Y_test'), Y_test)
    np.save(os.path.join(save_path, 'snr_test'), snr_test)

    del snr_list[ori_start:ori_end]
    time_slist = [0, 5]
    fre_slist = [0, 5]
    # 增量数据
    for k in range(1, domain_num):
        # inc_start = random.randint(0, len(snr_list)) // p * p
        inc_start = 0
        inc_end = inc_start + p
        print(f'第{k}个增量域内包含信噪比:', f'{snr_list[inc_start:inc_end]}')
        time.sleep(5)
        kkk = 0
        for tt in time_slist:
            for ff in fre_slist:
                X_inc_train, lbl_inc_train = [], []
                X_inc_val, lbl_inc_val = [], []
                X_inc_test, lbl_inc_test = [], []
                for i in range(inc_start, inc_end):
                    snr = snr_list[i]
                    for mod in mod_list:
                        # 对每个信噪比的数据均匀切分
                        temp = Xd[(mod, snr)]
                        label = (mod, snr)
                        # 对预处理好的数据进行打包，并按6:2:2划分数据
                        n_examples = temp.shape[0]
                        n_train = n_examples * 0.6
                        n_val = n_examples * 0.2
                        train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
                        val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val), replace=False)
                        test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

                        X_inc_train.append(time_shift(frequency_shift(temp[train_idx], ff), tt))
                        X_inc_val.append(time_shift(frequency_shift(temp[val_idx], ff), tt))
                        X_inc_test.append(time_shift(frequency_shift(temp[test_idx], ff), tt))

                        for i in range(int(n_train)):
                            lbl_inc_train.append(label)
                        for i in range(int(n_train), int(n_examples - n_val)):
                            lbl_inc_val.append(label)
                        for i in range(int(n_train + n_val), int(n_examples)):
                            lbl_inc_test.append(label)
                X_inc_train = np.vstack(X_inc_train)
                X_inc_val = np.vstack(X_inc_val)
                X_inc_test = np.vstack(X_inc_test)
                Y_inc_train = np.array(list(map(lambda x: mod_list.index(lbl_inc_train[x][0]), np.arange(0, len(lbl_inc_train)))))
                Y_inc_val = np.array(list(map(lambda x: mod_list.index(lbl_inc_val[x][0]), np.arange(0, len(lbl_inc_val)))))
                Y_inc_test = np.array(list(map(lambda x: mod_list.index(lbl_inc_test[x][0]), np.arange(0, len(lbl_inc_test)))))
                # 得到训练数据和测试数据的信噪比标签
                snr_inc_train = np.array(list(map(lambda x: snr_list.index(lbl_inc_train[x][1]), np.arange(0, len(lbl_inc_train)))))
                snr_inc_val = np.array(list(map(lambda x: snr_list.index(lbl_inc_val[x][1]), np.arange(0, len(lbl_inc_val)))))
                snr_inc_test = np.array(list(map(lambda x: snr_list.index(lbl_inc_test[x][1]), np.arange(0, len(lbl_inc_test)))))

                np.save(os.path.join(save_path, 'X_inc_train_'+str(k+kkk*12)), X_inc_train)
                np.save(os.path.join(save_path, 'Y_inc_train_'+str(k+kkk*12)), Y_inc_train)
                np.save(os.path.join(save_path, 'snr_inc_train_'+str(k+kkk*12)), snr_inc_train)
                np.save(os.path.join(save_path, 'X_inc_val_'+str(k+kkk*12)), X_inc_val)
                np.save(os.path.join(save_path, 'Y_inc_val_'+str(k+kkk*12)), Y_inc_val)
                np.save(os.path.join(save_path, 'snr_inc_val_'+str(k+kkk*12)), snr_inc_val)
                np.save(os.path.join(save_path, 'X_inc_test_'+str(k+kkk*12)), X_inc_test)
                np.save(os.path.join(save_path, 'Y_inc_test_'+str(k+kkk*12)), Y_inc_test)
                np.save(os.path.join(save_path, 'snr_inc_test_'+str(k+kkk*12)), snr_inc_test)
                kkk += 1
        del snr_list[inc_start:inc_end]

# 7种增强策略
def splitDomainRML2016A_7_methods(path, save_path, domain_num=5, ori_snr='high'):
    Xd = pickle.load(open(path, 'rb'), encoding='latin')
    np.random.seed(3407)
    v, p = 8, 3
    snr_list, mod_list = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    if ori_snr == 'high':
        snr_list.reverse()

    ori_start = 0
    ori_end = ori_start + v

    X_train, lbl_train= [], []
    X_val, lbl_val= [], []
    X_test, lbl_test = [], []
    # 增强策略6种
    time_slist = [0, 10]
    fre_slist = [0, 10]
    guas_noise_list = [0, 0.1]
    amp_scale_list = [1, 0.5]
    time_mask_list = [0, 0.1]
    phase_rotation_list = [0, 90]
    # 原始数据
    for i in range(ori_start, ori_end):
        snr = snr_list[i]
        for mod in mod_list:
            # 对每个信噪比的数据均匀切分
            temp = Xd[(mod, snr)]
            label = (mod, snr)
            # 对预处理好的数据进行打包，并按6:2:2划分数据
            n_examples = temp.shape[0]
            n_train = n_examples * 0.6
            n_val = n_examples * 0.2
            train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
            val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val), replace=False)
            test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

            X_train.append(temp[train_idx])
            X_val.append(temp[val_idx])
            X_test.append(temp[test_idx])

            for i in range(int(n_train)):
                lbl_train.append(label)
            for i in range(int(n_train), int(n_examples - n_val)):
                lbl_val.append(label)
            for i in range(int(n_train + n_val), int(n_examples)):
                lbl_test.append(label)

    X_train = np.vstack(X_train)
    X_val = np.vstack(X_val)
    X_test = np.vstack(X_test)
    
    
    snr_train = np.array(list(map(lambda x: snr_list.index(lbl_train[x][1]), np.arange(0, len(lbl_train)))))
    snr_val = np.array(list(map(lambda x: snr_list.index(lbl_val[x][1]), np.arange(0, len(lbl_val)))))
    snr_test = np.array(list(map(lambda x: snr_list.index(lbl_test[x][1]), np.arange(0, len(lbl_test)))))
    # 得到训练数据和测试数据的类型标签
    Y_train = np.array(list(map(lambda x: mod_list.index(lbl_train[x][0]), np.arange(0, len(lbl_train)))))
    Y_val = np.array(list(map(lambda x: mod_list.index(lbl_val[x][0]), np.arange(0, len(lbl_val)))))
    Y_test = np.array(list(map(lambda x: mod_list.index(lbl_test[x][0]), np.arange(0, len(lbl_test)))))
    # save_path = os.path.join(save_path, 'First8_domain_num_' + str(domain_num)+'_2')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, 'X_train'), X_train)
    np.save(os.path.join(save_path, 'Y_train'), Y_train)
    np.save(os.path.join(save_path, 'snr_train'), snr_train)
    np.save(os.path.join(save_path, 'X_val'), X_val)
    np.save(os.path.join(save_path, 'Y_val'), Y_val)
    np.save(os.path.join(save_path, 'snr_val'), snr_val)
    np.save(os.path.join(save_path, 'X_test'), X_test)
    np.save(os.path.join(save_path, 'Y_test'), Y_test)
    np.save(os.path.join(save_path, 'snr_test'), snr_test)

    del snr_list[ori_start:ori_end]
    
    zq_list = [[0, 10, 0.1, 1.0, 0.1, 90],\
                [10, 10, 0.0, 1.0, 0.0, 90],\
                [0, 10, 0.0, 0.5, 0.1, 90],\
                [0, 10, 0.1, 1.0, 0.0, 0],\
                [10, 10, 0.0, 0.5, 0.1, 0],\
                [10, 0, 0.1, 1.0, 0.0, 0],\
                [0, 10, 0.1, 1.0, 0.0, 90],\
                [0, 10, 0.1, 0.5, 0.1, 90],\
                [10, 0, 0.1, 1.0, 0.1, 0],\
                [10, 10, 0.0, 1.0, 0.1, 90],\
                [10, 10, 0.1, 0.5, 0.0, 0],\
                [0, 10, 0.0, 0.5, 0.1, 0]]
    # 增量数据
    inc_nums = 1
    for k in range(1, domain_num):
        # inc_start = random.randint(0, len(snr_list)) // p * p
        inc_start = 0
        inc_end = inc_start + p
        print(f'第{k}个增量域内包含信噪比:', f'{snr_list[inc_start:inc_end]}')
        time.sleep(5)
        for kkk in range(len(zq_list)):
            time_values, freq_values, noise_values, amp_values, time_mask_values, phase_values = zq_list[kkk]
            X_inc_train, lbl_inc_train = [], []
            X_inc_val, lbl_inc_val = [], []
            X_inc_test, lbl_inc_test = [], []
            for i in range(inc_start, inc_end):
                snr = snr_list[i]
                for mod in mod_list:
                    # 对每个信噪比的数据均匀切分
                    temp = Xd[(mod, snr)]
                    label = (mod, snr)
                    # 对预处理好的数据进行打包，并按6:2:2划分数据
                    n_examples = temp.shape[0]
                    n_train = n_examples * 0.6
                    n_val = n_examples * 0.2
                    train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
                    val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val), replace=False)
                    test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))
                    # 进行数据增强
                    X_inc_train.append(apply_augmentation(temp[train_idx], time_values, freq_values, noise_values, amp_values, time_mask_values, phase_values))
                    X_inc_val.append(apply_augmentation(temp[val_idx], time_values, freq_values, noise_values, amp_values, time_mask_values, phase_values))
                    X_inc_test.append(apply_augmentation(temp[test_idx], time_values, freq_values, noise_values, amp_values, time_mask_values, phase_values))

                    

                    for i in range(int(n_train)):
                        lbl_inc_train.append(label)
                    for i in range(int(n_train), int(n_examples - n_val)):
                        lbl_inc_val.append(label)
                    for i in range(int(n_train + n_val), int(n_examples)):
                        lbl_inc_test.append(label)
            X_inc_train = np.vstack(X_inc_train)
            X_inc_val = np.vstack(X_inc_val)
            X_inc_test = np.vstack(X_inc_test)
            Y_inc_train = np.array(list(map(lambda x: mod_list.index(lbl_inc_train[x][0]), np.arange(0, len(lbl_inc_train)))))
            Y_inc_val = np.array(list(map(lambda x: mod_list.index(lbl_inc_val[x][0]), np.arange(0, len(lbl_inc_val)))))
            Y_inc_test = np.array(list(map(lambda x: mod_list.index(lbl_inc_test[x][0]), np.arange(0, len(lbl_inc_test)))))
            # 得到训练数据和测试数据的信噪比标签
            snr_inc_train = np.array(list(map(lambda x: snr_list.index(lbl_inc_train[x][1]), np.arange(0, len(lbl_inc_train)))))
            snr_inc_val = np.array(list(map(lambda x: snr_list.index(lbl_inc_val[x][1]), np.arange(0, len(lbl_inc_val)))))
            snr_inc_test = np.array(list(map(lambda x: snr_list.index(lbl_inc_test[x][1]), np.arange(0, len(lbl_inc_test)))))

            np.save(os.path.join(save_path, 'X_inc_train_'+str(inc_nums)), X_inc_train)
            np.save(os.path.join(save_path, 'Y_inc_train_'+str(inc_nums)), Y_inc_train)
            np.save(os.path.join(save_path, 'snr_inc_train_'+str(inc_nums)), snr_inc_train)
            np.save(os.path.join(save_path, 'X_inc_val_'+str(inc_nums)), X_inc_val)
            np.save(os.path.join(save_path, 'Y_inc_val_'+str(inc_nums)), Y_inc_val)
            np.save(os.path.join(save_path, 'snr_inc_val_'+str(inc_nums)), snr_inc_val)
            np.save(os.path.join(save_path, 'X_inc_test_'+str(inc_nums)), X_inc_test)
            np.save(os.path.join(save_path, 'Y_inc_test_'+str(inc_nums)), Y_inc_test)
            np.save(os.path.join(save_path, 'snr_inc_test_'+str(inc_nums)), snr_inc_test)
            inc_nums += 1
            print(f'第{k}个增量域的第{kkk}种方法已保存，形状为 {X_inc_train.shape}')
        del snr_list[inc_start:inc_end]

# 域有重叠的SNR
def splitDomainRML2016A_cd(path, save_path, domain_num=5, ori_snr='high'):
    Xd = pickle.load(open(path, 'rb'), encoding='latin')
    np.random.seed(3407)
    v, p = 8, 6
    snr_list, mod_list = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    if ori_snr == 'high':
        snr_list.reverse()
    ori_start = 0
    ori_end = ori_start + v
    del snr_list[ori_start:(ori_end-3)]
    # 不同域
    time_slist = [0, 10]
    fre_slist = [0, 10]
    guas_noise = [0, 0.1]
    amp_scale = [0, 0.8]
    
    
    # 增量数据
    for k in range(1, domain_num):
        inc_start = 0
        inc_end = inc_start + p
        print(f'第{k}个增量域内包含信噪比:', f'{snr_list[inc_start:inc_end]}')
        time.sleep(5)
        for tt in time_slist:
            for ff in fre_slist:
                X_inc_train, lbl_inc_train = [], []
                X_inc_val, lbl_inc_val = [], []
                X_inc_test, lbl_inc_test = [], []
                for i in range(inc_start, inc_end):
                    snr = snr_list[i]
                    for mod in mod_list:
                        # 对每个信噪比的数据均匀切分
                        temp = Xd[(mod, snr)]
                        label = (mod, snr)
                        # 对预处理好的数据进行打包，并按6:2:2划分数据
                        n_examples = temp.shape[0]
                        n_train = n_examples * 0.6
                        n_val = n_examples * 0.2
                        train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
                        val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val), replace=False)
                        test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

                        X_inc_train.append(time_shift(frequency_shift(temp[train_idx], ff), tt))
                        X_inc_val.append(time_shift(frequency_shift(temp[val_idx], ff), tt))
                        X_inc_test.append(time_shift(frequency_shift(temp[test_idx], ff), tt))

                        for i in range(int(n_train)):
                            lbl_inc_train.append(label)
                        for i in range(int(n_train), int(n_examples - n_val)):
                            lbl_inc_val.append(label)
                        for i in range(int(n_train + n_val), int(n_examples)):
                            lbl_inc_test.append(label)
                X_inc_train = np.vstack(X_inc_train)
                X_inc_val = np.vstack(X_inc_val)
                X_inc_test = np.vstack(X_inc_test)
                Y_inc_train = np.array(list(map(lambda x: mod_list.index(lbl_inc_train[x][0]), np.arange(0, len(lbl_inc_train)))))
                Y_inc_val = np.array(list(map(lambda x: mod_list.index(lbl_inc_val[x][0]), np.arange(0, len(lbl_inc_val)))))
                Y_inc_test = np.array(list(map(lambda x: mod_list.index(lbl_inc_test[x][0]), np.arange(0, len(lbl_inc_test)))))
                # 得到训练数据和测试数据的信噪比标签
                snr_inc_train = np.array(list(map(lambda x: snr_list.index(lbl_inc_train[x][1]), np.arange(0, len(lbl_inc_train)))))
                snr_inc_val = np.array(list(map(lambda x: snr_list.index(lbl_inc_val[x][1]), np.arange(0, len(lbl_inc_val)))))
                snr_inc_test = np.array(list(map(lambda x: snr_list.index(lbl_inc_test[x][1]), np.arange(0, len(lbl_inc_test)))))

                np.save(os.path.join(save_path, 'X_inc_train_'+str(k)), X_inc_train)
                np.save(os.path.join(save_path, 'Y_inc_train_'+str(k)), Y_inc_train)
                np.save(os.path.join(save_path, 'snr_inc_train_'+str(k)), snr_inc_train)
                np.save(os.path.join(save_path, 'X_inc_val_'+str(k)), X_inc_val)
                np.save(os.path.join(save_path, 'Y_inc_val_'+str(k)), Y_inc_val)
                np.save(os.path.join(save_path, 'snr_inc_val_'+str(k)), snr_inc_val)
                np.save(os.path.join(save_path, 'X_inc_test_'+str(k)), X_inc_test)
                np.save(os.path.join(save_path, 'Y_inc_test_'+str(k)), Y_inc_test)
                np.save(os.path.join(save_path, 'snr_inc_test_'+str(k)), snr_inc_test)
        del snr_list[inc_start:(inc_end-3)]


def splitDomainRML2018(path, save_path, domain_num=2, ori_snr='high'):
    f = h5py.File(path, 'r')
    np.random.seed(3407)
    X = f['X'][:, :, :]  # ndarray(2555904*1024*2),shape
    Y = f['Y'][:, :]  # ndarray((24*26*4096) * 24),class, 26:SNR, 24:class  one-hot类型，所以为*24
    Z = f['Z'][:]  # ndarray(2555904*1),SNR

    # 将ONE-HOT改成数字
    Y = np.argmax(Y, axis=1)
    Z = Z.reshape(-1)
    # 总共26个信噪比：-20~30，分为五个域，10+4+4+4+4
    snr_list = [i for i in range(-20, 32, 2)]
    print(snr_list)
    v, p = 10, 4
    if ori_snr == 'high':
        snr_list.reverse()
    
    ori_start = 0
    ori_end = ori_start + v

    
    X_train, Y_train, Z_train = [], [], []
    X_val, Y_val, Z_val = [], [], []
    X_test, Y_test, Z_test = [], [], []

    print(snr_list[ori_start: ori_end])
    time.sleep(5)
    # 原始数据
    for i in range(ori_start, ori_end):
        snr = snr_list[i]
        ind1 = np.where((Z == snr))[0]
        mod_list = [i for i in range(0, 24)]
        for mod in mod_list:
            # 对每个信噪比的数据均匀切分
            ind2 = np.where((Y == mod))[0]
            ind = np.intersect1d(ind1, ind2)
            Xt = X[ind]
            Yt = Y[ind]
            Zt = Z[ind]
            # 对预处理好的数据进行打包，并按6:2:2划分数据
            n_examples = Xt.shape[0]
            n_train = int(n_examples * 0.6)
            n_val = int(n_examples * 0.2)
            train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
            val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val), replace=False)
            test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

            X_train.append(Xt[train_idx])
            X_val.append(Xt[val_idx])
            X_test.append(Xt[test_idx])

            Y_train.append(Yt[train_idx])
            Y_val.append(Yt[val_idx])
            Y_test.append(Yt[test_idx])

            Z_train.append(Zt[train_idx])
            Z_val.append(Zt[val_idx])
            Z_test.append(Zt[test_idx])
            print(f'{snr}S-{mod}C: {len(train_idx)}  {len(val_idx)}  {len(test_idx)}')
    X_train = np.vstack(X_train)
    X_val = np.vstack(X_val)
    X_test = np.vstack(X_test)

    Y_train = np.hstack(Y_train)
    Y_val = np.hstack(Y_val)
    Y_test = np.hstack(Y_test)

    Z_train = np.hstack(Z_train)
    Z_val = np.hstack(Z_val)
    Z_test = np.hstack(Z_test)
    print(Y_train.shape, Z_train.shape)
    # save_path = os.path.join(save_path, 'Avg_domain_num_' + str(domain_num))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, 'X_train'), X_train)
    np.save(os.path.join(save_path, 'Y_train'), Y_train)
    np.save(os.path.join(save_path, 'snr_train'), Z_train)
    np.save(os.path.join(save_path, 'X_val'), X_val)
    np.save(os.path.join(save_path, 'Y_val'), Y_val)
    np.save(os.path.join(save_path, 'snr_val'), Z_val)
    np.save(os.path.join(save_path, 'X_test'), X_test)
    np.save(os.path.join(save_path, 'Y_test'), Y_test)
    np.save(os.path.join(save_path, 'snr_test'), Z_test)

    del snr_list[ori_start:ori_end]
    # 增量数据
    for k in range(1, domain_num):
        # inc_start = random.randint(0, len(snr_list)) // p * p
        inc_start = 0
        inc_end = inc_start + p
        print(f'第{k}个增量域内包含信噪比:', f'{snr_list[inc_start:inc_end]}')
        time.sleep(2)
        X_inc_train, Y_inc_train, Z_inc_train = [], [], []
        X_inc_val, Y_inc_val, Z_inc_val = [], [], []
        X_inc_test, Y_inc_test, Z_inc_test = [], [], []
        for i in range(inc_start, inc_end):
            snr = snr_list[i]
            mod_list = [i for i in range(0, 24)]
            ind1 = np.where(Z == snr)[0]
            for mod in mod_list:
                # 对每个信噪比的数据均匀切分
                ind2 = np.where((Y == mod))[0]
                ind = np.intersect1d(ind1, ind2)
                Xt = X[ind]
                Yt = Y[ind]
                Zt = Z[ind]
                # 对预处理好的数据进行打包，并按6:2:2划分数据
                n_examples = Xt.shape[0]
                n_train = int(n_examples * 0.6)
                n_val = int(n_examples * 0.2)
                train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
                val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val),
                                           replace=False)
                test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

                X_inc_train.append(Xt[train_idx])
                X_inc_val.append(Xt[val_idx])
                X_inc_test.append(Xt[test_idx])

                Y_inc_train.append(Yt[train_idx])
                Y_inc_val.append(Yt[val_idx])
                Y_inc_test.append(Yt[test_idx])

                Z_inc_train.append(Zt[train_idx])
                Z_inc_val.append(Zt[val_idx])
                Z_inc_test.append(Zt[test_idx])
                print(f'{snr}S-{mod}C: {len(train_idx)}  {len(val_idx)}  {len(test_idx)}')
        X_inc_train = np.vstack(X_inc_train)
        X_inc_val = np.vstack(X_inc_val)
        X_inc_test = np.vstack(X_inc_test)

        Y_inc_train = np.hstack(Y_inc_train)
        Y_inc_val = np.hstack(Y_inc_val)
        Y_inc_test = np.hstack(Y_inc_test)

        Z_inc_train = np.hstack(Z_inc_train)
        Z_inc_val = np.hstack(Z_inc_val)
        Z_inc_test = np.hstack(Z_inc_test)
        print(Y_inc_train.shape, Z_inc_train.shape)
        np.save(os.path.join(save_path, 'X_inc_train_'+str(k)), X_inc_train)
        np.save(os.path.join(save_path, 'Y_inc_train_'+str(k)), Y_inc_train)
        np.save(os.path.join(save_path, 'snr_inc_train_'+str(k)), Z_inc_train)
        np.save(os.path.join(save_path, 'X_inc_val_'+str(k)), X_inc_val)
        np.save(os.path.join(save_path, 'Y_inc_val_'+str(k)), Y_inc_val)
        np.save(os.path.join(save_path, 'snr_inc_val_'+str(k)), Z_inc_val)
        np.save(os.path.join(save_path, 'X_inc_test_'+str(k)), X_inc_test)
        np.save(os.path.join(save_path, 'Y_inc_test_'+str(k)), Y_inc_test)
        np.save(os.path.join(save_path, 'snr_inc_test_'+str(k)), Z_inc_test)

        del snr_list[inc_start:inc_end]

def load_SCF_SVD(path, domain_num=5, train_bz=512, index_domain=None, len_one=8193):
    train_path = path + '/Train/'
    test_path = path + '/Test/'
    X_train, Y_train, SNR_train, X_test, Y_test, SNR_test \
        = [], [], [], [], [], []
    domain_1_X_train, domain_1_Y_train, domain_1_SNR_train, domain_1_X_test, domain_1_Y_test, domain_1_SNR_test \
        = [], [], [], [], [], []
    domain_2_X_train, domain_2_Y_train, domain_2_SNR_train, domain_2_X_test, domain_2_Y_test, domain_2_SNR_test \
        = [], [], [], [], [], []
    domain_3_X_train, domain_3_Y_train, domain_3_SNR_train, domain_3_X_test, domain_3_Y_test, domain_3_SNR_test \
        = [], [], [], [], [], []
    domain_4_X_train, domain_4_Y_train, domain_4_SNR_train, domain_4_X_test, domain_4_Y_test, domain_4_SNR_test \
        = [], [], [], [], [], []
    # 检索所有训练文件信息
    for root, dirs, files in os.walk(train_path):
        for file in files:
            # 检查文件扩展名是否为.mat
            if file.endswith('.mat'):
                SNR = -1
                dB_index = file.find('_dB_')
                if dB_index != 1:
                    # 找到前一个'_'的位置
                    underscore_index = file.rfind('_', 0, dB_index)
                    if underscore_index != -1:
                        # 提取数字
                        number = file[underscore_index + 1:dB_index]
                        SNR = int(number)
                # 将文件的完整路径添加到列表中
                data = loadmat(train_path + file)
                X = data['train_data']
                X = X.reshape(X.shape[0])
                # 取出中间16行拿来训练和测试
                start_index = (X[0].shape[0] - len_one) // 2
                end_index = start_index + len_one
                X = [X[i][np.newaxis, start_index:end_index] for i in range(len(X))]
                Y = data['train_class'].reshape(-1)
                SNR_tr = np.ones(Y.shape) * SNR
                if 9 <= SNR <= 15:
                    X_train.extend(X)
                    Y_train.extend(Y)
                    SNR_train.extend(SNR_tr)
                elif 7 <= SNR <= 9:
                    domain_1_X_train.extend(X)
                    domain_1_Y_train.extend(Y)
                    domain_1_SNR_train.extend(SNR_tr)
                elif 5 <= SNR <= 7:
                    domain_2_X_train.extend(X)
                    domain_2_Y_train.extend(Y)
                    domain_2_SNR_train.extend(SNR_tr)
                elif 3 <= SNR <= 5:
                    domain_3_X_train.extend(X)
                    domain_3_Y_train.extend(Y)
                    domain_3_SNR_train.extend(SNR_tr)
                else:
                    domain_4_X_train.extend(X)
                    domain_4_Y_train.extend(Y)
                    domain_4_SNR_train.extend(SNR_tr)
                print(len(X), X[0].shape)
    # 检索所有测试文件信息
    for root, dirs, files in os.walk(test_path):
        for file in files:
            # 检查文件扩展名是否为.mat
            if file.endswith('.mat'):
                SNR = -1
                dB_index = file.find('_dB_')
                if dB_index != 1:
                    # 找到前一个'_'的位置
                    underscore_index = file.rfind('_', 0, dB_index)
                    if underscore_index != -1:
                        # 提取数字
                        number = file[underscore_index + 1:dB_index]
                        SNR = int(number)
                # 将文件的完整路径添加到列表中
                data = loadmat(test_path + file)
                X = data['test_data']
                X = X.reshape(X.shape[0])
                # 取出中间16行拿来训练和测试
                start_index = (X[0].shape[0] - len_one) // 2
                end_index = start_index + len_one
                X = [X[i][np.newaxis, start_index:end_index] for i in range(len(X))]
                Y = data['test_class'].reshape(-1)
                SNR_tr = np.ones(Y.shape) * SNR
                if 9 <= SNR <= 15:
                    X_test.extend(X)
                    Y_test.extend(Y)
                    SNR_test.extend(SNR_tr)
                elif 7 <= SNR <= 9:
                    domain_1_X_test.extend(X)
                    domain_1_Y_test.extend(Y)
                    domain_1_SNR_test.extend(SNR_tr)
                elif 5 <= SNR <= 7:
                    domain_2_X_test.extend(X)
                    domain_2_Y_test.extend(Y)
                    domain_2_SNR_test.extend(SNR_tr)
                elif 3 <= SNR <= 5:
                    domain_3_X_test.extend(X)
                    domain_3_Y_test.extend(Y)
                    domain_3_SNR_test.extend(SNR_tr)
                else:
                    domain_4_X_test.extend(X)
                    domain_4_Y_test.extend(Y)
                    domain_4_SNR_test.extend(SNR_tr)
    # 转为numpy
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    SNR_train = np.array(SNR_train)
    SNR_test = np.array(SNR_test)
    train_dataset = Getdata_SCF_SVD(X_train, Y_train, SNR_train, 0)
    domain_1_X_train = np.vstack(domain_1_X_train)
    domain_1_X_test = np.vstack(domain_1_X_test)
    domain_1_Y_train = np.array(domain_1_Y_train)
    domain_1_Y_test = np.array(domain_1_Y_test)
    domain_1_SNR_train = np.array(domain_1_SNR_train)
    domain_1_SNR_test = np.array(domain_1_SNR_test)

    domain_2_X_train = np.vstack(domain_2_X_train)
    domain_2_X_test = np.vstack(domain_2_X_test)
    domain_2_Y_train = np.array(domain_2_Y_train)
    domain_2_Y_test = np.array(domain_2_Y_test)
    domain_2_SNR_train = np.array(domain_2_SNR_train)
    domain_2_SNR_test = np.array(domain_2_SNR_test)

    domain_3_X_train = np.vstack(domain_3_X_train)
    domain_3_X_test = np.vstack(domain_3_X_test)
    domain_3_Y_train = np.array(domain_3_Y_train)
    domain_3_Y_test = np.array(domain_3_Y_test)
    domain_3_SNR_train = np.array(domain_3_SNR_train)
    domain_3_SNR_test = np.array(domain_3_SNR_test)

    domain_4_X_train = np.vstack(domain_4_X_train)
    domain_4_X_test = np.vstack(domain_4_X_test)
    domain_4_Y_train = np.array(domain_4_Y_train)
    domain_4_Y_test = np.array(domain_4_Y_test)
    domain_4_SNR_train = np.array(domain_4_SNR_train)
    domain_4_SNR_test = np.array(domain_4_SNR_test)


    domain_1_train_dataset = Getdata_SCF_SVD(domain_1_X_train, domain_1_Y_train, domain_1_SNR_train, 1)
    domain_2_train_dataset = Getdata_SCF_SVD(domain_2_X_train, domain_2_Y_train, domain_2_SNR_train, 2)
    domain_3_train_dataset = Getdata_SCF_SVD(domain_3_X_train, domain_3_Y_train, domain_3_SNR_train, 3)
    domain_4_train_dataset = Getdata_SCF_SVD(domain_4_X_train, domain_4_Y_train, domain_4_SNR_train, 4)

    train_loader = DataLoader(train_dataset, train_bz, shuffle=True)
    domain_1_train_dataloader = DataLoader(domain_1_train_dataset, train_bz, shuffle=True)
    domain_2_train_dataloader = DataLoader(domain_2_train_dataset, train_bz, shuffle=True)
    domain_3_train_dataloader = DataLoader(domain_3_train_dataset, train_bz, shuffle=True)
    domain_4_train_dataloader = DataLoader(domain_4_train_dataset, train_bz, shuffle=True)

    test_dataset = Getdata_SCF_SVD(X_test, Y_test, SNR_test, 0)
    domain_1_test_dataset = Getdata_SCF_SVD(domain_1_X_test, domain_1_Y_test, domain_1_SNR_test, 1)
    domain_2_test_dataset = Getdata_SCF_SVD(domain_2_X_test, domain_2_Y_test, domain_2_SNR_test, 2)
    domain_3_test_dataset = Getdata_SCF_SVD(domain_3_X_test, domain_3_Y_test, domain_3_SNR_test, 3)
    domain_4_test_dataset = Getdata_SCF_SVD(domain_4_X_test, domain_4_Y_test, domain_4_SNR_test, 4)

    test_loader = DataLoader(test_dataset, train_bz, shuffle=False)
    domain_1_test_dataloader = DataLoader(domain_1_test_dataset, train_bz, shuffle=False)
    domain_2_test_dataloader = DataLoader(domain_2_test_dataset, train_bz, shuffle=False)
    domain_3_test_dataloader = DataLoader(domain_3_test_dataset, train_bz, shuffle=False)
    domain_4_test_dataloader = DataLoader(domain_4_test_dataset, train_bz, shuffle=False)

    inc_train_datasets = [domain_1_train_dataset, domain_2_train_dataset, domain_3_train_dataset, domain_4_train_dataset]
    inc_test_datasets = [domain_1_test_dataset, domain_2_test_dataset, domain_3_test_dataset, domain_4_test_dataset]
    inc_train_loaders = [domain_1_train_dataloader, domain_2_train_dataloader, domain_3_train_dataloader, domain_4_train_dataloader]
    inc_test_loaders = [domain_1_test_dataloader, domain_2_test_dataloader, domain_3_test_dataloader, domain_4_test_dataloader]
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, SNR_train.shape, SNR_test.shape)
    print(domain_1_X_train.shape, domain_1_X_test.shape, domain_1_Y_train.shape, domain_1_Y_test.shape, domain_1_SNR_train.shape, domain_1_SNR_test.shape)
    print(domain_2_X_train.shape, domain_2_X_test.shape, domain_2_Y_train.shape, domain_2_Y_test.shape, domain_2_SNR_train.shape, domain_2_SNR_test.shape)
    print(domain_3_X_train.shape, domain_3_X_test.shape, domain_3_Y_train.shape, domain_3_Y_test.shape, domain_3_SNR_train.shape, domain_3_SNR_test.shape)
    print(domain_4_X_train.shape, domain_4_X_test.shape, domain_4_Y_train.shape, domain_4_Y_test.shape, domain_4_SNR_train.shape, domain_4_SNR_test.shape)

    if index_domain is not None:
        random_domain = [[0, 1, 2, 3], [0, 3, 1, 2], [3, 0, 2, 1], [3, 2, 1, 0]]
        random_domain = random_domain[index_domain]
    else:
        random_domain = [0, 3, 1, 2, ]
    # random_domain.reverse()
    # random.shuffle(random_domain)
    print(random_domain)
    inc_train_datasets = [inc_train_datasets[i] for i in random_domain]
    inc_train_loaders = [inc_train_loaders[i] for i in random_domain]
    inc_test_datasets = [inc_test_datasets[i] for i in random_domain]
    inc_test_loaders = [inc_test_loaders[i] for i in random_domain]


    return random_domain, train_dataset, test_dataset, inc_train_datasets, inc_test_datasets, train_loader, test_loader, inc_train_loaders, inc_test_loaders

def load_SCF_SVD_test(path, domain_num=5, train_bz=512, index_domain=None, len_one=8193):
    X_train, Y_train= [], []
    # 检索所有训练文件信息
    # 将文件的完整路径添加到列表中
    data = loadmat(path)
    X = data['train_data']
    X = X.reshape(X.shape[0])
    # 取出中间16行拿来训练和测试
    start_index = (X[0].shape[0] - len_one) // 2
    end_index = start_index + len_one
    X = [X[i][np.newaxis, start_index:end_index] for i in range(len(X))]
    Y = data['train_class'].reshape(-1)
    X_train.extend(X)
    Y_train.extend(Y)
    print(len(X), X[0].shape)
    # 检索所有测试文件信息

    # 转为numpy
    X_train = np.vstack(X_train)

    Y_train = np.array(Y_train)
    SNR_train = np.ones(Y_train.shape)
    train_dataset = Getdata_SCF_SVD(X_train, Y_train, SNR_train, 0)





    train_loader = DataLoader(train_dataset, train_bz, shuffle=True)





    return train_dataset, train_loader

# 加载按域划分的数据
def loadDomainRML2016A(path, domain_num=2):
    np.random.seed(3407)
    train_dataset = Getdata_RML2016A(
        np.load(os.path.join(path, 'X_train.npy')),
        np.load(os.path.join(path, 'Y_train.npy')),
        np.load(os.path.join(path, 'snr_train.npy')))
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_dataset = Getdata_RML2016A(
        np.load(os.path.join(path, 'X_val.npy')),
        np.load(os.path.join(path, 'Y_val.npy')),
        np.load(os.path.join(path, 'snr_val.npy')))
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)
    test_dataset = Getdata_RML2016A(
        np.load(os.path.join(path, 'X_test.npy')),
        np.load(os.path.join(path, 'Y_test.npy')),
        np.load(os.path.join(path, 'snr_test.npy')))
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)
    inc_train_datasets, inc_train_loaders, inc_val_datasets, inc_val_loaders, inc_test_datasets, inc_test_loaders = [], [], [], [], [], []
    for k in range(1, domain_num):
        inc_train_dataset = Getdata_RML2016A(
            np.load(os.path.join(path, 'X_inc_train.npy')),
            np.load(os.path.join(path, 'Y_inc_train.npy')),
            np.load(os.path.join(path, 'snr_inc_train.npy')))
        inc_train_loader = DataLoader(inc_train_dataset, batch_size=256, shuffle=True)
        inc_val_dataset = Getdata_RML2016A(
            np.load(os.path.join(path, 'X_inc_val.npy')),
            np.load(os.path.join(path, 'Y_inc_val.npy')),
            np.load(os.path.join(path, 'snr_inc_val.npy')))
        inc_val_loader = DataLoader(inc_val_dataset, batch_size=512, shuffle=True)
        inc_test_dataset = Getdata_RML2016A(
            np.load(os.path.join(path, 'X_inc_test.npy')),
            np.load(os.path.join(path, 'Y_inc_test.npy')),
            np.load(os.path.join(path, 'snr_inc_test.npy')))
        inc_test_loader = DataLoader(inc_test_dataset, batch_size=512, shuffle=True)
        inc_train_datasets.append(inc_train_dataset)
        inc_val_datasets.append(inc_val_dataset)
        inc_test_datasets.append(inc_test_dataset)
        inc_train_loaders.append(inc_train_loader)
        inc_val_loaders.append(inc_val_loader)
        inc_test_loaders.append(inc_test_loader)
    return train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders


class Getdata_RML2016A(Dataset):
    def __init__(self, data, label, snrs, transform=None):
        super().__init__()
        self.X = data
        self.lbl = label
        self.snrs = snrs
        self.transform = transform
        print("shape of all data:", self.X.shape)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index])
        x = x.unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        y = self.lbl[index]
        snr = self.snrs[index]
        return x, y, snr, index

    def __len__(self):
        return (self.X.shape[0])

# SVD
def loadDomainRML2016A_SVD(path, domain_num=5, train_bz=512, index_domain=None):
    # setup_seed(3000)
    train_dataset = Getdata_RML2016A_SVD(
        np.load(os.path.join(path, 'X_train.npy')),
        np.load(os.path.join(path, 'Y_train.npy')),
        np.load(os.path.join(path, 'snr_train.npy')), 0)
    train_loader = DataLoader(train_dataset, batch_size=train_bz, shuffle=True)
    val_dataset = Getdata_RML2016A_SVD(
        np.load(os.path.join(path, 'X_val.npy')),
        np.load(os.path.join(path, 'Y_val.npy')),
        np.load(os.path.join(path, 'snr_val.npy')), 0)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
    test_dataset = Getdata_RML2016A_SVD(
        np.load(os.path.join(path, 'X_test.npy')),
        np.load(os.path.join(path, 'Y_test.npy')),
        np.load(os.path.join(path, 'snr_test.npy')), 0)
    test_loader = DataLoader(test_dataset, batch_size=train_bz, shuffle=False)

    inc_train_datasets, inc_train_loaders, inc_val_datasets, inc_val_loaders, inc_test_datasets, inc_test_loaders = [], [], [], [], [], []
    if index_domain is not None:
        if domain_num == 5:
            random_domain = [[1, 2, 3, 4], [1, 4, 2, 3], [4, 1, 3, 2], [4, 3, 2, 1]]
        else:
            random_domain = [[i for i in range(1, domain_num)]]
            random.seed(3407)
            for j in range(1, 5):
                random_domain.append(random.sample(range(1, domain_num), k=domain_num-1))
            print(random_domain)
        random_domain = random_domain[index_domain]
    else:
        random_domain = [1, 4, 2, 3, ]
    # random_domain.reverse()
    # random.shuffle(random_domain)
    print(random_domain)
    for k in range(1, domain_num):
        s = random_domain[k-1]
        inc_train_dataset = Getdata_RML2016A_SVD(
            np.load(os.path.join(path, 'X_inc_train_'+str(s)+'.npy')),
            np.load(os.path.join(path, 'Y_inc_train_'+str(s)+'.npy')),
            np.load(os.path.join(path, 'snr_inc_train_'+str(s)+'.npy')), k)
        inc_train_loader = DataLoader(inc_train_dataset, batch_size=train_bz, shuffle=True)
        inc_val_dataset = Getdata_RML2016A_SVD(
            np.load(os.path.join(path, 'X_inc_val_'+str(s)+'.npy')),
            np.load(os.path.join(path, 'Y_inc_val_'+str(s)+'.npy')),
            np.load(os.path.join(path, 'snr_inc_val_'+str(s)+'.npy')), k)
        inc_val_loader = DataLoader(inc_val_dataset, batch_size=1000, shuffle=False)
        inc_test_dataset = Getdata_RML2016A_SVD(
            np.load(os.path.join(path, 'X_inc_test_'+str(s)+'.npy')),
            np.load(os.path.join(path, 'Y_inc_test_'+str(s)+'.npy')),
            np.load(os.path.join(path, 'snr_inc_test_'+str(s)+'.npy')), k)
        inc_test_loader = DataLoader(inc_test_dataset, batch_size=train_bz, shuffle=False)
        inc_train_datasets.append(inc_train_dataset)
        inc_val_datasets.append(inc_val_dataset)
        inc_test_datasets.append(inc_test_dataset)
        inc_train_loaders.append(inc_train_loader)
        inc_val_loaders.append(inc_val_loader)
        inc_test_loaders.append(inc_test_loader)
    return random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders
class Getdata_RML2016A_SVD(Dataset):
    def __init__(self, data, label, snrs, domain_class, transform=None):
        super().__init__()
        self.X = data
        self.lbl = label
        self.snrs = snrs
        self.transform = transform
        self.domain_class = domain_class
        print("shape of all data:", self.X.shape)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index])
        x = x.unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        y = self.lbl[index]
        snr = self.snrs[index]
        dlabel = max(0, self.domain_class)
        return x, y, snr, dlabel, index

    def __len__(self):
        return (self.X.shape[0])

class Getdata_SCF_SVD(Dataset):
    def __init__(self, data, label, snrs, domain_class, transform=None):
        super().__init__()
        self.X = data
        self.lbl = label
        self.snrs = snrs
        self.transform = transform
        self.domain_class = domain_class
        print("shape of all data:", self.X.shape)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index])
        x = x.unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        y = self.lbl[index]
        snr = self.snrs[index]
        dlabel = max(0, self.domain_class)
        return x, y, snr, dlabel, index

    def __len__(self):
        return (self.X.shape[0])

def loadDomainRML2018_SVD(path, domain_num=1, train_bz=256, index_domain=None):
    train_dataset = Getdata_RML2018_SVD(
        np.load(os.path.join(path, 'X_train.npy')),
        np.load(os.path.join(path, 'Y_train.npy')),
        np.load(os.path.join(path, 'snr_train.npy')), 0)
    train_loader = DataLoader(train_dataset, batch_size=train_bz, shuffle=True)
    val_dataset = Getdata_RML2018_SVD(
        np.load(os.path.join(path, 'X_val.npy')),
        np.load(os.path.join(path, 'Y_val.npy')),
        np.load(os.path.join(path, 'snr_val.npy')), 0)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_dataset = Getdata_RML2018_SVD(
        np.load(os.path.join(path, 'X_test.npy')),
        np.load(os.path.join(path, 'Y_test.npy')),
        np.load(os.path.join(path, 'snr_test.npy')), 0)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    inc_train_datasets, inc_train_loaders, inc_val_datasets, inc_val_loaders, inc_test_datasets, inc_test_loaders = [], [], [], [], [], []
    if index_domain is not None:
        random_domain = [[1, 2, 3, 4], [1, 4, 2, 3], [4, 1, 3, 2], [4, 3, 2, 1]]
        random_domain = random_domain[index_domain]
    else:
        random_domain = [1, 4, 2, 3, ]
    # random_domain.reverse()
    # random.shuffle(random_domain)
    print(random_domain)
    for k in range(1, domain_num):
        s = random_domain[k-1]
        inc_train_dataset = Getdata_RML2018_SVD(
            np.load(os.path.join(path, 'X_inc_train_'+str(s)+'.npy')),
            np.load(os.path.join(path, 'Y_inc_train_'+str(s)+'.npy')),
            np.load(os.path.join(path, 'snr_inc_train_'+str(s)+'.npy')), k)
        inc_train_loader = DataLoader(inc_train_dataset, batch_size=train_bz, shuffle=True)
        inc_val_dataset = Getdata_RML2018_SVD(
            np.load(os.path.join(path, 'X_inc_val_'+str(s)+'.npy')),
            np.load(os.path.join(path, 'Y_inc_val_'+str(s)+'.npy')),
            np.load(os.path.join(path, 'snr_inc_val_'+str(s)+'.npy')), k)
        inc_val_loader = DataLoader(inc_val_dataset, batch_size=512, shuffle=False)
        inc_test_dataset = Getdata_RML2018_SVD(
            np.load(os.path.join(path, 'X_inc_test_'+str(s)+'.npy')),
            np.load(os.path.join(path, 'Y_inc_test_'+str(s)+'.npy')),
            np.load(os.path.join(path, 'snr_inc_test_'+str(s)+'.npy')), k)
        inc_test_loader = DataLoader(inc_test_dataset, batch_size=512, shuffle=False)
        inc_train_datasets.append(inc_train_dataset)
        inc_val_datasets.append(inc_val_dataset)
        inc_test_datasets.append(inc_test_dataset)
        inc_train_loaders.append(inc_train_loader)
        inc_val_loaders.append(inc_val_loader)
        inc_test_loaders.append(inc_test_loader)
    return random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders

class Getdata_RML2018_SVD(Dataset):
    def __init__(self, data, label, snrs, domain_class, transform=None):
        super().__init__()
        self.X = np.transpose(data, (0, 2, 1))
        self.lbl = label
        self.snrs = snrs
        self.transform = transform
        self.domain_class = domain_class
        print("shape of all data:", self.X.shape, self.lbl.shape, self.snrs.shape)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index])
        x = x.unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        y = self.lbl[index]
        snr = self.snrs[index]
        dlabel = max(0, self.domain_class)
        return x, y, snr, dlabel, index

    def __len__(self):
        return (self.X.shape[0])


def load_six_data(path, batchsize=512, index_domain=None):
    # 初始化标签列表
    domain_list = [os.path.join(path, "gentbrugge"), os.path.join(path, "rabot"), os.path.join(path, "merelbeke"), os.path.join(path, "igent"),
                   os.path.join(path, "reep"), os.path.join(path, "uz")]
    np.random.seed(2016)
    # 分别提取每个域的训练，验证，测试数据
    train_datasets, val_datasets, test_datasets = [], [], []
    train_loaders, val_loaders, test_loaders = [], [], []
    test_X_all, test_Y_all = [], []
    for k in range(len(domain_list)):
        domain_path = domain_list[k]
        train_X_lists, val_X_lists, test_X_lists = [], [], []
        train_Y_lists, val_Y_lists, test_Y_lists = [], [], []
        dvbt_files = glob.glob(os.path.join(domain_path, 'dvbt*.bin'))
        wifi_files = glob.glob(os.path.join(domain_path, 'wf*.bin'))
        lte_files = glob.glob(os.path.join(domain_path, 'lte*.bin'))
        files_list = [dvbt_files, wifi_files, lte_files]
        for i in range(len(files_list)):
            files = files_list[i]
            for file_path in files:
                with open(file_path, 'rb') as file:
                    # 读取文件内容
                    raw2 = np.fromfile(file, dtype=np.float32)
                raw2 = raw2.T
                # 定义每个样本的点数
                samples = 8192
                # 计算可以完整分割的样本数
                arrlength = int(np.floor(len(raw2) / samples))
                # 去除不能完整分割的部分
                chop = len(raw2) - samples * arrlength
                raw2 = raw2[:-chop]

                # 分离 I 和 Q 通道
                raw3 = raw2.reshape(2, -1)
                q_sample = raw3[1, :]
                i_sample = raw3[0, :]

                # 重新整形 I 和 Q 样本
                i_sample_re = i_sample.reshape(4096, -1).T
                q_sample_re = q_sample.reshape(4096, -1).T

                # 挑选训练，验证，和测试样本
                n_examples = i_sample_re.shape[0]
                n_train = n_examples * 0.6
                n_val = n_examples * 0.2
                train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
                val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val),
                                           replace=False)
                test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

                # 合并 I 和 Q 样本
                data_train = np.concatenate(
                    (i_sample_re[train_idx, :, np.newaxis], q_sample_re[train_idx, :, np.newaxis]), axis=-1).transpose(0, 2, 1)
                data_val = np.concatenate((i_sample_re[val_idx, :, np.newaxis], q_sample_re[val_idx, :, np.newaxis]),
                                          axis=-1).transpose(0, 2, 1)
                data_test = np.concatenate((i_sample_re[test_idx, :, np.newaxis], q_sample_re[test_idx, :, np.newaxis]),
                                           axis=-1).transpose(0, 2, 1)
                print(data_train.shape)

                train_X_lists.append(data_train)
                val_X_lists.append(data_val)
                test_X_lists.append(data_test)

                train_Y_lists.append(np.ones(data_train.shape[0]) * i)
                val_Y_lists.append(np.ones(data_val.shape[0]) * i)
                test_Y_lists.append(np.ones(data_test.shape[0]) * i)
                test_X_all.append(data_test)
                test_Y_all.append(np.ones(data_test.shape[0]) * i)

        train_X = np.concatenate(train_X_lists, axis=0)
        train_Y = np.concatenate(train_Y_lists, axis=0)
        val_X = np.concatenate(val_X_lists, axis=0)
        val_Y = np.concatenate(val_Y_lists, axis=0)
        test_X = np.concatenate(test_X_lists, axis=0)
        test_Y = np.concatenate(test_Y_lists, axis=0)
        
        print(train_X.shape, train_Y.shape)
        train_set = Getdata(train_X, train_Y, 0) if k <= 1 else Getdata(train_X, train_Y, k-1)
        val_set = Getdata(val_X, val_Y, 0) if k <= 1 else Getdata(val_X, val_Y, k-1)
        test_set = Getdata(test_X, test_Y, 0) if k <= 1 else Getdata(test_X, test_Y, k-1)
        # train_set = Getdata(train_X, train_Y, k) 
        # val_set = Getdata(val_X, val_Y, k)
        # test_set = Getdata(test_X, test_Y, k)
        train_datasets.append(train_set)
        val_datasets.append(val_set)
        test_datasets.append(test_set)
        train_loaders.append(DataLoader(train_set, batch_size=batchsize, shuffle=True))
        val_loaders.append(DataLoader(val_set, batch_size=batchsize, shuffle=False))
        test_loaders.append(DataLoader(test_set, batch_size=batchsize, shuffle=False))
    # X = np.concatenate(test_X_all, axis=0)
    # Y = np.concatenate(test_Y_all, axis=0)
    # indices = np.arange(len(X))
    # random_indices = np.random.choice(indices, size=1000, replace=False)

    # X1 = X[random_indices]
    # Y1 = Y[random_indices]

    # np.save(r'/media/zxr/DATA2/XZTW/data/test/wir/X_test.npy', X1)
    # np.save(r'/media/zxr/DATA2/XZTW/data/test/wir/Y_test.npy', Y1)
    if index_domain is not None:
        random_domain = [[2, 3, 4, 5], [3, 2, 5, 4], [4, 5, 2, 3], [4, 5, 3, 2]]
        random_domain = random_domain[index_domain]
    else:
        random_domain = [2, 3, 4, 5]
    return random_domain, train_datasets[0:2], val_datasets[0:2], test_datasets[0:2], \
        [train_datasets[i] for i in random_domain], [val_datasets[i] for i in random_domain], [test_datasets[i] for i in
                                                                                               random_domain], \
        train_loaders[0:2], val_loaders[0:2], test_loaders[0:2], \
        [train_loaders[i] for i in random_domain], [val_loaders[i] for i in random_domain], [test_loaders[i] for i in
                                                                                             random_domain]
    # if index_domain is not None:
    #     random_domain = [[1, 2, 3, 4, 5], [1, 3, 2, 5, 4], [4, 5, 2, 3, 1], [1, 4, 5, 3, 2], [4, 1, 5, 2, 3]]
    #     random_domain = random_domain[index_domain]
    # else:
    #     random_domain = [1, 2, 3, 4, 5]
    # return random_domain, train_datasets[0], val_datasets[0], test_datasets[0], \
    #     [train_datasets[i] for i in random_domain], [val_datasets[i] for i in random_domain], [test_datasets[i] for i in
    #                                                                                            random_domain], \
    #     train_loaders[0], val_loaders[0], test_loaders[0], \
    #     [train_loaders[i] for i in random_domain], [val_loaders[i] for i in random_domain], [test_loaders[i] for i in
    #                                                                                          random_domain]


def load_six_data_six_zq(path, batchsize=512, index_domain=None):
    # 初始化标签列表
    domain_list = [os.path.join(path, "gentbrugge"), os.path.join(path, "rabot"), os.path.join(path, "merelbeke"), os.path.join(path, "igent"),
                   os.path.join(path, "reep"), os.path.join(path, "uz")]
    np.random.seed(2016)
    # 分别提取每个域的训练，验证，测试数据
    train_datasets, val_datasets, test_datasets = [], [], []
    train_loaders, val_loaders, test_loaders = [], [], []
    test_X_all, test_Y_all = [], []
    zq_list = [[0, 10, 0.1, 1.0, 0.1, 90],\
                [10, 10, 0.0, 1.0, 0.0, 90],\
                [0, 10, 0.0, 0.5, 0.1, 90],\
                [0, 10, 0.1, 1.0, 0.0, 0],\
                [10, 10, 0.0, 0.5, 0.1, 0],\
                [10, 0, 0.1, 1.0, 0.0, 0],\
                [0, 10, 0.1, 1.0, 0.0, 90],\
                [0, 10, 0.1, 0.5, 0.1, 90],\
                [10, 0, 0.1, 1.0, 0.1, 0],\
                [10, 10, 0.0, 1.0, 0.1, 90],\
                [10, 10, 0.1, 0.5, 0.0, 0],\
                [0, 10, 0.0, 0.5, 0.1, 0]]
    for k in range(len(domain_list)):
        domain_path = domain_list[k]
        train_X_lists, val_X_lists, test_X_lists = [], [], []
        train_Y_lists, val_Y_lists, test_Y_lists = [], [], []
        dvbt_files = glob.glob(os.path.join(domain_path, 'dvbt*.bin'))
        wifi_files = glob.glob(os.path.join(domain_path, 'wf*.bin'))
        lte_files = glob.glob(os.path.join(domain_path, 'lte*.bin'))
        files_list = [dvbt_files, wifi_files, lte_files]
        for i in range(len(files_list)):
            files = files_list[i]
            for file_path in files:
                with open(file_path, 'rb') as file:
                    # 读取文件内容
                    raw2 = np.fromfile(file, dtype=np.float32)
                raw2 = raw2.T
                # 定义每个样本的点数
                samples = 8192
                # 计算可以完整分割的样本数
                arrlength = int(np.floor(len(raw2) / samples))
                # 去除不能完整分割的部分
                chop = len(raw2) - samples * arrlength
                raw2 = raw2[:-chop]

                # 分离 I 和 Q 通道
                raw3 = raw2.reshape(2, -1)
                q_sample = raw3[1, :]
                i_sample = raw3[0, :]

                # 重新整形 I 和 Q 样本
                i_sample_re = i_sample.reshape(4096, -1).T
                q_sample_re = q_sample.reshape(4096, -1).T

                # 挑选训练，验证，和测试样本
                n_examples = i_sample_re.shape[0]
                n_train = n_examples * 0.6
                n_val = n_examples * 0.2
                train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
                val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val),
                                           replace=False)
                test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

                # 合并 I 和 Q 样本
                data_train = np.concatenate(
                    (i_sample_re[train_idx, :, np.newaxis], q_sample_re[train_idx, :, np.newaxis]), axis=-1).transpose(0, 2, 1)
                data_val = np.concatenate((i_sample_re[val_idx, :, np.newaxis], q_sample_re[val_idx, :, np.newaxis]),
                                          axis=-1).transpose(0, 2, 1)
                data_test = np.concatenate((i_sample_re[test_idx, :, np.newaxis], q_sample_re[test_idx, :, np.newaxis]),
                                           axis=-1).transpose(0, 2, 1)
                print(data_train.shape)

                train_X_lists.append(data_train)
                val_X_lists.append(data_val)
                test_X_lists.append(data_test)

                train_Y_lists.append(np.ones(data_train.shape[0]) * i)
                val_Y_lists.append(np.ones(data_val.shape[0]) * i)
                test_Y_lists.append(np.ones(data_test.shape[0]) * i)
                test_X_all.append(data_test)
                test_Y_all.append(np.ones(data_test.shape[0]) * i)

        train_X = np.concatenate(train_X_lists, axis=0)
        train_Y = np.concatenate(train_Y_lists, axis=0)
        val_X = np.concatenate(val_X_lists, axis=0)
        val_Y = np.concatenate(val_Y_lists, axis=0)
        test_X = np.concatenate(test_X_lists, axis=0)
        test_Y = np.concatenate(test_Y_lists, axis=0)
        time_values, freq_values, noise_values, amp_values, time_mask_values, phase_values = zq_list[k]
        train_X = apply_augmentation(train_X, time_values, freq_values, noise_values, amp_values, time_mask_values, phase_values)
        val_X = apply_augmentation(val_X, time_values, freq_values, noise_values, amp_values, time_mask_values, phase_values)
        test_X = apply_augmentation(test_X, time_values, freq_values, noise_values, amp_values, time_mask_values, phase_values)

        print(train_X.shape, train_Y.shape)
        train_set = Getdata(train_X, train_Y, 0) if k <= 1 else Getdata(train_X, train_Y, k-1)
        val_set = Getdata(val_X, val_Y, 0) if k <= 1 else Getdata(val_X, val_Y, k-1)
        test_set = Getdata(test_X, test_Y, 0) if k <= 1 else Getdata(test_X, test_Y, k-1)
        train_datasets.append(train_set)
        val_datasets.append(val_set)
        test_datasets.append(test_set)
        train_loaders.append(DataLoader(train_set, batch_size=batchsize, shuffle=True))
        val_loaders.append(DataLoader(val_set, batch_size=batchsize, shuffle=False))
        test_loaders.append(DataLoader(test_set, batch_size=batchsize, shuffle=False))
    if index_domain is not None:
        random_domain = [[2, 3, 4, 5], [3, 2, 5, 4], [4, 5, 2, 3], [4, 5, 3, 2]]
        random_domain = random_domain[index_domain]
    else:
        random_domain = [2, 3, 4, 5]
    return random_domain, train_datasets[0:2], val_datasets[0:2], test_datasets[0:2], \
        [train_datasets[i] for i in random_domain], [val_datasets[i] for i in random_domain], [test_datasets[i] for i in
                                                                                               random_domain], \
        train_loaders[0:2], val_loaders[0:2], test_loaders[0:2], \
        [train_loaders[i] for i in random_domain], [val_loaders[i] for i in random_domain], [test_loaders[i] for i in
                                                                                             random_domain]



class Getdata(Dataset):
    def __init__(self, data, label, dlabel, transform=None):
        super().__init__()
        self.X = data
        self.lbl = label
        self.dlabel = dlabel
        self.transform = transform
        print("shape of all data:", self.X.shape)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index])
        x = x.unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        y = self.lbl[index]
        return x, y, self.dlabel, self.dlabel, index

    def __len__(self):
        return (self.X.shape[0])

def split_wir(path, save_path=r"/data2/ALL_Data/Wir/"):
    # 初始化标签列表
    domain_list = [os.path.join(path, "gentbrugge"), os.path.join(path, "rabot"), os.path.join(path, "merelbeke"), os.path.join(path, "igent"),
                   os.path.join(path, "reep"), os.path.join(path, "uz")]
    np.random.seed(2016)
    # 分别提取每个域的训练，验证，测试数据
    train_X_lists, val_X_lists, test_X_lists = [], [], []
    train_Y_lists, val_Y_lists, test_Y_lists = [], [], []
    train_D_lists, val_D_lists, test_D_lists = [], [], []
    for k in range(len(domain_list)):
        domain_path = domain_list[k]
        dvbt_files = glob.glob(os.path.join(domain_path, 'dvbt*.bin'))
        wifi_files = glob.glob(os.path.join(domain_path, 'wf*.bin'))
        lte_files = glob.glob(os.path.join(domain_path, 'lte*.bin'))
        files_list = [dvbt_files, wifi_files, lte_files]
        for i in range(len(files_list)):
            files = files_list[i]
            for file_path in files:
                with open(file_path, 'rb') as file:
                    # 读取文件内容
                    raw2 = np.fromfile(file, dtype=np.float32)
                raw2 = raw2.T
                # 定义每个样本的点数
                samples = 2048
                # 计算可以完整分割的样本数
                arrlength = int(np.floor(len(raw2) / samples))
                # 去除不能完整分割的部分
                chop = len(raw2) - samples * arrlength
                raw2 = raw2[:-chop]

                # 分离 I 和 Q 通道
                raw3 = raw2.reshape(2, -1)
                q_sample = raw3[1, :]
                i_sample = raw3[0, :]

                # 重新整形 I 和 Q 样本
                i_sample_re = i_sample.reshape(1024, -1).T
                q_sample_re = q_sample.reshape(1024, -1).T

                # 挑选训练，验证，和测试样本
                n_examples = i_sample_re.shape[0]
                n_train = n_examples * 0.6
                n_val = n_examples * 0.2
                train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
                val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val),
                                           replace=False)
                test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

                # 合并 I 和 Q 样本
                data_train = np.concatenate(
                    (i_sample_re[train_idx, :, np.newaxis], q_sample_re[train_idx, :, np.newaxis]), axis=-1).transpose(0, 2, 1)
                data_val = np.concatenate((i_sample_re[val_idx, :, np.newaxis], q_sample_re[val_idx, :, np.newaxis]),
                                          axis=-1).transpose(0, 2, 1)
                data_test = np.concatenate((i_sample_re[test_idx, :, np.newaxis], q_sample_re[test_idx, :, np.newaxis]),
                                           axis=-1).transpose(0, 2, 1)

                train_X_lists.append(data_train)
                val_X_lists.append(data_val)
                test_X_lists.append(data_test)

                train_Y_lists.append(np.ones(data_train.shape[0]) * i)
                val_Y_lists.append(np.ones(data_val.shape[0]) * i)
                test_Y_lists.append(np.ones(data_test.shape[0]) * i)

                train_D_lists.append(np.ones(data_train.shape[0]) * k)
                val_D_lists.append(np.ones(data_val.shape[0]) * k)
                test_D_lists.append(np.ones(data_test.shape[0]) * k)

    train_X = np.concatenate(train_X_lists, axis=0)
    train_Y = np.concatenate(train_Y_lists, axis=0)
    train_D = np.concatenate(train_D_lists, axis=0)
    val_X = np.concatenate(val_X_lists, axis=0)
    val_Y = np.concatenate(val_Y_lists, axis=0)
    val_D = np.concatenate(val_D_lists, axis=0)
    test_X = np.concatenate(test_X_lists, axis=0)
    test_Y = np.concatenate(test_Y_lists, axis=0)
    test_D = np.concatenate(test_D_lists, axis=0)
    print(train_X.shape, train_Y.shape, train_D.shape)
    print(val_X.shape, val_Y.shape, val_D.shape)
    np.save(os.path.join(save_path, 'X_train'), train_X)
    np.save(os.path.join(save_path, 'Y_train'), train_Y)
    np.save(os.path.join(save_path, 'D_train'), train_D)
    np.save(os.path.join(save_path, 'X_val'), val_X)
    np.save(os.path.join(save_path, 'Y_val'), val_Y)
    np.save(os.path.join(save_path, 'D_val'), val_D)
    np.save(os.path.join(save_path, 'X_test'), test_X)
    np.save(os.path.join(save_path, 'Y_test'), test_Y)
    np.save(os.path.join(save_path, 'D_test'), test_D)



class JointDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self._len = sum([len(d) for d in self.datasets])
        self.lbl = np.concatenate([d.lbl for d in self.datasets])
        self.X = np.concatenate([d.X for d in self.datasets])
    def __len__(self):
        'Denotes the total number of samples'
        return self._len

    def __getitem__(self, index):
        for d in self.datasets:
            if len(d) <= index:
                index -= len(d)
            else:
                args = d[index]
                return args

if __name__ == '__main__':
    # path = r'2018.01/GOLD_XYZ_OSC.0001_1024.hdf5'
    # domain_num = 5
    # save_path = r'/origin_data/2018_domain_First10_inc4'
    #splitDomainRML2018(path=path, domain_num=domain_num, save_path=save_path)
    
    path = r'data/RML2016.10a_dict.pkl'
    domain_num = 49
    # save_path = r'./origin_data/First8_domain_num_49_7_zq'
    # splitDomainRML2016A_cd(path=path, domain_num=domain_num, save_path=save_path)
    # splitDomainRML2016A_7_methods(path=path, save_path=save_path, domain_num=5)
    save_path = r'data/Wir/wir'
    # load_six_data_six_zq(path=save_path, batchsize=512, index_domain=0)
    
    # path = r'./data/Wir/wir'
    # split_wir(path)
