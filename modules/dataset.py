from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
import os
from torch.utils.data import DataLoader
from collections import Counter
from scipy.interpolate import interp1d
import h5py
import pandas as pd
from scipy.io import loadmat

def loadRML2016A(path, test=None, index=0):
    Xd = pickle.load(open(path, 'rb'), encoding='latin')
    np.random.seed(2016)  
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X_train = []
    lbl_train = []
    X_val = []
    lbl_val = []
    X_test = []
    lbl_test = []

    # SNR:[-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    test_SNR = []
    test_SNR.append(snrs[0])
    test_SNR.append(snrs[-1])
    print(snrs)
    
    if test == True:
        snr = snrs[index]
        for mod in mods:
            # 对每个信噪比的数据均匀切分
            temp = Xd[(mod, snr)]
            label = (mod, snr)
            # 对预处理好的数据进行打包，并按6:2:2划分数据
            n_examples = temp.shape[0]
            n_train = n_examples * 0.6
            n_val = n_examples * 0.2
            train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
            val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=200, replace=False)
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
    else:
        for mod in mods:
            for snr in snrs:
                # 对每个模式下每个信噪比的数据均匀切分
                temp = Xd[(mod, snr)]
                label = (mod, snr)
                # 对预处理好的数据进行打包，并按6:2:2划分数据
                n_examples = temp.shape[0]
                n_train = n_examples * 0.6
                n_val = n_examples * 0.2
                train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
                val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=200, replace=False)
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

    # 得到训练数据和测试数据的类型标签
    Y_train = np.array(list(map(lambda x: mods.index(lbl_train[x][0]), np.arange(0, len(lbl_train)))))
    Y_val = np.array(list(map(lambda x: mods.index(lbl_val[x][0]), np.arange(0, len(lbl_val)))))
    Y_test = np.array(list(map(lambda x: mods.index(lbl_test[x][0]), np.arange(0, len(lbl_test)))))
    
    # 得到训练数据和测试数据的信噪比标签
    snr_train = np.array(list(map(lambda x: snrs.index(lbl_train[x][1]), np.arange(0, len(lbl_train)))))
    snr_val = np.array(list(map(lambda x: snrs.index(lbl_val[x][1]), np.arange(0, len(lbl_val)))))
    snr_test = np.array(list(map(lambda x: snrs.index(lbl_test[x][1]), np.arange(0, len(lbl_test)))))


    return (X_train, Y_train, snr_train), (X_val, Y_val, snr_val), (X_test, Y_test, snr_test)

class Getdata_RML2016A(Dataset):
    def __init__(self, data, label, transform = None):
        super().__init__()
        self.X = data
        self.lbl = label
        self.transform = transform
        print("shape of all data:", self.X.shape)
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index])
        x = x.unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        y = self.lbl[index]
        return x, y
        
    def __len__(self):
        return(self.X.shape[0])

class mydataset(object):
    def __init__(self):
        self.x = None
        self.labels = None
        self.dlabels = None
        self.pclabels = None
        self.pdlabels = None
        self.task = None
        self.dataset = None
        self.loader = None
        self.transform = None
        self.target_transform = None

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'pclabel':
            self.pclabels = tlabels
        elif label_type == 'pdlabel':
            self.pdlabels = tlabels
        elif label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def set_labels_by_index(self, tlabels=None, tindex=None, label_type='domain_label'):
        if label_type == 'pclabel':
            self.pclabels[tindex] = tlabels
        elif label_type == 'pdlabel':
            self.pdlabels[tindex] = tlabels
        elif label_type == 'domain_label':
            self.dlabels[tindex] = tlabels
        elif label_type == 'class_label':
            self.labels[tindex] = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        x = self.input_trans(self.x[index])
        pdtarget = self.target_trans(self.pdlabels[index])

        return x, pdtarget, index

    def __len__(self):
        return len(self.x)

def loadRML2016(path, test=None, index=0):
    Xd = pickle.load(open(path, 'rb'), encoding='latin')
    np.random.seed(3407)
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X_train = []
    lbl_train = []
    X_val = []
    lbl_val = []
    X_test = []
    lbl_test = []

    # SNR:[-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    test_SNR = []
    test_SNR.append(snrs[0])
    test_SNR.append(snrs[-1])
    
    if test == True:
        snr = snrs[index]
        mod_test = ['8PSK', 'BPSK', 'QAM16', 'QAM64', 'QPSK']
        for mod in mod_test:
            # 对每个信噪比的数据均匀切分
            temp = Xd[(mod, snr)]
            label = (mod, snr)
            # 对预处理好的数据进行打包，并按6:2:2划分数据，制作成投入网络训练的格式
            n_examples = temp.shape[0]
            n_train = int(n_examples * 0.6)
            n_val = int(n_examples * 0.2)
            train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
            val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val), replace=False)
            test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

            X_train.append(temp[train_idx])
            X_val.append(temp[val_idx])
            X_test.append(temp[test_idx])

            for i in range(int(n_train)):
                lbl_train.append(label)
            for i in range(int(n_train), int(n_train + n_val)):
                lbl_val.append(label)
            for i in range(int(n_train + n_val), int(n_examples)):
                lbl_test.append(label)
                
    else:
        snr_test = [-8, 18]
        mod_test = ['8PSK', 'BPSK', 'QAM16', 'QAM64', 'QPSK']
        for mod in mod_test:
            for snr in snrs:
                # 对每个模式下每个信噪比的数据均匀切分
                temp = Xd[(mod, snr)]
                label = (mod, snr)
                # 对预处理好的数据进行打包，并按6:2:2划分数据，制作成投入网络训练的格式
                n_examples = temp.shape[0]
                n_train = int(n_examples * 0.6)
                n_val = int(n_examples * 0.2)
                train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
                val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val), replace=False)
                test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

                X_train.append(temp[train_idx])
                X_val.append(temp[val_idx])
                X_test.append(temp[test_idx])

                for i in range(int(n_train)):
                    lbl_train.append(label)
                for i in range(int(n_train), int(n_train + n_val)):
                    lbl_val.append(label)
                for i in range(int(n_train + n_val), int(n_examples)):
                    lbl_test.append(label)
                
    X_train = np.vstack(X_train)
    X_val = np.vstack(X_val)
    X_test = np.vstack(X_test)

    # 得到训练数据和测试数据的类型标签
    Y_train = np.array(list(map(lambda x: mod_test.index(lbl_train[x][0]), np.arange(0, len(lbl_train)))))
    Y_val = np.array(list(map(lambda x: mod_test.index(lbl_val[x][0]), np.arange(0, len(lbl_val)))))
    Y_test = np.array(list(map(lambda x: mod_test.index(lbl_test[x][0]), np.arange(0, len(lbl_test)))))
    
    # 得到训练数据和测试数据的信噪比标签
    snr_train = np.array(list(map(lambda x: snrs.index(lbl_train[x][1]), np.arange(0, len(lbl_train)))))
    snr_val = np.array(list(map(lambda x: snrs.index(lbl_val[x][1]), np.arange(0, len(lbl_val)))))
    snr_test = np.array(list(map(lambda x: snrs.index(lbl_test[x][1]), np.arange(0, len(lbl_test)))))
    
    print(set(Y_train))
    
    return (X_train, Y_train, snr_train), (X_val, Y_val, snr_val), (X_test, Y_test, snr_test)

def loadRML2016B(path, test=None, index=0):
    Xd = pickle.load(open(path, 'rb'), encoding='iso-8859-1')
    np.random.seed(3407)  
    mods, snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1]]
    X_train = []
    lbl_train = []
    X_val = []
    lbl_val = []
    X_test = []
    lbl_test = []

    # SNR:[-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    test_SNR = []
    test_SNR.append(snrs[0])
    test_SNR.append(snrs[-1])
    
    if test == True:
        snr = snrs[index]
        mod_test = ['8PSK', 'BPSK', 'QAM16', 'QAM64', 'QPSK']
        for mod in mods:
        # for mod in mod_test:
            # 对每个信噪比的数据均匀切分
            temp = Xd[(mod, snr)]
            label = (mod, snr)
            # 对预处理好的数据进行打包，并按7:2:1划分数据，制作成投入网络训练的格式
            n_examples = temp.shape[0]
            n_train = int(n_examples * 0.7)
            n_val = int(n_examples * 0.2)
            train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
            val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val), replace=False)
            test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

            X_train.append(temp[train_idx])
            X_val.append(temp[val_idx])
            X_test.append(temp[test_idx])

            for i in range(int(n_train)):
                lbl_train.append(label)
            for i in range(int(n_train), int(n_train + n_val)):
                lbl_val.append(label)
            for i in range(int(n_train + n_val), int(n_examples)):
                lbl_test.append(label)
    else:
        mod_test = ['8PSK', 'BPSK', 'QAM16', 'QAM64', 'QPSK']
        for mod in mods:
        # for mod in mod_test:
            for snr in snrs:
                # 对每个模式下每个信噪比的数据均匀切分
                temp = Xd[(mod, snr)]
                label = (mod, snr)
                # 对预处理好的数据进行打包，并按7:2:1划分数据，制作成投入网络训练的格式
                n_examples = temp.shape[0]
                n_train = int(n_examples * 0.7)
                n_val = int(n_examples * 0.2)
                train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
                val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val), replace=False)
                test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

                X_train.append(temp[train_idx])
                X_val.append(temp[val_idx])
                X_test.append(temp[test_idx])

                for i in range(int(n_train)):
                    lbl_train.append(label)
                for i in range(int(n_train), int(n_train + n_val)):
                    lbl_val.append(label)
                for i in range(int(n_train + n_val), int(n_examples)):
                    lbl_test.append(label)
    
    X_train = np.vstack(X_train)
    X_val = np.vstack(X_val)
    X_test = np.vstack(X_test)

    # 得到训练数据和测试数据的类型标签
    Y_train = np.array(list(map(lambda x: mod_test.index(lbl_train[x][0]), np.arange(0, len(lbl_train)))))
    Y_val = np.array(list(map(lambda x: mod_test.index(lbl_val[x][0]), np.arange(0, len(lbl_val)))))
    Y_test = np.array(list(map(lambda x: mod_test.index(lbl_test[x][0]), np.arange(0, len(lbl_test)))))
    
    # 得到训练数据和测试数据的信噪比标签
    snr_train = np.array(list(map(lambda x: snrs.index(lbl_train[x][1]), np.arange(0, len(lbl_train)))))
    snr_val = np.array(list(map(lambda x: snrs.index(lbl_val[x][1]), np.arange(0, len(lbl_val)))))
    snr_test = np.array(list(map(lambda x: snrs.index(lbl_test[x][1]), np.arange(0, len(lbl_test)))))
    
    return (X_train, Y_train, snr_train), (X_val, Y_val, snr_val), (X_test, Y_test, snr_test)

def loadRML2018(path, test=None, index=0):
    f = h5py.File(path,'r')
    np.random.seed(3407)
    X = f['X'][:,:,:]  # ndarray(2555904*1024*2),shape
    Y = f['Y'][:,:]  # ndarray(2M*24),class
    Z = f['Z'][:]  # ndarray(2M*1),SNR
    
    # 将ONE-HOT改成数字
    Y = np.argmax(Y, axis=1)
    
    # 取出五类数据
    indices = np.where(np.isin(Y, [3, 4, 5, 12, 14]))
    
    X = X[indices]
    Y = Y[indices]
    Z = Z[indices]

    # change label
    label_mapping = {3:1, 5:0, 12:2, 14:3}

    vfunc = np.vectorize(lambda x: label_mapping.get(x, x))
    Y = vfunc(Y)
    
    n_examples = X.shape[0]
    n_train = int(n_examples * 0.6)
    n_val = int(n_examples * 0.2)

    train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
    val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val), replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    snr_train = Z[train_idx]
    X_val = X[val_idx]
    Y_val = Y[val_idx]
    snr_val = Z[val_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]
    snr_test = Z[test_idx]

    X_train = np.transpose(np.array(X_train), (0, 2, 1))
    X_test = np.transpose(np.array(X_test), (0, 2, 1))
    X_val= np.transpose(np.array(X_val), (0, 2, 1))

    # SNR [-2, 18]
    # snr_indices = []
    # for index, value in enumerate(snr_test):
    #     if -2 <= value <= 18:
    #         snr_indices.append(index)

    # X_test = X_test[snr_indices]
    # Y_test = Y_test[snr_indices]
    # snr_test = snr_test[snr_indices]
    
    return (X_train, Y_train, snr_train), (X_val, Y_val, snr_val), (X_test, Y_test, snr_test)

def loadHisarMOD2019(path, test_mod=None, index=0):
    train_path = path + '/Train/'
    test_path = path + '/Test/'
    
    data1 = h5py.File(train_path + 'train.mat','r')
    train=data1['data_save'][:]
    train=train.swapaxes(0,2)

    # data2 = h5py.File(test_path+'test.mat','r')
    # test=data2['data_save'][:]
    # test=test.swapaxes(0,2)
    
    train_labels = pd.read_csv(train_path + 'train_labels1.csv',header=None)
    train_labels=np.array(train_labels)
    
    test_labels = pd.read_csv(test_path + 'test_labels1.csv',header=None)
    test_labels =np.array(test_labels)
    
    train_snr=pd.read_csv(train_path + 'train_snr.csv',header=None)
    train_snr=np.array(train_snr)

    test_snr=pd.read_csv(test_path + 'test_snr.csv',header=None)
    test_snr=np.array(test_snr)
    
    # 取出五类数据
    train_indices = np.where(np.isin(train_labels, [0, 1, 2, 8, 10]))
    train = np.take(train, train_indices, axis=0)[0]
    train_labels = train_labels[train_indices]
    train_snr = train_snr[train_indices]
    
    # test_indices = np.where(np.isin(test_labels, [0, 1, 2, 8, 10]))    
    # test = np.take(test, test_indices, axis=0)[0]
    # test_labels = test_labels[test_indices]
    # test_snr = test_snr[test_indices]
    
    # change label
    label_mapping = {2:0, 0:1, 8:2, 10:3, 1:4}
    vfunc = np.vectorize(lambda x: label_mapping.get(x, x))
    train_labels = vfunc(train_labels)
    test_labels = vfunc(test_labels)
    
    n_examples = train.shape[0]
    n_train = int(n_examples * 0.8)
    train_idx = list(np.random.choice(range(0, n_examples), size=n_train, replace=False))
    val_idx = list(set(range(0, n_examples)) - set(train_idx))
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    
    X_train = train[train_idx]
    Y_train = train_labels[train_idx]
    snr_train = train_snr[train_idx]
    
    X_val = train[val_idx]
    Y_val = train_labels[val_idx]
    snr_val = train_labels[val_idx]
    
    # X_test = test
    # Y_test = test_labels
    # snr_test = test_snr
    
    return (X_train, Y_train, snr_train), (X_val, Y_val, snr_val)

def loadHisarMOD2019_test(path, test_mod=None, index=0):
    test_path = path + '/Test/'

    data2 = h5py.File(test_path+'test.mat','r')
    test=data2['data_save'][:]
    test=test.swapaxes(0,2)

    test_labels = pd.read_csv(test_path + 'test_labels1.csv',header=None)
    test_labels =np.array(test_labels)

    test_snr=pd.read_csv(test_path + 'test_snr.csv',header=None)
    test_snr=np.array(test_snr)
    
    # 取出五类数据
    test_indices = np.where(np.isin(test_labels, [0, 1, 2, 8, 10]))    
    test = np.take(test, test_indices, axis=0)[0]
    test_labels = test_labels[test_indices]
    test_snr = test_snr[test_indices]
    
    # change label
    label_mapping = {2:0, 0:1, 8:2, 10:3, 1:4}
    vfunc = np.vectorize(lambda x: label_mapping.get(x, x))
    test_labels = vfunc(test_labels)
    
    X_test = test
    Y_test = test_labels
    snr_test = test_snr
    
    # SNR [-2, 18]
    # snr_indices = []
    # for index, value in enumerate(snr_test):
    #     if -2 <= value <= 18:
    #         snr_indices.append(index)

    # X_test = X_test[snr_indices]
    # Y_test = Y_test[snr_indices]
    # snr_test = snr_test[snr_indices]
    
    return X_test, Y_test, snr_test

class Getdata_RML2018(Dataset):
    def __init__(self, data, label, transform = None):
        super().__init__()
        self.X = data
        self.lbl = label
        self.transform = transform
        print("shape of all data:", self.X.shape)
        
    def __getitem__(self, index):
        x = self.X[index]
        # set inter func [2, 1024] --> [2, 128]
        # f = interp1d(np.arange(x.shape[1]), x, kind='linear', axis=1)
        # x = f(np.linspace(0, 1023, 128))
        # x = x[:, ::8]
        x = x[:, :128]
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        y = self.lbl[index]
        return x, y
        
    def __len__(self):
        return(self.X.shape[0])

class Getdata_RML2016A(Dataset):
    def __init__(self, data, label, transform = None):
        super().__init__()
        self.X = data
        self.lbl = label
        self.transform = transform
        print("shape of all data:", self.X.shape)
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index])
        x = x.unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        y = self.lbl[index]
        return x, y
        
    def __len__(self):
        return(self.X.shape[0])
    
class Getdata_RML2016A_snr(Dataset):
    def __init__(self, data, label, snr, transform = None):
        super().__init__()
        self.X = data
        self.lbl = label
        self.snr = snr
        self.transform = transform
        print("shape of all data:", self.X.shape)
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index])
        x = x.unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        y = self.lbl[index]
        snr = self.snr[index]
        return x, y, snr
        
    def __len__(self):
        return(self.X.shape[0])
    
class Getdata_RML2016A_OODversion(mydataset):
    def __init__(self, data, label, snrlabel, transform = None):
        super().__init__()
        self.X = data
        self.lbl = label
        self.transform = transform
        self.slabels = snrlabel
        # 定义映射函数
        def mapping_function(snr):
            if 0 <= snr <= 4:
                return 0
            elif 5 <= snr <= 9:
                return 1
            elif 10 <= snr <= 19:
                return 2
            
        self.dlabels = np.array([mapping_function(value) for value in snrlabel])
        print("shape of all data:", self.X.shape)
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index])
        x = x.unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        # y ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
        y = self.lbl[index]
        dy = self.dlabels[index]
        sy = self.slabels[index]
        
        return x, y, dy, sy, index
        
    def __len__(self):
        return(self.X.shape[0])


def loadSCF(path, test_mod=None, index=0, len_one=8193):
    train_path = path + '/Train/'
    test_path = path + '/Test/'
    X_train, Y_train, SNR_train, X_test, Y_test, SNR_test \
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
                start_index = (X[0].shape[0]-len_one) // 2
                end_index = start_index + len_one
                X = [X[i][np.newaxis, start_index:end_index] for i in range(len(X))]
                X_train.extend(X)
                Y = data['train_class'].reshape(-1)
                Y_train.extend(Y)
                SNR = np.ones(Y.shape)*SNR
                SNR_train.extend(SNR)
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
                X_test.extend(X)
                Y = data['test_class'].reshape(-1)
                Y_test.extend(Y)
                SNR = np.ones(Y.shape) * SNR
                SNR_test.extend(SNR)
    # 转为numpy
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    SNR_train = np.array(SNR_train)
    SNR_test = np.array(SNR_test)

    # 按照信噪比进行划分，取

    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, SNR_train.shape, SNR_test.shape)

    return (X_train, Y_train, SNR_train), (X_test, Y_test, SNR_test)

class RML_ALL(Dataset):
    def __init__(self, root_path, flag='train', file_list=None, limit_size=None):
        self.all_mods = {'32PSK': 0, '8PSK': 1, 'AM-DSB': 2, 'OQPSK': 3, '8PAM': 4, 
                         '8QAM': 5, '8FSK': 6, '4QAM': 7, '16FSK': 8, 'OOK': 9, '128QAM': 10, 
                         'GMSK': 11, '16APSK': 12, 'BPSK': 13, 'WBFM': 14, 'QPSK': 15, 'AM-DSB-WC': 16, 
                         'AM-DSB-SC': 17, '4FSK': 18, 'AM-USB': 19, 'GFSK': 20, 'QAM16': 21, 'AM-SSB-SC': 22, 
                         'QAM64': 23, 'PM': 24, 'PAM4': 25, 'AM-SSB-WC': 26, '128APSK': 27, '32APSK': 28, '8ASK': 29, 
                         '32QAM': 30, 'AM-SSB': 31, '256QAM': 32, 'CPFSK': 33, '64PSK': 34, '16PSK': 35, '64APSK': 36, 'FM': 37, 
                         '2FSK': 38, '4ASK': 39, '16PAM': 40, 'AM-LSB': 41}
        self.map_func = np.vectorize(lambda x: self.all_mods.get(x, x))
        self.path = root_path
        self.path_2016A = self.path + '/2016A'
        self.path_2016B = self.path + '/2016B'
        self.path_2016C = self.path + '/2016C'
        self.path_2018 = self.path + '/2018'
        self.path_2019 = self.path + '/2019'
        (X_train, Y_train, snr_train), (X_val, Y_val, snr_val), (X_test, Y_test, snr_test) = self.load_all_data()
        print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)
        # print(Y_train)
        # print(np.array(set(Y_train)))
        # print(Y_val)
        # print(np.array(set(Y_val)))
        # print(Y_test)
        # print(np.array(set(Y_test)))

        if flag == 'train':
            self.X, self.lbl = X_train, Y_train
        elif flag == 'val':
            self.X, self.lbl = X_val, Y_val
        elif flag == 'test':
            self.X, self.lbl = X_test, Y_test
    
    def load_data(self, path):
        X_train = np.load(os.path.join(path, 'X_train.npy'))
        Y_train = np.load(os.path.join(path, 'Y_train.npy'))
        snr_train = np.load(os.path.join(path, 'snr_train.npy'))

        X_val = np.load(os.path.join(path, 'X_val.npy'))
        Y_val = np.load(os.path.join(path, 'Y_val.npy'))
        snr_val = np.load(os.path.join(path, 'snr_val.npy'))

        X_test = np.load(os.path.join(path, 'X_test.npy'))
        Y_test = np.load(os.path.join(path, 'Y_test.npy'))
        snr_test = np.load(os.path.join(path, 'snr_test.npy'))
        return (X_train, Y_train, snr_train), (X_val, Y_val, snr_val), (X_test, Y_test, snr_test)

    def load_all_data(self):
        (XA_train, YA_train, snrA_train), (XA_val, YA_val, snrA_val), (XA_test, YA_test, snrA_test) = self.load_data(self.path_2016A)
        YA_train, YA_val, YA_test = self.map_func(YA_train.reshape(-1)), self.map_func(YA_val.reshape(-1)), self.map_func(YA_test.reshape(-1))
        (XB_train, YB_train, snrB_train), (XB_val, YB_val, snrB_val), (XB_test, YB_test, snrB_test) = self.load_data(self.path_2016B)
        YB_train, YB_val, YB_test = self.map_func(YB_train.reshape(-1)), self.map_func(YB_val.reshape(-1)), self.map_func(YB_test.reshape(-1))
        (XC_train, YC_train, snrC_train), (XC_val, YC_val, snrC_val), (XC_test, YC_test, snrC_test) = self.load_data(self.path_2016C)
        YC_train, YC_val, YC_test = self.map_func(YC_train.reshape(-1)), self.map_func(YC_val.reshape(-1)), self.map_func(YC_test.reshape(-1))
        (X8_train, Y8_train, snr8_train), (X8_val, Y8_val, snr8_val), (X8_test, Y8_test, snr8_test) = self.load_data(self.path_2018)
        Y8_train, Y8_val, Y8_test = self.map_func(Y8_train.reshape(-1)), self.map_func(Y8_val.reshape(-1)), self.map_func(Y8_test.reshape(-1))
        (X9_train, Y9_train, snr9_train), (X9_val, Y9_val, snr9_val), (X9_test, Y9_test, snr9_test) = self.load_data(self.path_2019)
        Y9_train, Y9_val, Y9_test = self.map_func(Y9_train.reshape(-1)), self.map_func(Y9_val.reshape(-1)), self.map_func(Y9_test.reshape(-1))
        print(XA_train.shape)
        X_train = np.vstack([np.transpose(XA_train, (0, 1, 2)), np.transpose(XB_train, (0, 1, 2)), 
                             np.transpose(XC_train, (0, 1, 2)), np.transpose(X8_train[:, :128, :], (0, 2, 1)), X9_train[:, :, :128]])
        X_val = np.vstack([np.transpose(XA_val, (0, 1, 2)), np.transpose(XB_val, (0, 1, 2)), 
                             np.transpose(XC_val, (0, 1, 2)), np.transpose(X8_val[:, :128, :], (0, 2, 1)), X9_val[:, :, :128]])
        X_test = np.vstack([np.transpose(XA_test, (0, 1, 2)), np.transpose(XB_test, (0, 1, 2)), 
                             np.transpose(XC_test, (0, 1, 2)), np.transpose(X8_test[:, :128, :], (0, 2, 1)), X9_test[:, :, :128]])
        
        Y_train = np.hstack([YA_train, YB_train, YC_train, Y8_train, Y9_train])
        Y_val = np.hstack([YA_val, YB_val, YC_val, Y8_val, Y9_val])
        Y_test = np.hstack([YA_test, YB_test, YC_test, Y8_test, Y9_test])

        snr_train = np.hstack([snrA_train, snrB_train, snrC_train, snr8_train, snr9_train])
        snr_val = np.hstack([snrA_val, snrB_val, snrC_val, snr8_val, snr9_val])
        snr_test = np.hstack([snrA_test, snrB_test, snrC_test, snr8_test, snr9_test])

        return (X_train, Y_train, snr_train), (X_val, Y_val, snr_val), (X_test, Y_test, snr_test)
    
    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index])
        x = torch.squeeze(x)
        y = self.lbl[index]
        return x, y
    
    def __len__(self):
        return(self.X.shape[0])


    