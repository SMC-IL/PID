import copy
import sys
sys.path.append('/media/zxr/DATA1/lzy/XinCL')
sys.path.append('../XincL_local')
sys.path.append('../XinCL')
sys.path.append('/media/xaserver/data2/lzy/XinCL')
import numpy as np
import time
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
from modules.model import PETCGDNN, MCLDNN, DAE, CLDNN, SupConMCLDNN, feat_bottleneck, LinearClassifier, LSTMModel, GRUModel, MCLDNN_SVD, orthogonal_loss, ReverseLayerF
from data.dataset_sample_inc import splitRML2016A, splitDomainRML2016A, loadRML2016A, load_six_data_six_zq,loadDomainRML2016A, load_SCF_SVD, load_six_data, loadDomainRML2018_SVD, loadDomainRML2016A_SVD, JointDataset
from utils.util import plot_confusion_matrix, FocalLoss, setup_seed
from sklearn.metrics import confusion_matrix
import os
from utils.scheduler import PolynomialLR
import tqdm
import matplotlib.pyplot as plt
from buf_inc_Buf.save_buffer import load_buffer, save_buffer
from torch import optim
import glob
import torch.nn.functional as F
import random
import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


'''
    无回放数据，每次增量使用先前的模型作为教师模型，当前模型作为学生模型，进行蒸馏
'''


####参数
batchsize = 1024
start_epoch = 0
training_epoch = 200
#classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

"""
=== RML2016 数据读取 === 
"""
start = time.time()
path = r'./data/RML2016.10a_dict.pkl'
num_class, domain_num = 3, 5
# save_path, input_shape = r'/media/zxr/DATA1/lzy/XinCL/origin_data/First8_domain_num_49_7_zq', [2, 128]
# random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, \
# val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders = loadDomainRML2016A_SVD(save_path, domain_num=domain_num, train_bz=512, index_domain=0)
#
# save_path = r'origin_data/2018_domain_First10_inc4'
# random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, \
# val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders = loadDomainRML2018_SVD(save_path, domain_num=5, train_bz=512)
# num_class, domain_num = 4, 5
# path = r'/media/zxr/DATA1/lzy/XinCL/data/SCF'

# (random_domain, train_dataset, test_dataset, inc_train_datasets, inc_test_datasets, train_loader,
#      test_loader, inc_train_loaders, inc_test_loaders) = load_SCF_SVD(path, domain_num=domain_num, len_one=16)

# inc_train_datasets.append(train_dataset)
# inc_test_datasets.append(test_dataset)
# joint_train_dataset = JointDataset(inc_train_datasets)
# joint_test_dataset = JointDataset(inc_test_datasets)
# Jtest_loader = DataLoader(dataset=joint_test_dataset, batch_size=512, shuffle=False)

num_class, domain_num = 3, 5
save_path = r'/media/zxr/DATA1/lzy/XinCL/data/Wir/wir'

random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, \
                    val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders = load_six_data_six_zq(
                    save_path, batchsize=512)
test_dataset = JointDataset(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=1024,shuffle=False)

directory = '/media/zxr/DATA1/lzy/XinCL/check_point/MCLDNN/model_2016A/domain_split/Joint_49_7/MCLDNN_epoch_394_valAcc_0.30355263157894735.pth'

FAA_S_best, FAA_D_best, best_model = 0, 0, None
# 使用 glob 模块查找所有 .pth 文件
pth_files = glob.glob(directory+'*.pth')
best_path, best_acc = "", 0
for path in ['/media/zxr/DATA1/lzy/XinCL/check_point/Wir_CNN/Joint_6/MCLDNN_epoch_279_valAcc_0.6534090909090909.pth']:

    model = torch.load(path)

    # model_2016A.eval()
    # all_predicts = torch.empty(0, 1).cuda()
    # all_targets = torch.empty(0).cuda()
    # with torch.no_grad():
    #     for imgs, targets, snr, dlabel, _ in Jtest_loader:
    #         imgs = imgs.cuda().float()
    #         targets = targets.cuda().long()
    #         imgs1 = imgs[:, :, 0, :]
    #         imgs2 = imgs[:, :, 1, :]
    #         # outputs = model_2016A(imgs, imgs1, imgs2)
    #         outputs = model_2016A(imgs)
    #
    #         predicts = outputs.argmax(dim=1, keepdim=True)
    #         all_targets = torch.cat([all_targets, targets])
    #         all_predicts = torch.cat([all_predicts, predicts], dim=0)
    #
    #     correct_ = all_predicts.eq(all_targets.view_as(all_predicts)).sum().item()
    #     accuracy_ = correct_ / float(len(joint_test_dataset))
    #     print("val_acc:", accuracy_)
    #模型测试
    model.eval()
    all_predicts = torch.empty(0, 1).cuda()
    all_targets = torch.empty(0).cuda()
    correct = torch.zeros(1).squeeze().cuda()
    correct_ = list(0. for i in range(num_class))
    accs = np.zeros((domain_num,))
    for i in range(domain_num):
        model.eval()
        all_predicts = torch.empty(0, 1).cuda()
        all_targets = torch.empty(0).cuda()
        if i == 0:
            test_L = test_loader
            test_D = test_dataset
        else:
            test_L = inc_test_loaders[i - 1]
            test_D = inc_test_datasets[i - 1]
        with torch.no_grad():
            for imgs, targets, snr, _, _ in test_L:
                imgs = imgs.cuda().float()
                targets = targets.cuda().long()

                imgs1 = imgs[:, :, 0, :]
                imgs2 = imgs[:, :, 1, :]
                # outputs = lcf(bottleneck(model_2016A.encoder(imgs, imgs1, imgs2)))
                # outputs = model_2016A(imgs, imgs1, imgs2)
                # outputs = model_2016A(imgs, imgs1, imgs2)
                # outputs = model(imgs, imgs1, imgs2)
                outputs = model(imgs)
                predicts = outputs.argmax(dim=1, keepdim=True)
                all_targets = torch.cat([all_targets, targets])
                all_predicts = torch.cat([all_predicts, predicts], dim=0)

            correct_ = all_predicts.eq(all_targets.view_as(all_predicts)).sum().item()
            accuracy_ = correct_ / float(len(test_D))
            accs[i] = accuracy_ * 100
            print(accs)
    print(f'accs:', accs)
    data_len = np.zeros((domain_num, ))
    for i in range(domain_num):
        if i == 0:
            test_D = test_dataset
        else:
            test_D = inc_test_datasets[i - 1]
        data_len[i] = len(test_D)
    acc_path = os.path.join(r'../check_point/MCLDNN_SVD/model/domain_split/First8_domain_num_5/SVD_inc', 'Accuracy.txt')
    # np.save(acc_path, accs)
    FAA_D = accs.sum() / float(domain_num)
    FAA_S = np.sum(accs*data_len)/np.sum(data_len)
    print(f'{path}:FAA:{"%.4f"%FAA_D}    FAA_S:{"%.4f"%FAA_S}')
    # if FAA_S > FAA_S_best:
    #     FAA_S_best = FAA_S
    #     FAA_D_best = FAA_D
    #     best_model = path
    # print(f'bestpath:{best_model}: FAA_D:{"%.4f"%FAA_D}    FAA_S:{"%.4f"%FAA_S}')





