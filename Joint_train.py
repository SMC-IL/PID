import copy
import sys
import numpy as np
import time
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
from modules.model import PETCGDNN, MCLDNN, DAE, CLDNN, SupConMCLDNN, feat_bottleneck, Wir_CNN, Wir_CNN_BN, SCF_CNN, SCF_CNN_16, LinearClassifier, LSTMModel, GRUModel, MCLDNN_SVD, orthogonal_loss, PETCGDNN2, ReverseLayerF, ICAMC
from data.dataset_sample_inc import splitRML2016A, splitDomainRML2016A, loadRML2016A, load_six_data_six_zq,loadDomainRML2016A, load_six_data, load_SCF_SVD, load_SCF_SVD, loadDomainRML2016A_SVD, JointDataset, loadDomainRML2018_SVD
from utils.util import plot_confusion_matrix, FocalLoss, setup_seed
from sklearn.metrics import confusion_matrix
import os
from utils.scheduler import PolynomialLR
import tqdm
import matplotlib.pyplot as plt
from buf_inc_Buf.save_buffer import load_buffer, save_buffer
from torch import optim

import torch.nn.functional as F
import random
import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


'''
    无回放数据，每次增量使用先前的模型作为教师模型，当前模型作为学生模型，进行蒸馏
'''


####参数
batchsize = 512
start_epoch = 0
training_epoch = 300
#classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

"""
=== RML2016 数据读取 === 
"""
start = time.time()
# path = r'./data/RML2016.10a_dict.pkl'
num_class, domain_num = 3, 5
# save_path, input_shape = r'origin_data/First8_domain_num_49_7_zq', [2, 128]
# save_path, input_shape = r'origin_data/2018_domain_First10_inc4', [2, 1024]

# random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, \
# val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders = loadDomainRML2016A_SVD(save_path, domain_num=domain_num, train_bz=512, index_domain=0)
#
# random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, \
# val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders = loadDomainRML2018_SVD(save_path, domain_num=5, train_bz=batchsize)


# path = r'/data/SCF'

# (random_domain, train_dataset, test_dataset, inc_train_datasets, inc_test_datasets, train_loader,
#      test_loader, inc_train_loaders, inc_test_loaders) = load_SCF_SVD(path, domain_num=domain_num, len_one=16)
# inc_train_datasets.append(train_dataset)
# inc_test_datasets.append(test_dataset)
# joint_train_dataset = JointDataset(inc_train_datasets)
# joint_test_dataset = JointDataset(inc_test_datasets)
# Jtrain_loader = DataLoader(dataset=joint_train_dataset, batch_size=batchsize, shuffle=True)
# Jtest_loader = DataLoader(dataset=joint_test_dataset, batch_size=512, shuffle=False)


save_path = r'data/Wir/wir'

random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, \
                    val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders = load_six_data_six_zq(
                    save_path, batchsize=512)
            
inc_train_datasets.extend(train_dataset)
inc_test_datasets.extend(test_dataset)
joint_train_dataset = JointDataset(inc_train_datasets)
joint_test_dataset = JointDataset(inc_test_datasets)
Jtrain_loader = DataLoader(dataset=joint_train_dataset, batch_size=1024, shuffle=True)
Jtest_loader = DataLoader(dataset=joint_test_dataset, batch_size=512, shuffle=False)
# inc_val_datasets.append(val_dataset)
# joint_val_dataset = JointDataset(inc_val_datasets)
# Jval_loader = DataLoader(dataset=joint_val_dataset, batch_size=512, shuffle=False)
# joint_val_dataset = joint_test_dataset
# Jval_loader = Jtest_loader
end = time.time()
print("load dataset time: {:.3f} s".format(end - start))
# model = MCLDNN(classes=num_class).cuda()
# model = PETCGDNN2(classes=num_class, input_shape=input_shape).cuda()
# model = SCF_CNN().cuda()
# model = SCF_CNN_16().cuda()
model = Wir_CNN().cuda()
for name, param in model.named_parameters():
    if 'weight' in name:
        nn.init.kaiming_normal_(param.data)
    param.requires_grad = True
setup_seed(3407)
"""
=== joint ===
"""
Train=True
accs = np.zeros((domain_num, domain_num))
if Train:
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, betas=(0.9, 0.99),
                                  weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    CrossLoss = nn.CrossEntropyLoss()  # .cuda()

    correct = torch.zeros(1).squeeze().cuda()
    correct_ = list(0. for i in range(num_class))
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_acc = 0
    for epoch in range(start_epoch+1, training_epoch+1, 1):
        # 模型训练
        model.train()
        with tqdm.tqdm(Jtrain_loader, unit="batch") as tepoch:
            for idx, (data, target, snr, dlabel, _) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                data, target, dlabel = data.cuda().float(), target.cuda().long(), dlabel.cuda().long()   # Data to device

                # 获取IQ序列的单独向量
                data1 = data[:, :, 0, :].cuda().float()
                data2 = data[:, :, 1, :].cuda().float()
                output = model(data)
                # output = model(data, data1, data2)
                losstr = CrossLoss(output, target)
                optimizer.zero_grad()
                losstr.backward()
                optimizer.step()
                predict_ = output.argmax(dim=1, keepdim=True)
                correct = predict_.eq(target.view_as(predict_)).sum().item()

                accuracy = correct/len(data)
                tepoch.set_postfix(loss=losstr.item(), accuracy='{:.3f}'.format(accuracy))

        if (epoch + 1) % 5 == 0:
            scheduler.step()
        epochs.append(epoch)
        train_losses.append(losstr.item())
        train_accs.append(accuracy)


        #模型测试
        model.eval()
        all_predicts = torch.empty(0, 1).cuda()
        all_targets = torch.empty(0).cuda()
        with torch.no_grad():
            for imgs, targets, snr, dlabel, _ in Jtest_loader:

                imgs = imgs.cuda().float()
                targets = targets.cuda().long()
                imgs1 = imgs[:,:,0,:]
                imgs2 = imgs[:,:,1,:]
                outputs = model(imgs)

                predicts = outputs.argmax(dim=1, keepdim=True)
                all_targets = torch.cat([all_targets, targets])
                all_predicts = torch.cat([all_predicts, predicts], dim=0)

            correct_ = all_predicts.eq(all_targets.view_as(all_predicts)).sum().item()
            accuracy_ = correct_/len(joint_test_dataset)
            print("val_acc:", accuracy_)

        val_losses.append(losstr.item())
        val_accs.append(accuracy_)

        if accuracy_ > best_acc:
            best_acc = accuracy_
            torch.save(model,
                       r'/media/zxr/DATA1/lzy/XinCL/check_point/Wir_CNN/Joint_6/MCLDNN_epoch_{}_valAcc_{}.pth'.format(
                           epoch + 1, accuracy_))






