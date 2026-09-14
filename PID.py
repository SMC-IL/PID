import copy
import sys
sys.path.append('/media/zxr/DATA1/lzy/XinCL')
sys.path.append('../XincL_local')
sys.path.append('/data/lzy/XinCL/')
import pandas as pd
import numpy as np
import time
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
from modules.model import PETCGDNN, MCLDNN, DAE, CLDNN, SupConMCLDNN, feat_bottleneck, PETCGDNN2_SVD, LinearClassifier, LSTMModel, GRUModel, MCLDNN_SVD, PETCGDNN_SVD, CLDNN_SVD, orthogonal_loss, ReverseLayerF, cosine_similarity_loss
from data.dataset_sample_inc import splitRML2016A, splitDomainRML2016A, loadRML2016A, loadDomainRML2018_SVD, loadDomainRML2016A, loadDomainRML2016A_SVD
from utils.util import plot_confusion_matrix, FocalLoss, setup_seed
from sklearn.metrics import confusion_matrix
import os
from utils.scheduler import PolynomialLR, WarmupLR
import tqdm
import matplotlib.pyplot as plt
from buf_inc_Buf.save_buffer import load_buffer, save_buffer
from torch import optim

import torch.nn.functional as F
import random
import argparse


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# -*- coding: utf-8 -*-
#### 参数
dataset = 2016
if dataset == 2018:
    batchsize = 256
else:
    batchsize = 512
start_epoch = 0
training_epoch = 20
#classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

"""
=== RML2016 数据读取 === 
"""
start = time.time()
path = r'./data/RML2016.10a_dict.pkl'
seed_list = [3407, 0, 2024]
backbone_list = ['MCLDNN', 'PETCGDNN', 'CLDNN']
for backbone in backbone_list[1:2]:
    One_flag = False
    for seed in seed_list:
        setup_seed(seed)
        for ind_dom in range(0, 4, 1):
            if dataset == 2018:
                # 2018
                num_class, domain_num, sto_rate, amt = 24, 5, 0.001, batchsize//4
                save_path, input_size = r'../origin_data/2018_domain_First10_inc4', [2, 1024]
                random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, \
                val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders = loadDomainRML2018_SVD(save_path, domain_num=5, train_bz=batchsize, index_domain=ind_dom)
            else:
                # 2016
                num_class, domain_num, sto_rate, amt = 11, 5, 0.005, batchsize//4
                save_path, input_size = r'/data/lzy/XinCL/origin_data/First8_domain_num_'+str(domain_num)+'_cd63', [2, 128]
                random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, \
                    val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders = loadDomainRML2016A_SVD(
                    save_path, domain_num=5, train_bz=batchsize, index_domain=ind_dom)

            

            """
            === buf增量 ===
            """
            Train = True
            CrossLoss = nn.CrossEntropyLoss()
            accs = np.zeros((domain_num, domain_num))
            if backbone == "MCLDNN":
                if dataset == 2018:
                    pre_model = torch.load(
                        r'../check_point/MCLDNN/model_2018/domain_split/First10_inc4/MCLDNN_epoch_23_valAcc_0.9377136752136752.pth')
                else:
                    pre_model = torch.load(
                        r'../check_point/MCLDNN/model_2016A/domain_split/First8_domain_num_5/MCLDNN_epoch_12_valAcc_0.8839772727272728.pth')
            if backbone == "PETCGDNN":
                if dataset == 2018:
                    pre_model = torch.load(
                        r'../check_point/PETCGDNN/model_2018/domain_split/First10_inc4/PETCGDNN_epoch_46_valAcc_0.8835164835164835.pth')
                else:
                    pre_model = torch.load(
                        r'check_point/PETCGDNN2/model_2016A/domain_split/First8_domain_num_5/PETCGDNN_epoch_195_valAcc_0.9194886363636363.pth')
            if backbone == "CLDNN":
                if dataset == 2018:
                    pre_model = torch.load(
                        r'check_point/MCLDNN/model_2018/domain_split/First10_inc4/MCLDNN_epoch_23_valAcc_0.9377136752136752.pth')
                else:
                    pre_model = torch.load(
                        r'../check_point/CLDNN/model_2016A/domain_split/First8_domain_num_5/CLDNN_epoch_187_valAcc_0.7263636363636363.pth')
                One_flag = True
            # pre_model = torch.load(r'check_point/PETCGDNN/model_2018/domain_split/First10_inc4/PETCGDNN_epoch_46_valAcc_0.8835164835164835.pth')
            print('-'*20+'预训练模型保存buf'+'-'*20)
            save_path = r'/data/lzy/XinCL/methods/buf_inc_Buf'
            save_buffer(train_dataset, pre_model, input_two=input_size[1], rate=sto_rate, save_path=save_path, num_class=num_class, One_flag=One_flag)
            # 加载buf
            buffer = load_buffer(load_path=save_path, input_two=input_size[1], num_class=num_class)
            pre_model.eval()
            pretrained_state_dict = pre_model.state_dict()
            # print(pretrained_state_dict.keys())
            S_dict = {k: v for k, v in pretrained_state_dict.items() if 'S' in k}
            fc_dict = {k: v for k, v in pretrained_state_dict.items() if 'fc' in k and 'encoder' not in k}
            bias_dict = {k: v for k, v in pretrained_state_dict.items() if 'fc' in k and 'bias' in k and 'encoder' not in k}
            if backbone == "MCLDNN":
                beta, lr = 5, 1e-2
                model = MCLDNN_SVD(classes=num_class, domain_classes=1, fc_dict=fc_dict)
            if backbone == "PETCGDNN":
                beta, lr = 10, 1e-2
                model = PETCGDNN2_SVD(input_shape=input_size, classes=num_class, domain_classes=1, fc_dict=fc_dict)
            if backbone == "CLDNN":
                beta, lr = 10, 1e-2
                model = CLDNN_SVD(classes=num_class, domain_classes=1, fc_dict=fc_dict)

            # model = PETCGDNN_SVD(input_shape=input_size, classes=num_class, domain_classes=1, fc_dict=fc_dict)
            model_state_dict = model.state_dict()
            # print(model_state_dict.keys(), pretrained_state_dict.keys())
            new_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict and 'encoder' in k}
            for name, param in model.named_parameters():
                print(name, param.data.shape)
            print(new_state_dict.keys())
            model.load_state_dict(new_state_dict, strict=False)

            # 统计模型参数
            total_params = sum(p.numel() for p in model.parameters())
            # total_params += sum(p.numel() for p in model.buffers())

            print(f'{total_params:,} total parameters.')
            print(f'{total_params / (1024):.2f}K total parameters.')

            model.cuda()
            model.eval()
            print('-'*20+'预训练模型测试所有域数据'+'-'*20)
            # 预训练模型先测试一轮，保存accs
            for i in range(domain_num):
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
                        if One_flag:
                            outputs, _, _, _, _ = model(imgs, only_flag=True)
                        else:
                            outputs, _, _, _, _ = model(imgs, imgs1, imgs2, only_flag=True)
                        predicts = outputs.argmax(dim=1, keepdim=True)
                        all_targets = torch.cat([all_targets, targets])
                        all_predicts = torch.cat([all_predicts, predicts], dim=0)

                    correct_ = all_predicts.eq(all_targets.view_as(all_predicts)).sum().item()
                    accuracy_ = correct_ / float(len(test_D))
                    accs[0][i] = accuracy_ * 100
            print(accs[0])
            pre_model = copy.deepcopy(model)

            '''
                先增量训练20轮，保存模型，然后调用save_buffer保存回放数据，然后更新buffer
            '''

            st_time = time.time()
            total_pas = []
            for k in range(1, domain_num):

                pre_model.eval()
                pretrained_state_dict = pre_model.state_dict()
                S_dict = {k: v for k, v in pretrained_state_dict.items() if 'S' in k}
                fc_dict = {k: v for k, v in pretrained_state_dict.items() if 'fc' in k and 'encoder' not in k}
                bias_dict = {k: v for k, v in pretrained_state_dict.items() if 'fc' in k and 'bias' in k and 'encoder' not in k}
                if backbone == "MCLDNN":
                    model = MCLDNN_SVD(classes=num_class, domain_classes=k+1, fc_dict=fc_dict)
                if backbone == "PETCGDNN":
                    model = PETCGDNN2_SVD(input_shape=input_size, classes=num_class, domain_classes=k+1, fc_dict=fc_dict)
                if backbone == "CLDNN":
                    model = CLDNN_SVD(classes=num_class, domain_classes=k+1, fc_dict=fc_dict)
                # model = PETCGDNN_SVD(input_shape=input_size, classes=num_class, domain_classes=k+1, fc_dict=fc_dict)
                model_state_dict = model.state_dict()
                # print(model_state_dict.keys(), pretrained_state_dict.keys())
                new_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict and 'encoder' in k}

                print(new_state_dict.keys())
                model.load_state_dict(new_state_dict, strict=False)

                # 统计模型参数
                total_params1 = sum(p.numel() for p in model.parameters())
                # total_params += sum(p.numel() for p in model.buffers())
                total_pas.append(total_params1)
                print(f'{total_params1:,} total parameters.')
                print(f'{total_params1 / (1024):.2f}K total parameters.')
                print(f'Inc:{(total_params1-total_params)/1024:.2f}K.')
                print(f'Inc(%):{(total_params1-total_params) / total_params * 100:.2f}.')
                if k >= 2:
                    print(f'Inc:{(total_pas[len(total_pas)-1] - total_pas[len(total_pas)-2]) / 1024:.2f}K.')
                    print(f'Inc(%):{(total_pas[len(total_pas)-1] - total_pas[len(total_pas)-2]) / total_pas[0] * 100:.2f}.')

                best_acc = 0
                print('-' * 20 + f"增量第{k}个域开始训练" + '-' * 20)
                print('buffer数据量:', buffer.buffer_num)
                for name, param in model.named_parameters():
                    param.requires_grad = False
                for name, param in model.named_parameters():
                    if 'prompt_layer' in name:
                        param.requires_grad = True
                for name, param in model.named_parameters():
                    if 'fc' in name and 'weight' not in name and 'encoder' not in name:
                        param.requires_grad = True

                # for name, param in model_2016A.named_parameters():
                #     if 'S' in name:
                #         param.data[:-1] = S_dict[name]

                model.cuda()
                for name, param in model.named_parameters():
                    print(name, param.requires_grad, param.shape, param.device)

                if Train:

                    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.99), weight_decay=1e-5)
                    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

                    CrossLoss = nn.CrossEntropyLoss()#.cuda()

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
                        with tqdm.tqdm(inc_train_loaders[k-1], unit="batch") as tepoch:
                            for idx, (data, target, snr, dlabel, _) in enumerate(tepoch):
                                if data.shape[0] < batchsize:
                                    break
                                tepoch.set_description('Epoch:'+ str(epoch))
                                data, target, dlabel = data.cuda().float(), target.cuda().long(), dlabel.cuda().long()   # Data to device
                                dd = dlabel[0]
                                # 加入buffer数据
                                # print(data.shape, target.shape, snr.shape)
                                buffer.shuffle_()
                                for d in range(k):
                                    buffer_x, buffer_y, buffer_logits, buffer_s, buffer_dlabel = buffer.onlysample(amt=batchsize//k, task=d)
                                    data = torch.cat([data, buffer_x.unsqueeze(1).cuda().float()], dim=0)
                                    target = torch.cat([target, buffer_y.cuda().long()])
                                    dlabel = torch.cat([dlabel, buffer_dlabel.cuda().long()])

                                # 获取IQ序列的单独向量
                                data1 = data[:, :, 0, :].cuda().float()
                                data2 = data[:, :, 1, :].cuda().float()
                                for name, param in model.named_parameters():
                                    # 只训练S矩阵的最后一行参数，其余参数冻结
                                    if 'S' in name:
                                        param.data[:k, :] = S_dict[name]
                                    if 'fc' in name and 'bias' in name and 'encoder' not in name:
                                        param.data[:k, :] = bias_dict[name]
                                if One_flag:
                                    output, dpredict, s_1, s_2, s_3 = model(data)
                                else:
                                    output, dpredict, s_1, s_2, s_3 = model(data, data1, data2)
                                loss_prompt = CrossLoss(dpredict, dlabel)
                                losst = CrossLoss(output, target)
                                if epoch <= beta:
                                    # losstr = losst + loss_prompt
                                    losstr = losst + loss_prompt
                                    optimizer.zero_grad()
                                    losstr.backward()
                                else:
                                    # for name, param in model_2016A.named_parameters():
                                    #     if 'prompt_layer' in name:
                                    #         param.requires_grad = False

                                    # 找到是这个域并且分对域的样本
                                    # dd = dpredict.argmax(dim=1, keepdim=True).reshape(-1)
                                    # index_true = torch.where(dd==dlabel)[0]
                                    # losstr = CrossLoss(output[index_true], target[index_true]) + loss_orth
                                    losstr = CrossLoss(output, target)
                                    optimizer.zero_grad()
                                    losstr.backward()
                                    for name, param in model.named_parameters():
                                        if 'prompt_layer' in name:
                                            # print(param)
                                            # param.grad.zero_()
                                            param.requires_grad=False
                                # losstr = CrossLoss(output, target)
                                # for name, param in model_2016A.named_parameters():
                                #     # 只训练S矩阵的最后一行参数，其余参数冻结
                                #     if 'S' in name:
                                #         param.grad[:k].zero_()
                                #     if 'fc' in name and 'bias' in name:
                                #         param.grad[:k].zero_()
                                optimizer.step()
                                for name, param in model.named_parameters():
                                    # 只训练S矩阵的最后一行参数，其余参数冻结
                                    if 'S' in name:
                                        param.data[:k, :] = S_dict[name]
                                    if 'fc' in name and 'bias' in name and 'encoder' not in name:
                                        param.data[:k, :] = bias_dict[name]
                                predict_ = output.argmax(dim=1, keepdim=True)
                                correct = predict_.eq(target.view_as(predict_)).sum().item()

                                accuracy = correct/len(data)
                                tepoch.set_postfix(loss=losstr.item(), accuracy='{:.3f}'.format(accuracy))
                        print(cosine_similarity_loss(s_1) + cosine_similarity_loss(s_2)+cosine_similarity_loss(s_3))
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
                            for imgs, targets, snr, dlabel, _ in inc_val_loaders[k-1]:

                                imgs = imgs.cuda().float()
                                targets = targets.cuda().long()
                                imgs1 = imgs[:,:,0,:]
                                imgs2 = imgs[:,:,1,:]
                                # outputs, _, _, _, _ = model_2016A(imgs, imgs1, imgs2)
                                if One_flag:
                                    outputs, _, _, _, _ = model(imgs)
                                else:
                                    outputs, _, _, _, _ = model(imgs, imgs1, imgs2)

                                predicts = outputs.argmax(dim=1, keepdim=True)
                                all_targets = torch.cat([all_targets, targets])
                                all_predicts = torch.cat([all_predicts, predicts], dim=0)

                            correct_ = all_predicts.eq(all_targets.view_as(all_predicts)).sum().item()
                            accuracy_ = correct_/float(len(inc_val_datasets[k-1]))
                            print("val_acc:", accuracy_)
                        val_accs.append(accuracy_)
                    # torch.save(model_2016A, r'check_point/MCLDNN_SVD/model_2016A/domain_split/First8_domain_num_5/SVD_inc/D{}_MCLDNN_SVD_epoch_{}_valAcc_{}.pth'.format(k,training_epoch, accuracy_))
                    # np.savetxt(r'check_point/MCLDNN_SVD/model_2016A/domain_split/First8_domain_num_5/SVD_inc/train_acc_{}.txt'.format(k), train_accs)
                    # np.savetxt(r'check_point/MCLDNN_SVD/model_2016A/domain_split/First8_domain_num_5/SVD_inc/train_loss_{}.txt'.format(k), train_losses)
                    # np.savetxt(r'check_point/MCLDNN_SVD/model_2016A/domain_split/First8_domain_num_5/SVD_inc/val_acc_{}.txt'.format(k), val_accs)
                    # np.savetxt(r'check_point/MCLDNN_SVD/model_2016A/domain_split/First8_domain_num_5/SVD_inc/val_loss_{}.txt'.format(k), val_losses)
                if k < domain_num-1:
                    print('-' * 20 + '增量第'+str(k)+'个域开始保存更新buffer' + '-' * 20)
                    save_buffer(inc_train_datasets[k-1], model, input_two=input_size[1], rate=sto_rate, save_path=save_path, domain=k, num_class=num_class, One_flag=One_flag)
                    # 加载buf
                    buffer = load_buffer(load_path=save_path, domain=k, input_two=input_size[1], num_class=num_class)
                print('-' * 20 + '增量第'+str(k)+'个域开始测试所有域' + '-' * 20)
                # 模型测试, 保存acc
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
                            # outputs, _, _, _, _ = model_2016A(imgs, imgs1, imgs2)
                            if One_flag:
                                outputs, _, _, _, _ = model(imgs)
                            else:
                                outputs, _, _, _, _ = model(imgs, imgs1, imgs2)
                            predicts = outputs.argmax(dim=1, keepdim=True)
                            all_targets = torch.cat([all_targets, targets])
                            all_predicts = torch.cat([all_predicts, predicts], dim=0)

                        correct_ = all_predicts.eq(all_targets.view_as(all_predicts)).sum().item()
                        accuracy_ = correct_ / float(len(test_D))
                        accs[k][i] = accuracy_ * 100
                print(f'accs[{k}]:', accs[k])
                pre_model = copy.deepcopy(model)
            end_time = time.time()
            print('增量训练4次，时间为:', end_time-st_time, '秒')
            data_len = np.zeros((domain_num,))
            for i in range(domain_num):
                if i == 0:
                    test_D = test_dataset
                else:
                    test_D = inc_test_datasets[i - 1]
                data_len[i] = len(test_D)
            print('-' * 20 + f'计算保存FAA和FF' + '-' * 20)
            accs_transfer, accs_trans_sum = accs.copy(), 0
            for i in range(accs_transfer.shape[0]):
                accs_transfer[i, 0:i + 1] = 0
            for j in range(1, accs_transfer.shape[1]):
                accs_trans_sum += np.sum(accs_transfer[:, j]) / j
            accs_transfer = accs_trans_sum / (domain_num - 1)
            accs_avg = np.mean(accs, axis=0)
            print(accs_avg)
            avg = np.mean(accs_avg)
            for i in range(accs.shape[0] - 1):
                accs[i, i + 1:] = 0
            acc_path = os.path.join(r'check_point/MCLDNN_SVD/model/domain_split/First8_domain_num_5/SVD_inc',
                                    'Accuracy.txt')
            # np.save(acc_path, accs)
            FAA_D = accs[-1].sum() / float(domain_num)
            FAA_S = np.sum(accs[-1] * data_len) / np.sum(data_len)
            FF_arr = accs[:-1, :-1] - accs[-1, :-1]
            FF = np.zeros((domain_num - 1,))
            FF_D = 0
            for i in range(FF_arr.shape[1]):
                FF[i] = np.max(FF_arr[:, i])
                FF_D += np.max(FF_arr[:, i])
            FF_path = os.path.join(r'check_point/MCLDNN_SVD/model/domain_split/First8_domain_num_5/SVD_inc',
                                   'Forget.txt')
            # np.save(FF_path, FF_arr)
            FF_S = np.sum(FF * data_len[:-1]) / np.sum(data_len[:-1])
            FF_D = FF_D / float(FF_arr.shape[1])
            # 增量平均准确率
            print('准确率矩阵:', accs)
            print('每次增量平均准确率:', np.sum(accs, axis=1).reshape(-1))
            print(FF_arr)
            print(
                f'{random_domain}:FAA:{"%.2f" % FAA_D}   FF:{"%.2f" % FF_D}    FAA_S:{"%.2f" % FAA_S}   FF_S:{"%.2f" % FF_S}  avg:{"%.2f" % avg}  transfer:{"%.2f" % accs_transfer}')

            new_data = f'{backbone}-{str(seed)}-{random_domain}:FAA:{"%.2f" % FAA_D}   FF:{"%.2f" % FF_D}    FAA_S:{"%.2f" % FAA_S}   FF_S:{"%.2f" % FF_S}'
            filepath = 'methods/output_csv/fb/PID_' + str(dataset) + '_' + str(beta) + backbone + '_cd63Y.csv'

            if not os.path.exists(filepath):
                # 文件不存在，创建一个包含新数据的单列DataFrame
                df = pd.DataFrame([new_data])
                df.to_csv(filepath, header=False, index=False)  # 写入文件，不包含表头和索引
            else:
                # 文件存在，读取现有数据
                current_data = pd.read_csv(filepath, header=None)  # 读取单列数据
                # 将新数据追加到现有数据的末尾
                updated_data = pd.concat([current_data, pd.Series(new_data)], ignore_index=True)
                # 将更新后的数据写回文件
                updated_data.to_csv(filepath, mode='w', header=False, index=False)

            # 释放显存
            del random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, \
                val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders, pre_model, model
            torch.cuda.empty_cache()
            time.sleep(10)



