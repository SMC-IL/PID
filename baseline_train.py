import numpy as np
import time
import torch
import torch.nn as nn
import sys
sys.path.append('/media/xaserver/data2/lzy/XinCL')
from modules.model import MCLDNN, feat_bottleneck, LinearClassifier, SupConMCLDNN, TransNet, LSTMModel, DAE, PETCGDNN, Wir_CNN_BN, PETCGDNN2, MCLDNN_SVD, CLDNN, ICAMC, Wir_CNN, SCF_CNN, SCF_CNN_16
from data.dataset_sample_inc import splitRML2016A, splitDomainRML2016A, loadRML2016A, loadDomainRML2016A_SVD, load_six_data_six_zq, JointDataset, loadDomainRML2018_SVD, load_SCF_SVD, load_six_data
# from data.dataset import Getdata_RML2016A, loadRML2016A
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from utils.scheduler import PolynomialLR
import tqdm
import matplotlib.pyplot as plt
from data.dataset_class_inc import *
from thop import clever_format, profile

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'



def evaluate_model_stats(model, One_flag=False, batch_size=512):
    """
    一次性计算模型的 MACs, 参数量, 推理显存 和 训练显存
    注意：显存测试依赖 GPU 环境
    """
    if not torch.cuda.is_available():
        print("警告: 未检测到 GPU，显存测试将跳过或不准确。建议在 CUDA 环境下运行。")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # 1. 确保模型在正确的设备上
    model = model.to(device)
    
    # 2. 构造正确维度的输入数据 (修复了之前的 Shape Mismatch 问题)
    # input1 形状: [Batch_Size, Channels, Height, Width]
    # input1 = torch.randn(batch_size, 1, 2, 1024).to(device)
    # input2 = torch.randn(batch_size, 1, 1024).to(device)
    # input3 = torch.randn(batch_size, 1, 1024).to(device)
    input1 = torch.randn(batch_size, 1, 16, 16).to(device)
    
    if One_flag:
        inputs = (input1,)
    else:
        inputs = (input1, input2, input3)

    print("="*50)
    print("开始评估模型...")
    print(f"Batch Size: {batch_size}")
    print("="*50)

    # ---------------------------------------------------------
    # 第一部分：计算静态指标 (MACs & Params) 使用 thop
    # ---------------------------------------------------------
    print("[1/3] 正在分析计算量 (MACs) 和参数量...")
    # 置为 eval 模式以保证算力评估不受 dropout 等随机层影响
    model.eval() 
    
    # verbose=False 可以关闭烦人的逐层打印
    macs, params = profile(model, inputs=inputs, verbose=False)
    GFLOPs, all_GFLOPs = macs*2, macs*6  # 1 MAC ≈ 2 FLOPs，训练阶段总计算量约为前向传播的 3 倍
    macs_str, params_str, GFLOPs_str, all_GFLOPs_str = clever_format([macs, params, GFLOPs, all_GFLOPs], "%.3f")

    if device.type == 'cuda':
        # ---------------------------------------------------------
        # 第二部分：计算【推理阶段】峰值显存
        # ---------------------------------------------------------
        print("[2/3] 正在测试纯推理 (Inference) 显存占用...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(*inputs)
            
        inference_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # 转换为 MB

        # ---------------------------------------------------------
        # 第三部分：计算【训练阶段】峰值显存 (包含激活值和梯度)
        # ---------------------------------------------------------
        print("[3/3] 正在测试训练 (Training) 显存占用...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model.train() # 切换回训练模式
        
        # 强制要求计算梯度以模拟真实的训练显存缓存
        input1.requires_grad = True
        
        out = model(*inputs)
        loss = out.sum()
        loss.backward() # 反向传播会分配额外的显存用于梯度
        
        training_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        inference_mem = 0
        training_mem = 0

    # ---------------------------------------------------------
    # 打印最终报告
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🎯 模型综合评估报告")
    print("="*50)
    print(f"总参数量 (Parameters) : {params_str}")
    print(f"前向传播总计算量 (MACs)       : {macs_str} (注意: 1 MAC 约等于 2 FLOPs)")
    print(f"前向传播总计算量 (FLOPs)       : {GFLOPs_str} (注意: 1 MAC 约等于 2 FLOPs)")
    print(f"总计算量 (FLOPs)       : {all_GFLOPs_str} (注意: 1 MAC 约等于 2 FLOPs)")
    if device.type == 'cuda':
        print(f"推理峰值显存 (Batch={batch_size}) : {inference_mem:.2f} MB")
        print(f"训练峰值显存 (Batch={batch_size}) : {training_mem:.2f} MB")
    print("="*50)

#### 参数
start_epoch = 0
training_epoch = 100
classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
# snr_list = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
# snr_index = -1
# """
# === RML2016A 数据读取 ===
# """
start = time.time()
num_class, domain_num = 11, 5
# save_path = r'data/Wir/wir'
# path = r'/media/zxr/DATA1/lzy/XinCL/data/SCF'
# save_path, input_size = r'/media/zxr/DATA1/lzy/XinCL/origin_data/First8_domain_num_49_7_zq', [2, 128]
# # save_path, input_size = r'origin_data/2018_domain_First10_inc4', [2, 1024]
# random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, \
# val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders = loadDomainRML2016A_SVD(save_path, domain_num=5, train_bz=512)

# save_path, input_size = r'origin_data/2018_domain_First10_inc4', [2, 1024]
# random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, \
# val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders = loadDomainRML2018_SVD(save_path, domain_num=5, train_bz=512)
            
save_path = r'data/Wir/wir'
random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, \
val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders = load_six_data_six_zq(save_path, batchsize=1024)
if type(train_dataset) == list:
    train_dataset = JointDataset(train_dataset)
    val_dataset = JointDataset(val_dataset)
    test_dataset = JointDataset(test_dataset)
train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)

# save_path = r'data/SCF'
# random_domain, train_dataset, test_dataset, inc_train_datasets, inc_test_datasets, train_loader, \
#                     test_loader, inc_train_loaders, inc_test_loaders = load_SCF_SVD(save_path, domain_num=5, train_bz=512, len_one=16)
end = time.time()
"""
=== 模型训练 ===
"""
# model = MCLDNN().cuda()
# model = TransNet(25).cuda()
# model = MCLDNN(classes=num_class).cuda()
# model = PETCGDNN2(input_shape=input_size, classes=num_class).cuda()
model = Wir_CNN().cuda()
for name, param in model.named_parameters():
    if 'weight' in name and param.dim() >= 2:
        nn.init.kaiming_normal_(param.data)
    param.requires_grad = True
# model = torch.load('check_point/CLDNN/model_2018/domain_split/First10_inc4/CLDNN_epoch_63_valAcc_0.7150335775335775.pth')

# model_2016A = PETCGDNN().cuda()
# model_2016A = SupConMCLDNN().cuda()
# bottleneck = feat_bottleneck(128, 128).cuda()
# lcf = LinearClassifier(128, 11).cuda()

# # 加载预训练模型
# path_state_dict = r'check_point/MCLDNN/model_2016A/domain_split/Avg_domain_num_5/MCLDNN_epoch_5_valAcc_0.8980681818181818.pth'
# model_2016A = torch.load(path_state_dict)
# print("Load Pretrained Model Successful!")

# pretrained_model = torch.load(r'check_point/MCLDNN/model_2016A/domain_split/First8_domain_num_5/MCLDNN_epoch_2_valAcc_0.8771022727272727.pth')
# pretrained_state_dict = pretrained_model.state_dict()
# model_2016A = MCLDNN_SVD().cuda()
# model_state_dict = model_2016A.state_dict()
# # print(model_state_dict.keys(), pretrained_state_dict.keys())
# new_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
# print(new_state_dict.keys())
# model_2016A.load_state_dict(new_state_dict, strict=False)
# # bottleneck = torch.load("E:\行为识别\SMC\checkpoint_SMC\MCLDNN\Sup_bottleneck.pth")
# # lcf = torch.load("E:\行为识别\SMC\checkpoint_SMC\MCLDNN\Sup_lcf.pth")

# 定义优化器，损失
# optimizer = torch.optim.Adam(list(model_2016A.parameters())+list(bottleneck.parameters())+list(lcf.parameters()), lr=1e-3, weight_decay=1e-5)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, betas=(0.9, 0.99), weight_decay=1e-5)
scheduler = PolynomialLR(optimizer, max_iter=training_epoch, power=0.8)
# scaler = torch.cuda.amp.GradScaler()
# NUM_ACCUMULATION_STEPS = 8
CrossLoss = nn.CrossEntropyLoss()

correct = torch.zeros(1).squeeze().cuda()
correct_ = list(0. for i in range(num_class))
epochs = []
train_losses = []
train_accs = []
val_losses = []
val_accs = []
best_acc = 0
Test = False

if Test:
    accs = []
    # 模型测试
    model = torch.load(r'/media/zxr/DATA1/lzy/XinCL/check_point/MCLDNN/model_2016A/domain_split/4h4l_FSDIL/MCLDNN_epoch_88_valAcc_0.8542045454545455.pth')
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
                outputs = model(imgs, imgs1, imgs2)

                predicts = outputs.argmax(dim=1, keepdim=True)
                all_targets = torch.cat([all_targets, targets])
                all_predicts = torch.cat([all_predicts, predicts], dim=0)

            correct_ = all_predicts.eq(all_targets.view_as(all_predicts)).sum().item()
            accuracy_ = correct_ / float(len(test_D))
            accs.append(accuracy_)
            print(accuracy_)
    print("Test_acc:", np.mean(accs))

else:
    # evaluate_model_stats(model, One_flag=False, batch_size=512)
    for epoch in range(start_epoch+1, training_epoch+1, 1):
        # 模型训练
        model.train()
        with tqdm.tqdm(train_loader, unit="batch") as tepoch: # 🌟 1. 定义进度条
            # for idx, (data, target, snr, _, _) in enumerate(tepoch):# 🌟 2. 设置迭代器")
            for idx, (data, target, _, _, _) in enumerate(tepoch):# 🌟 2. 设置迭代器")
                tepoch.set_description(f"Epoch {epoch}")  # 🌟 3. 设置开头
                data, target = data.cuda().float(), target.cuda().long()    # Data to device
                # 获取IQ序列的单独向量
                data1 = data[:, :, 0, :]
                data2 = data[:, :, 1, :]
                output = model(data)
                # output = model(data, data1, data2)
                # output = model_2016A(data)
                # if idx == 1:
                #     print('--更新前--')
                #     for name, params in model_2016A.named_parameters():
                #         print("name: ", name)
                #         print("grad: ", params.grad)
                losstr = CrossLoss(output, target)                      # Calculate loss
                # loss2 = CrossLoss(xd, data)
                optimizer.zero_grad()
                losstr.backward()
                optimizer.step()
                # if idx == 1:
                #     print('--更新后--')
                #     for name, params in model_2016A.named_parameters():
                #         print("name: ", name)
                #         print("grad: ", params.grad)
                    
                predict_ = output.argmax(dim=1, keepdim=True)
                correct = predict_.eq(target.view_as(predict_)).sum().item() 

                accuracy = correct/data.shape[0]
                tepoch.set_postfix(loss=losstr.item(), accuracy='{:.3f}'.format(accuracy)) # 🌟 4. 设置结尾

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
            for imgs, targets, snr, _, _ in test_loader:
            # for imgs, targets, _ in test_loader:

                imgs = imgs.cuda().float()
                targets = targets.cuda().long()
                imgs1 = imgs[:,:,0,:]
                imgs2 = imgs[:,:,1,:]
                # outputs = lcf(bottleneck(model_2016A.encoder(imgs, imgs1, imgs2)))
                outputs = model(imgs)
                # outputs = model(imgs, imgs1, imgs2)
                #outputs = model_2016A(imgs)

                loss1 = CrossLoss(outputs, targets)
                # loss2 = CrossLoss(xds, imgs)
                loss = loss1
                predicts = outputs.argmax(dim=1, keepdim=True)
                all_targets = torch.cat([all_targets, targets])
                all_predicts = torch.cat([all_predicts, predicts], dim=0)
        
            correct_ = all_predicts.eq(all_targets.view_as(all_predicts)).sum().item()
            accuracy_ = correct_/float(len(test_dataset))
            print("val_acc:", accuracy_)
        
        val_losses.append(loss.item())
        val_accs.append(accuracy_)
            
        if accuracy_ > best_acc:
            best_acc = accuracy_
            torch.save(model, r'/media/zxr/DATA1/lzy/XinCL/check_point/Wir_CNN/six_zq/Wir_CNN_BN_epoch_{}_valAcc_{}.pth'.format(epoch+1, accuracy_))
            # torch.save(model, r'check_point/PETCGDNN2/model_2018/domain_split/First10_inc4/PETCGDNN_epoch_{}_valAcc_{}.pth'.format(epoch+1, accuracy_))
            # torch.save(model,
            #            r'check_point/Wir_CNN/BN_First/Wir_CNN_BN_epoch_{}_valAcc_{}.pth'.format(
            #                epoch + 1, accuracy_))
            # torch.save(bottleneck, '/media/zxr/DATA1/ICL/SIL/baseline/MCLDNN/check_point/Sup_bottleneck.pth')
            # torch.save(lcf, '/media/zxr/DATA1/ICL/SIL/baseline/MCLDNN/check_point/Sup_lcf.pth')
            
    # np.savetxt(r'check_point/CLDNN/model_2018/domain_split/First10_inc4/train_acc.txt', train_accs)
    # np.savetxt(r'check_point/CLDNN/model_2018/domain_split/First10_inc4/train_loss.txt', train_losses)
    # np.savetxt(r'check_point/CLDNN/model_2018/domain_split/First10_inc4/val_acc.txt', val_accs)
    # np.savetxt(r'check_point/CLDNN/model_2018/domain_split/First10_inc4/val_loss.txt', val_losses)



    # 模型测试
    # model_2016A.eval()

    # with torch.no_grad():
    #     for imgs, targets in test_dataloader:
    #                 imgs = imgs.cuda().float()
    #                 targets = targets.cuda().long()

    #                 imgs1 = imgs[:,:,0,:]
    #                 imgs2 = imgs[:,:,1,:]
    #                 outputs = model_2016A(imgs, imgs1, imgs2)

    #                 predicts = outputs.argmax(dim=1, keepdim=True)
    #                 correct_ = predicts.eq(targets.view_as(predicts)).sum().item()
    #                 accuracy_ = correct_/len(imgs)
    #                 print("Test_acc:", accuracy_)
                    
    #                 targets_ = targets.cpu().numpy()
    #                 predicts_ = predicts.cpu().numpy()
                    
    #                 # 计算全部的混淆矩阵
    #                 confnorm= confusion_matrix(targets_, predicts_)
    #                 plot_confusion_matrix(confnorm, labels=classes,save_filename='./SMC/mclstm_total_confusion.png')
                    
    # plt.subplot(121)
    # plt.plot(epochs, train_losses, color = 'b')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # # 绘制 acc 曲线
    # plt.subplot(122)
    # plt.plot(epochs, train_accs, color = 'r')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.savefig(os.path.join(r'check_point/MCLDNN/image', '73domain_origin.png'))
    # plt.show()