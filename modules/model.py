import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math
from fft_conv_pytorch import FFTConv2d
from einops import rearrange, repeat
from modules.hyp_layers import HNNLayer

class RFF(nn.Module):
    def __init__(self, d, D):
        super(RFF, self).__init__()
        self.d = d
        self.D = D
        self.omega, self.b = self.generate_weights()

    def generate_weights(self):
        omega = torch.randn(self.d, self.D)
        b = torch.rand(self.D) * 2 * math.pi
        return omega, b

    def forward(self, x):
        x_omega = x @ self.omega
        x_omega_b = x_omega + self.b
        return torch.cat([torch.cos(x_omega_b), torch.sin(x_omega_b)], dim=-1)

class CNN_base(nn.Module):
    def __init__(self):
        super(CNN_base, self).__init__()
        self.convblock = nn.Sequential(
            nn.ZeroPad2d(padding=2),
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.ZeroPad2d(padding=2),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(2,3), padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Flatten(),
        )

        self.AvgPool1d = nn.AdaptiveAvgPool2d(32)
        self.fc1 = nn.Linear(in_features=33120, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=11)

    def forward(self, x):
        x1 = self.convblock(x)
        # print("after convblock(x1)", x1.shape)
        # x1 = self.AvgPool1d(x1)
        # print("after AvgPool1d(x1)", x1.shape)
        out = x1
        Batch, Length = out.size()
        # [64, 19008]
        # print(out.shape)
        out = out.view(Batch, -1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, 0.1, self.training)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out

class CNN_RFF(nn.Module):
    def __init__(self):
        super(CNN_RFF, self).__init__()
        self.zeropad = nn.ZeroPad2d(padding=2)
        self.bn1 = nn.BatchNorm2d(1)
        self.RFFconv1 = FFTConv2d(in_channels=1, out_channels=64, kernel_size=(1,3), padding=1)
        self.Resnetblock1 = Resnet_block_v1(in_channels=64, out_channels=64)
        self.bn2 = nn.BatchNorm2d(64)
        self.reLu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.RFFconv2 = FFTConv2d(in_channels=64, out_channels=16, kernel_size=(2,3), padding=2)
        self.Resnetblock2 = Resnet_block_v1(in_channels=16, out_channels=16)
        self.bn3 = nn.BatchNorm2d(16)
        self.Flatten = nn.Flatten()

        self.AvgPool2d = nn.AdaptiveAvgPool2d(15)
        self.fc1 = nn.Linear(in_features=3600, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=11)
        # self.fc3 = nn.Linear(in_features=512, out_features=11)

    def forward(self, x):
        # x.shape [64, 1, 2, 128]
        x1 = self.zeropad(x)
        x1 = self.bn1(x1)
        x1 = self.RFFconv1(x1)
        x1 = self.Resnetblock1(x1)
        x1 = self.bn2(x1)
        x1 = self.reLu(x1)
        x1 = self.dropout(x1)
        
        x1 = self.zeropad(x1)
        x1 = self.RFFconv2(x1)
        #print("after RFFConv2d(x1)", x1.shape)
        x1 = self.Resnetblock2(x1)
        #print("after Res(x1)", x1.shape)
        x1 = self.bn3(x1)
        x1 = self.reLu(x1)
        x1 = self.dropout(x1)
        x1 = self.AvgPool2d(x1)
        # print("after AvgPool2d(x1)", x1.shape)
        x1 = self.Flatten(x1)
        # print("after Flatten(x1)", x1.shape)
        out = x1
        Batch, Length = out.size()

        # 用于对比学习，只输出features
        out = out.view(Batch, -1)
        # out = F.relu(self.fc1(out))
        # out = F.dropout(out, 0.1, self.training)
        # out = self.fc2(out)
        # out = self.fc3(out)
        return out
    
class SupConRffCNN(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=3600, feat_dim=128):
        super(SupConRffCNN, self).__init__()
        self.encoder = CNN_RFF()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim))
        
    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        # print("features' shape:", feat.shape)
        return feat

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, feat_dim, num_classes=11):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

# 残差模块
class Resnet_block_v1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding=1)
        self.bn0 = nn.BatchNorm2d(num_features = out_channels)
        self.bn1 = nn.BatchNorm2d(num_features = out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding=1)
        self.bn2 = nn.BatchNorm2d(num_features = out_channels)
        self.reLu = nn.ReLU()
        self.con1x = FFTConv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1)
    def forward(self, x):
        x0 = self.con1x(x)
        x0 = self.reLu(self.bn0(x0))
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.reLu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        return x0 + x1 + x

# GRU
class GRUModel_FE(nn.Module):
    def __init__(self, classes = 11):
        super(GRUModel_FE, self).__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=128, num_layers=2, batch_first=True)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]

        return x
     
class GRUModel(nn.Module):
    def __init__(self, classes=11):
        super(GRUModel, self).__init__()
        self.encoder = GRUModel_FE()
        self.fc = nn.Linear(128, classes)
        
    def forward(self, input):
        x = self.encoder(input)
        x = self.fc(x)

        return x
    
# LSTM
class LSTMModel_FE(nn.Module):
    def __init__(self, classes = 11):
        super(LSTMModel_FE, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=128, num_layers=2, batch_first=True)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]

        return x
    
class LSTMModel(nn.Module):
    def __init__(self, classes=11):
        super(LSTMModel, self).__init__()
        self.encoder = LSTMModel_FE()
        self.fc = nn.Linear(128, classes)
        
    def forward(self, input):
        x = self.encoder(input)
        x = self.fc(x)

        return x
    
# IC-AMCNET

    
# DAE模型
class DAE_FE(nn.Module):
    def __init__(self, classes = 11):
        super(DAE_FE, self).__init__()
        self.dr = 0.3
        self.lstm1 = nn.LSTM(input_size=2, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(32, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, classes)
        self.decoder = nn.Linear(32, 2)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = F.dropout(x, self.dr)
        x, (s1, c1) = self.lstm2(x)
        s1 = s1.squeeze(0)

        return x, s1
    
class DAE(nn.Module):
    def __init__(self, classes=11):
        super(DAE, self).__init__()
        self.dr = 0.3
        self.encoder = DAE_FE()
        self.fc1 = nn.Linear(32, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, classes)
        self.decoder = nn.Linear(32, 2)


    def forward(self, input1):
        x, s1 = self.encoder(input1)
        xc = F.relu(self.fc1(s1))
        xc = self.bn1(xc)
        xc = F.dropout(xc, self.dr)
        xc = F.relu(self.fc2(xc))
        xc = self.bn2(xc)
        xc = F.dropout(xc, self.dr)
        label = self.fc3(xc)
        
        xd = self.decoder(x)
        xd = xd.permute(0, 2, 1)
        xd = xd.unsqueeze(1)

        return label, xd



# MCLDNN模型
class SVD_Conv2d(nn.Module):
    """Kernel Number first SVD Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, bias, domain_num=1, origin_weight=None, origin_bias=None,
                 padding_mode='zeros', device=None, dtype=torch.float,
                 rank=2):
        super(SVD_Conv2d, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.domain_num = domain_num
        self.conv_U = nn.Conv2d(rank, out_channels, (1, 1), (1, 1), 0, (1, 1), 1, bias)
        self.conv_V = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, dilation, groups, False)
        Ss = nn.Parameter(torch.empty((1, rank, 1, 1), **factory_kwargs), requires_grad=True)
        name_S = 'SS_0'
        if origin_weight is not None:
            weight = origin_weight.reshape(origin_weight.shape[0], -1)
            U, S, V = torch.svd(weight)
            V = V.T
            U = U.reshape(origin_weight.shape[0], S.shape[0], 1, 1)
            V = V.reshape(S.shape[0], origin_weight.shape[1], origin_weight.shape[2], origin_weight.shape[3])
            S = S.reshape(1, S.shape[0], 1, 1)
            if domain_num == 1:
                self.conv_U.weight.data.copy_(U)
                self.conv_V.weight.data.copy_(V)
                Ss.data.copy_(S)
            if origin_bias is not None and bias == True:
                self.conv_U.bias.data.copy_(origin_bias)
        self.register_parameter(name_S, Ss)
        self.S_list = [self.SS_0]
        for i in range(1, domain_num):
            name_S = 'SS_' + str(i)
            Ss = nn.Parameter(torch.zeros((1, rank, 1, 1), **factory_kwargs), requires_grad=True)
            self.register_parameter(name_S, Ss)
            self.S_list.append(Ss)
        if domain_num > 1:
            self.flag = True
        else:
            self.flag = False

    def forward(self, x, pos=-1):
        # pos=-1时是训练过程中，pos为其它值是在测试过程中使用的
        if self.flag:
            self.S_origin()
            self.flag = False
        x = self.conv_V(x)
        # S = getattr(self, 'S_'+str(self.domain_num-1))
        # x = x.mul(S)
        x = x.mul(self.S_list[pos])
        output = self.conv_U(x)
        return output

    def S_origin(self):
        # S = getattr(self, 'S_'+str(self.domain_num-1))
        S = self.S_list[-1]
        for s in self.S_list[:-1]:
            S.data += s.data
        S.data /= len(self.S_list[:-1])
        self.S_list[-1].data.copy_(S.data)

class SVD_Conv2d_bias(nn.Module):
    """Kernel Number first SVD Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, bias=True, domain_num=1, origin_weight=None, origin_bias=None,
                 padding_mode='zeros', device=None, dtype=torch.float,
                 rank=2):
        super(SVD_Conv2d_bias, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.domain_num = domain_num
        self.conv_U = nn.Conv2d(rank, out_channels, (1, 1), (1, 1), 0, (1, 1), 1, False)
        self.conv_V = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, dilation, groups, False)
        Ss = nn.Parameter(torch.empty((1, rank, 1, 1), **factory_kwargs), requires_grad=True)
        name_S = 'SS_0'
        name_B = 'bias_0'
        B = nn.Parameter(torch.empty((1, out_channels, 1, 1), **factory_kwargs), requires_grad=True)
        if origin_weight is not None:
            weight = origin_weight.reshape(origin_weight.shape[0], -1)
            U, S, V = torch.svd(weight)
            V = V.T
            U = U.reshape(origin_weight.shape[0], S.shape[0], 1, 1)
            V = V.reshape(S.shape[0], origin_weight.shape[1], origin_weight.shape[2], origin_weight.shape[3])
            S = S.reshape(1, S.shape[0], 1, 1)
            if domain_num == 1:
                self.conv_U.weight.data.copy_(U)
                self.conv_V.weight.data.copy_(V)
                Ss.data.copy_(S)
        if origin_bias is not None:
            origin_bias = origin_bias.reshape(1, -1, 1, 1)
            B.data.copy_(origin_bias)

        self.register_parameter(name_B, B)
        self.bias_list = [self.bias_0]
        self.register_parameter(name_S, Ss)
        self.S_list = [self.SS_0]
        for i in range(1, domain_num):
            name_S = 'SS_' + str(i)
            name_B = 'bias_' + str(i)
            Ss = nn.Parameter(torch.zeros((1, rank, 1, 1), **factory_kwargs), requires_grad=True)
            B = nn.Parameter(torch.zeros((1, out_channels, 1, 1), **factory_kwargs), requires_grad=True)
            self.register_parameter(name_S, Ss)
            self.register_parameter(name_B, B)
            self.S_list.append(Ss)
            self.bias_list.append(B)
        if domain_num > 1:
            self.flag = True
        else:
            self.flag = False

    def forward(self, x, pos=-1):
        # pos=-1时是训练过程中，pos为其它值是在测试过程中使用的
        if self.flag:
            self.origin_SB()
            self.flag = False
        x = self.conv_V(x)
        # S = getattr(self, 'S_'+str(self.domain_num-1))
        # x = x.mul(S)
        x = x.mul(self.S_list[pos])
        output = self.conv_U(x)
        output += self.bias_list[pos]
        return output

    def origin_SB(self):
        # S = getattr(self, 'S_'+str(self.domain_num-1))
        S = self.S_list[-1]
        B = self.bias_list[-1]
        for s in self.S_list[:-1]:
            S.data += s.data
        S.data /= len(self.S_list[:-1])
        self.S_list[-1].data.copy_(S.data)
        for b in self.bias_list[:-1]:
            B.data += b.data
        B.data /= len(self.bias_list[:-1])
        self.bias_list[-1].data.copy_(B.data)


class SVD_Linear_origin(nn.Linear):
    def __init__(self, in_features, out_features, domain_num=3, num_classes=11, name='trainable_S', fc_weight=None, fc_bias=None):
        super(SVD_Linear_origin, self).__init__(out_features, in_features)
        if fc_weight is not None:
            self.weight.data = fc_weight
        if fc_bias is not None:
            self.bias.data = fc_bias
        bias = nn.Parameter(torch.randn((domain_num, self.bias.shape[-1])), requires_grad=True)
        if domain_num > 1:
            bias.data[:-1, :] = self.bias.data
            self.bias = bias
        self.U, S, self.V = torch.svd(self.weight.T.detach())
        self.S = nn.Parameter(torch.randn((domain_num, S.shape[0])), requires_grad=True)  # .cuda()
        # 对矩阵进行 QR 分解，并取转置得到行向量之间正交的矩阵，可以探索一下效果如何，上面那行是直接加载进去了。下面注释这两行是正交分解Q和R，这样可以让这两组参数隔离的比较开
        # q, r = torch.qr(self.S.t())
        # self.S.data.copy_(q.t().cuda())
        self.S.data.copy_(S)
        self.register_parameter(name, self.S)
        # self.bias = self.bias
        # if domain_num > 1:
        #     self.bias = torch.cat(self.bias, self.bias.resize((1, -1)))
        #     print(self.bias.shape)
        #     self.register_parameter('bias_'+str(domain_num-1), self.bias_list[-1])
        # self.bn1 = nn.LayerNorm(out_features*2)
        # self.bn2 = nn.LayerNorm(out_features*2)
        # self.bn = nn.LayerNorm(out_features)
        # 可以探索层的多少
        # self.bn1 = nn.BatchNorm1d(out_features*2, affine=True)
        # self.bn2 = nn.BatchNorm1d(out_features*2, affine=True)
        # self.bn3 = nn.BatchNorm1d(out_features, affine=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.5)
        # self.bottleneck1 = nn.Linear(out_features, out_features*2)
        # self.bottleneck2 = nn.Linear(out_features*2, out_features*2)
        # self.bottleneck3 = nn.Linear(out_features*2, out_features)

    def forward(self, x, domain_predict):
        '''
        x = self.bottleneck1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.bottleneck2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.bottleneck3(x)
        x = self.bn3(x)
        '''
        # x = self.bn(x)
        # print('bias:', self.bias.shape)
        if len(self.bias.shape) > 1:
            # print(x.shape)
            # print(self.bias.shape)
            # print(torch.mm(torch.mm(x, self.U.cuda()).mul(self.S[domain_predict.argmax(dim=1), :]), self.V.t().cuda()).shape)
            Y = torch.mm(torch.mm(x, self.U.cuda()).mul(self.S[domain_predict.argmax(dim=1), :]), self.V.t().cuda()) + self.bias[domain_predict.argmax(dim=1), :]
        else:
            Y = torch.mm(torch.mm(x, self.U.cuda()).mul(self.S[domain_predict.argmax(dim=1), :]), self.V.t().cuda()) + self.bias
        # print(Y)
        # Y = F.linear(X, self.U.T.cuda())
        # Y = Y.mul(self.S[domain_predict.argmax(dim=1), :].cuda())
        # Y = F.linear(Y, self.V.cuda()) + self.bias.cuda()
        return Y, self.S

class SVD_Linear(nn.Linear):
    def __init__(self, in_features, out_features, domain_num=3, num_classes=11, name='trainable_S', fc_weight=None, fc_bias=None):
        super(SVD_Linear, self).__init__(out_features, in_features)
        if fc_weight is not None:
            self.weight.data = fc_weight
        if fc_bias is not None:
            self.bias.data = fc_bias
        bias = nn.Parameter(torch.randn((domain_num, self.bias.shape[-1])), requires_grad=True)
        if domain_num > 1:
            bias.data[:-1, :] = self.bias.data
            self.bias = bias
        self.U, S, self.V = torch.svd(self.weight.T.detach())
        self.S = nn.Parameter(torch.randn((domain_num, S.shape[0])), requires_grad=True)  # .cuda()
        self.S.data.copy_(S)
        self.register_parameter(name, self.S)

    def forward(self, x, domain_predict):
        
        if len(self.bias.shape) > 1:
            Y = torch.mm(torch.mm(x, self.U.cuda()).mul(self.S[domain_predict.argmax(dim=1), :]), self.V.t().cuda()) + self.bias[domain_predict.argmax(dim=1), :]
        else:
            Y = torch.mm(torch.mm(x, self.U.cuda()).mul(self.S[domain_predict.argmax(dim=1), :]), self.V.t().cuda()) + self.bias
        
        return Y, self.S

class SVD_Linear_new(nn.Linear):
    def __init__(self, in_features, out_features,  fc_weight, fc_bias, domain_num=3, num_classes=11, s_num=1, name='trainable_S', device=None, bias=True, dtype=torch.float):
        super(SVD_Linear_new, self).__init__(out_features, in_features)
        factory_kwargs = {'device': device, 'dtype': dtype}
        del self.bias
        self.input = None
        self.domain_num = domain_num
        self.bias_list, self.S_list = [], []
        name_S, name_bias = 'SS_0', 'bias_0'
        self.weight.data = fc_weight
        self.U, S_0, self.V = torch.svd(self.weight.T.detach())
        bias_0 = nn.Parameter(torch.empty((fc_bias.shape[-1],)), requires_grad=True)
        bias_0.data.copy_(fc_bias.data)
        S = nn.Parameter(torch.randn((S_0.shape[0],)), requires_grad=True)
        S.data.copy_(S_0)
        self.register_parameter(name_S, S)
        self.register_parameter(name_bias, bias_0)
        self.bias_list.append(bias_0)
        self.S_list.append(S)
        for i in range(1, domain_num):
            name_S, name_bias = 'SS_'+str(i), 'bias_'+str(i)
            S = nn.Parameter(torch.zeros((S_0.shape[0],)), requires_grad=True)
            b = nn.Parameter(torch.zeros((fc_bias.shape[-1],)), requires_grad=True)
            self.register_parameter(name_S, S)
            self.register_parameter(name_bias, b)
            self.bias_list.append(b)
            self.S_list.append(S)
        if domain_num > 1:
            self.flag = True
        else:
            self.flag = False

    def forward(self, x, pos=-1):
        self.input = x
        if self.flag:
            self.Origin()
            self.flag = False
        Y = (torch.mm(torch.mm(x, self.U.cuda()).mul(self.S_list[pos]), self.V.t().cuda())
             + self.bias_list[pos])
        return Y

    def Origin(self):
        S = self.S_list[-1]
        B = self.bias_list[-1]
        for s in self.S_list[:-1]:
            S.data += s.data
        for b in self.bias_list[:-1]:
            B.data += b.data
        S.data /= len(self.S_list[:-1])
        B.data /= len(self.bias_list[:-1])
        self.S_list[-1].data.copy_(S.data)
        self.bias_list[-1].data.copy_(B.data)

class MCLDNN_FE_SVD(nn.Module):
    def __init__(self, conv_2d_dict, conv_2d_bias_dict, domain_num=1, classes=11):
        super(MCLDNN_FE_SVD, self).__init__()
        # self.conv1_1 = nn.Conv2d(1, 50, kernel_size=(2, 8))
        self.convS_1 = SVD_Conv2d(1, 50, kernel_size=(2, 8), stride=1, padding=0, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[0] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[0] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=16)
        self.pad1_1 = nn.ZeroPad2d(padding=(0, 7, 1, 0))
        self.conv1_2 = nn.Conv1d(1, 50, kernel_size=8)
        self.conv1_3 = nn.Conv1d(1, 50, kernel_size=8)
        # self.conv2 = nn.Conv2d(50, 50, kernel_size=(1, 8))
        self.convS_2 = SVD_Conv2d(50, 50, kernel_size=(1, 8), stride=1, padding=0, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[1] if domain_num==1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[1] if domain_num==1 else None,
                                   domain_num=domain_num,rank=50)
        self.pad1_2 = nn.ZeroPad2d(padding=(0, 7, 0, 0))
        # self.conv4 = nn.Conv2d(100, 100, kernel_size=(2, 5), padding=0)
        self.convS_4 = SVD_Conv2d(100, 100, kernel_size=(2, 5), stride=1, padding=0, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[2] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[2] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=100)
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

    def forward(self, input1, input2, input3, pos=-1):
        input1 = self.pad1_1(input1)
        x1 = F.relu(self.convS_1(input1, pos))
        x2 = F.pad(input2, (0, 7))
        x3 = F.pad(input3, (0, 7))
        x2 = F.relu(self.conv1_2(x2))
        x3 = F.relu(self.conv1_3(x3))
        x2 = x2.unsqueeze(2)
        x3 = x3.unsqueeze(2)
        x = torch.cat([x2, x3], dim=2)
        x = self.pad1_2(x)
        x = F.relu(self.convS_2(x, pos))
        x = torch.cat([x1, x], dim=1)
        x = F.relu(self.convS_4(x, pos))
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]

        return x

class MCLDNN(nn.Module):
    def __init__(self, classes=11):
        super(MCLDNN, self).__init__()
        self.dr = 0.5
        self.encoder = MCLDNN_FE()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, classes)

    def forward(self, input1, input2, input3):
        x = self.encoder(input1, input2, input3)
        x = F.selu(self.fc1(x))
        # x = F.dropout(x, self.dr)
        x = F.selu(self.fc2(x))
        # x = F.dropout(x, self.dr)
        # res = x
        self.input = x
        x = self.fc3(x)
        return x

class Classifier_MCLDNN(nn.Module):
    def __init__(self):
        super(Classifier_MCLDNN, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        # x = F.dropout(x, self.dr)
        x = self.fc2(x)
        return x

# 创建模型
class Classifier_MCLDNN_tanh(nn.Module):
    def __init__(self, manifold, c=1.0, dropout=0.0, act=True, use_bias=True):
        super(Classifier_MCLDNN_tanh, self).__init__()
        self.manifold = manifold
        self.c = c
        self.h1 = HNNLayer(manifold, 128, 64, c, dropout=dropout, act=act, use_bias=use_bias)
        self.h2 = HNNLayer(manifold, 64, 32, c, dropout=dropout, act=act, use_bias=use_bias)

    def forward(self, x):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        x = self.h1(x_hyp)
        x = self.h2(x)
        return x


class MCLDNN_SVD_Conv(MCLDNN):
    def __init__(self, conv_2d_dict, conv_2d_bias_dict, fc_dict, domain_num=1, classes=11):
        super(MCLDNN_SVD_Conv, self).__init__()
        self.dr = 0.5
        self.encoder = MCLDNN_FE_SVD(conv_2d_dict=conv_2d_dict, conv_2d_bias_dict=conv_2d_bias_dict, domain_num=domain_num, classes=classes)
        self.fc1 = SVD_Linear_new(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=domain_num,
                              fc_weight=fc_dict['fc1.weight'], fc_bias=fc_dict['fc1.bias'])
        self.fc2 = SVD_Linear_new(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=domain_num,
                              fc_weight=fc_dict['fc2.weight'], fc_bias=fc_dict['fc2.bias'])
        self.fc3 = SVD_Linear_new(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=domain_num,
                              fc_weight=fc_dict['fc3.weight'], fc_bias=fc_dict['fc3.bias'])

    def forward(self, input1, input2, input3, pos=-1):
        x = self.encoder(input1, input2, input3, pos=pos)
        x = F.selu(self.fc1(x, pos))
        # x = F.dropout(x, self.dr)
        x = F.selu(self.fc2(x, pos))
        # x = F.dropout(x, self.dr)
        x = self.fc3(x, pos)
        return x

class MCLDNN_FE(nn.Module):
    def __init__(self, classes=11):
        super(MCLDNN_FE, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 50, kernel_size=(2, 8))
        self.pad1_1 = nn.ZeroPad2d(padding=(0, 7, 1, 0))
        self.conv1_2 = nn.Conv1d(1, 50, kernel_size=8)
        self.conv1_3 = nn.Conv1d(1, 50, kernel_size=8)
        self.conv2 = nn.Conv2d(50, 50, kernel_size=(1, 8
                                                    ))
        self.pad1_2 = nn.ZeroPad2d(padding=(0, 7, 0, 0))
        self.conv4 = nn.Conv2d(100, 100, kernel_size=(2, 5), padding=0)

        self.lstm1 = nn.LSTM(input_size=100, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

    def forward(self, input1, input2, input3):
        input1 = self.pad1_1(input1)
        x1 = F.relu(self.conv1_1(input1))
        x2 = F.pad(input2, (0, 7))
        x3 = F.pad(input3, (0, 7))
        x2 = F.relu(self.conv1_2(x2))
        x3 = F.relu(self.conv1_3(x3))
        x2 = x2.unsqueeze(2)
        x3 = x3.unsqueeze(2)
        x = torch.cat([x2, x3], dim=2)
        x = self.pad1_2(x)
        x = F.relu(self.conv2(x))
        x = torch.cat([x1, x], dim=1)
        x = F.relu(self.conv4(x))

        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]

        return x


class Classifier(nn.Module):
    def __init__(self, classes=11):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, classes)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        # x = F.dropout(x, self.dr)
        x = F.selu(self.fc2(x))
        # x = F.dropout(x, self.dr)
        x = self.fc3(x)
        return x

class Classifier_Wir(nn.Module):
    def __init__(self):
        super(Classifier_Wir, self).__init__()
        self.fc1 = nn.Linear(8176, 1024)
        self.fc2 = nn.Linear(1024, 256)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        # x = F.dropout(x, self.dr)
        x = self.fc2(x)
        return x

class Classifier_W(nn.Module):
    def __init__(self):
        super(Classifier_W, self).__init__()
        self.fc1 = nn.Linear(8176, 25)
        self.fc2 = nn.Linear(25, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, self.dr)
        x = self.fc2(x)
        return x

class Wir_CNN_FE(nn.Module):
    def __init__(self):
        super(Wir_CNN_FE, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 2))
        # 第二层卷积
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 3))
        # 第三层卷积
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2))


    def forward(self, x):
        # 应用第一层卷积和激活函数
        x = F.relu(self.conv1(x))
        # 应用第一层的Dropout
        # x = F.dropout(x, p=0.2)

        # 应用第二层卷积和激活函数
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(1, 2))
        # 应用第二层的Dropout
        # x = F.dropout(x, p=0.2)

        # 应用第三层卷积和激活函数
        x = F.relu(self.conv3(x))
        # 应用Max pooling
        x = F.max_pool2d(x, kernel_size=(1, 4))
        # 展平特征图，准备输入到全连接层
        x = x.view(x.size(0), -1)
        
        return x

class Wir_CNN(nn.Module):
    def __init__(self):
        super(Wir_CNN, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 2))
        # 第二层卷积
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 3))
        # 第三层卷积
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2))

        # 全连接层
        self.fc1 = nn.Linear(8176, 25)
        self.fc2 = nn.Linear(25, 3)

    def forward(self, x):
        # 应用第一层卷积和激活函数
        x = F.relu(self.conv1(x))
        # 应用第一层的Dropout
        # x = F.dropout(x, p=0.2)

        # 应用第二层卷积和激活函数
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(1, 2))
        # 应用第二层的Dropout
        # x = F.dropout(x, p=0.2)

        # 应用第三层卷积和激活函数
        x = F.relu(self.conv3(x))
        # 应用Max pooling
        x = F.max_pool2d(x, kernel_size=(1, 4))
        # 展平特征图，准备输入到全连接层
        x = x.view(x.size(0), -1)
        # 应用第一层全连接和激活函数
        self.input = x
        x = F.relu(self.fc1(x))
        # 应用第二层全连接和Softmax激活函数
        x = self.fc2(x)
        return x


class Wir_CNN_BN(nn.Module):
    def __init__(self):
        super(Wir_CNN_BN, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 2))
        # 第二层卷积
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 3))
        # 第三层卷积
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2))
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(16)
        # 全连接层
        self.fc1 = nn.Linear(8176, 25)
        self.fc2 = nn.Linear(25, 3)

    def forward(self, x):
        # 应用第一层卷积和激活函数
        x = F.relu(self.bn1(self.conv1(x)))
        # 应用第一层的Dropout
        # x = F.dropout(x, p=0.2)

        # 应用第二层卷积和激活函数
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(1, 2))
        # 应用第二层的Dropout
        # x = F.dropout(x, p=0.2)

        # 应用第三层卷积和激活函数
        x = F.relu(self.bn3(self.conv3(x)))
        # 应用Max pooling
        x = F.max_pool2d(x, kernel_size=(1, 4))
        # 展平特征图，准备输入到全连接层
        x = x.view(x.size(0), -1)
        # 应用第一层全连接和激活函数
        x = F.relu(self.fc1(x))
        # 应用第二层全连接和Softmax激活函数
        x = self.fc2(x)
        return x

class Wir_CNN_Conv(Wir_CNN):
    def __init__(self, conv_2d_dict, conv_2d_bias_dict, fc_dict, domain_num=1, classes=11):
        super(Wir_CNN_Conv, self).__init__()
        # 第一层卷积
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 2))
        self.convS_1 = SVD_Conv2d(1, 64, kernel_size=(1, 2), stride=1, padding=0, bias=True, dilation=1, groups=1,
                   origin_weight=list(conv_2d_dict.values())[0] if domain_num == 1 else None,
                   origin_bias=list(conv_2d_bias_dict.values())[0] if domain_num == 1 else None,
                   domain_num=domain_num, rank=2)
        # 第二层卷积
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 3))
        self.convS_2 = SVD_Conv2d(64, 32, kernel_size=(1, 3), stride=1, padding=0, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[1] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[1] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=32)
        # 第三层卷积
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2))
        self.convS_3 = SVD_Conv2d(32, 16, kernel_size=(2, 2), stride=1, padding=0, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[2] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[2] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=16)
        # 全连接层
        self.fc1 = SVD_Linear_new(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=domain_num,
                       fc_weight=fc_dict['fc1.weight'], fc_bias=fc_dict['fc1.bias'])
        self.fc2 = SVD_Linear_new(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=domain_num,
                       fc_weight=fc_dict['fc2.weight'], fc_bias=fc_dict['fc2.bias'])

    def forward(self, x, pos=-1):
        # 应用第一层卷积和激活函数
        x = F.relu(self.convS_1(x, pos))
        # 应用第一层的Dropout
        # x = F.dropout(x, p=0.2)

        # 应用第二层卷积和激活函数
        x = F.relu(self.convS_2(x, pos))
        x = F.max_pool2d(x, kernel_size=(1, 2))
        # 应用第二层的Dropout
        # x = F.dropout(x, p=0.2)

        # 应用第三层卷积和激活函数
        x = F.relu(self.convS_3(x, pos))
        # 应用Max pooling
        x = F.max_pool2d(x, kernel_size=(1, 4))
        # 展平特征图，准备输入到全连接层
        x = x.view(x.size(0), -1)
        # 应用第一层全连接和激活函数
        self.input = x
        x = F.relu(self.fc1(x, pos))
        # 应用第二层全连接和Softmax激活函数
        x = self.fc2(x, pos)
        return x


class Wir_CNN_Conv_2(Wir_CNN):
    def __init__(self, conv_2d_dict, conv_2d_bias_dict, fc_dict, domain_num=1, classes=11):
        super(Wir_CNN_Conv, self).__init__()
        # 第一层卷积
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 2))
        self.convS_1 = SVD_Conv2d(1, 64, kernel_size=(1, 2), stride=1, padding=0, bias=True, dilation=1, groups=1,
                   origin_weight=list(conv_2d_dict.values())[0] if domain_num == 1 else None,
                   origin_bias=list(conv_2d_bias_dict.values())[0] if domain_num == 1 else None,
                   domain_num=domain_num, rank=2)
        # 第二层卷积
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 3))
        self.convS_2 = SVD_Conv2d(64, 32, kernel_size=(1, 3), stride=1, padding=0, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[1] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[1] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=32)
        # 第三层卷积
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2))
        self.convS_3 = SVD_Conv2d(32, 16, kernel_size=(2, 2), stride=1, padding=0, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[2] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[2] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=16)
        # 全连接层
        self.fc1 = SVD_Linear_new(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=domain_num,
                       fc_weight=fc_dict['fc1.weight'], fc_bias=fc_dict['fc1.bias'])
        self.fc2 = SVD_Linear_new(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=domain_num,
                       fc_weight=fc_dict['fc2.weight'], fc_bias=fc_dict['fc2.bias'])
    def forward(self, x, pos=-1):
        # 应用第一层卷积和激活函数
        x = F.relu(self.convS_1(x, pos))
        # 应用第一层的Dropout
        # x = F.dropout(x, p=0.2)

        # 应用第二层卷积和激活函数
        x = F.relu(self.convS_2(x, pos))
        x = F.max_pool2d(x, kernel_size=(1, 2))
        # 应用第二层的Dropout
        # x = F.dropout(x, p=0.2)

        # 应用第三层卷积和激活函数
        x = F.relu(self.convS_3(x, pos))
        # 应用Max pooling
        x = F.max_pool2d(x, kernel_size=(1, 4))
        # 展平特征图，准备输入到全连接层
        x = x.view(x.size(0), -1)
        # 应用第一层全连接和激活函数
        x = F.relu(self.fc1(x, pos))
        # 应用第二层全连接和Softmax激活函数
        x = self.fc2(x, pos)
        return x




class SCF_CNN(nn.Module):
    def __init__(self):
        super(SCF_CNN, self).__init__()

        # 定义卷积层和激活层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.3)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 1), stride=1, padding=(1, 0))
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.3)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 1), stride=1, padding=(1, 0))
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.3)

        # 定义池化层
        self.max_pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, padding=0)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, padding=0)
        self.max_pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, padding=0)

        # 定义全连接层
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=1025 * 2 * 64, out_features=256)
        self.dense2 = nn.Linear(in_features=256, out_features=4)

    def forward(self, x):
        # 通过卷积层和激活层
        x = self.conv1(x)
        print(x.shape)
        x = self.leaky_relu1(x)
        x = self.max_pool1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.leaky_relu2(x)
        x = self.max_pool2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.leaky_relu3(x)
        x = self.max_pool3(x)
        print(x.shape)
        # 展平
        x = self.flatten(x)
        print(x.shape)
        # 通过全连接层
        x = self.dense1(x)
        print(x.shape)
        x = F.relu(x)
        x = self.dense2(x)
        print(x.shape)

        return x

class Classifier_SCF(nn.Module):
    def __init__(self):
        super(Classifier_SCF, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        # x = F.dropout(x, self.dr)
        x = self.fc2(x)
        return x


class Classifier_S(nn.Module):
    def __init__(self):
        super(Classifier_S, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, self.dr)
        x = self.fc2(x)
        return x

class SCF_CNN_16_FE(nn.Module):
    def __init__(self):
        super(SCF_CNN_16_FE, self).__init__()

        # 定义卷积层和激活层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.3)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.3)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.3)

        # 定义池化层
        self.max_pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, padding=0)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, padding=0)
        self.max_pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, padding=0)

        # 定义全连接层
        self.flatten = nn.Flatten()

    def forward(self, x):
        # 通过卷积层和激活层
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.max_pool3(x)
        # 展平
        x = self.flatten(x)

        return x

class SCF_CNN_16(nn.Module):
    def __init__(self):
        super(SCF_CNN_16, self).__init__()

        # 定义卷积层和激活层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.3)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.3)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.3)

        # 定义池化层
        self.max_pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, padding=0)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, padding=0)
        self.max_pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, padding=0)

        # 定义全连接层
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=256, out_features=256)
        self.dense2 = nn.Linear(in_features=256, out_features=4)

    def forward(self, x):
        # 通过卷积层和激活层
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.max_pool3(x)
        print(x.shape)
        # 展平
        x = self.flatten(x)
        print(x.shape)
        # 通过全连接层
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)

        return x

class SCF_CNN_16_Conv(SCF_CNN_16):
    def __init__(self, conv_2d_dict, conv_2d_bias_dict, fc_dict, domain_num=1, classes=11):
        super(SCF_CNN_16_Conv, self).__init__()

        # 定义卷积层和激活层
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.convS_1 = SVD_Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1, bias=True, dilation=1, groups=1,
                   origin_weight=list(conv_2d_dict.values())[0] if domain_num == 1 else None,
                   origin_bias=list(conv_2d_bias_dict.values())[0] if domain_num == 1 else None,
                   domain_num=domain_num, rank=9)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.3)

        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.convS_2 = SVD_Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[1] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[1] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=128)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.3)

        # self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.convS_3 = SVD_Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[2] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[2] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=64)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.3)

        # 定义池化层
        self.max_pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, padding=0)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, padding=0)
        self.max_pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, padding=0)

        # 定义全连接层
        self.flatten = nn.Flatten()
        #self.dense1 = nn.Linear(in_features=256, out_features=256)

        self.dense1 = SVD_Linear_new(self.dense1.weight.shape[0], self.dense1.weight.shape[1], domain_num=domain_num,
                       fc_weight=fc_dict['dense1.weight'], fc_bias=fc_dict['dense1.bias'])
        self.dense2 = SVD_Linear_new(self.dense2.weight.shape[0], self.dense2.weight.shape[1], domain_num=domain_num,
                     fc_weight=fc_dict['dense2.weight'], fc_bias=fc_dict['dense2.bias'])
    def forward(self, x, pos=-1):
        # 通过卷积层和激活层
        x = self.convS_1(x, pos)
        x = self.leaky_relu1(x)
        x = self.max_pool1(x)
        x = self.convS_2(x, pos)
        x = self.leaky_relu2(x)
        x = self.max_pool2(x)
        x = self.convS_3(x, pos)
        x = self.leaky_relu3(x)
        x = self.max_pool3(x)
        # 展平
        x = self.flatten(x)
        # 通过全连接层
        x = self.dense1(x, pos)
        x = F.relu(x)
        x = self.dense2(x, pos)

        return x
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.alpha
        return grad_input, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class MCLDNN_SVD(MCLDNN):
    def __init__(self, fc_dict=None, classes=11, domain_classes=1, s_num=1):
        super(MCLDNN_SVD, self).__init__()
        self.domain_classes = domain_classes
        if self.domain_classes >= 2:
            self.prompt_layer = nn.Sequential(nn.Linear(128, 128),
                                              # nn.LayerNorm(128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, 128),
                                              # nn.LayerNorm(128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, 128),
                                              # nn.LayerNorm(128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, domain_classes))
        if fc_dict is None:
            self.fc1 = SVD_Linear(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=self.domain_classes)
            self.fc2 = SVD_Linear(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=self.domain_classes)
            self.fc3 = SVD_Linear(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=self.domain_classes)
        else:
            self.fc1 = SVD_Linear(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=self.domain_classes, fc_weight=fc_dict['fc1.weight'], fc_bias=fc_dict['fc1.bias'])
            self.fc2 = SVD_Linear(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=self.domain_classes, fc_weight=fc_dict['fc2.weight'], fc_bias=fc_dict['fc2.bias'])
            self.fc3 = SVD_Linear(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=self.domain_classes, fc_weight=fc_dict['fc3.weight'], fc_bias=fc_dict['fc3.bias'])

    def forward(self, input1, input2, input3, only_flag=False, energy=None):
        x = self.encoder(input1, input2, input3)
        if self.domain_classes == 1:
            domain = torch.ones((x.shape[0], self.domain_classes))
        else:
            domain = self.prompt_layer(x)
        if only_flag:
            domain = torch.zeros((x.shape[0], self.domain_classes))
            domain[:, -1] = 1
        if energy is not None:
            if isinstance(energy, int):
                domain = torch.zeros((x.shape[0], self.domain_classes))
                domain[:, energy] = 1
            elif isinstance(energy, torch.Tensor):
                domain = energy
        x, s_1 = self.fc1(x, domain.detach())
        # print('1:', x)
        x = F.selu(x)
        # x = F.dropout(x, self.dr)
        # print('11:', x)
        x, s_2 = self.fc2(x, domain.detach())
        # x  = self.fc1(x)#, domain.detach())
        # x = F.selu(x)
        # x = F.dropout(x, self.dr)
        # x = self.fc2(x)#, domain.detach())
        x = F.selu(x)
        # x = F.dropout(x, self.dr)
        x, s_3 = self.fc3(x, domain)
        # print('SS')
        return x, domain, s_1, s_2, s_3

class MCLDNN_SVD_single(MCLDNN):
    def __init__(self, input_shape=[2, 128], input_shape2=[128], fc_dict=None, classes=11, domain_classes=1, s_num=1):
        super(MCLDNN_SVD_single, self).__init__()
        self.domain_classes = domain_classes
        self.encoder = MCLDNN_FE(classes=classes)
        if fc_dict is None:
            self.fc1 = SVD_Linear(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=self.domain_classes)
            self.fc2 = SVD_Linear(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=self.domain_classes)
            self.fc3 = SVD_Linear(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=self.domain_classes)
        else:
            self.fc1 = SVD_Linear(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=self.domain_classes,
                                  fc_weight=fc_dict['fc1.weight'], fc_bias=fc_dict['fc1.bias'])
            self.fc2 = SVD_Linear(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=self.domain_classes,
                                  fc_weight=fc_dict['fc2.weight'], fc_bias=fc_dict['fc2.bias'])
            self.fc3 = SVD_Linear(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=self.domain_classes,
                                  fc_weight=fc_dict['fc3.weight'], fc_bias=fc_dict['fc3.bias'])

    def forward(self, input1, input2, input3, testing=None):
        x1 = self.encoder(input1, input2, input3)
        if testing is None:
            domain = torch.zeros((x1.shape[0], self.domain_classes))
            domain[:, -1] = 1
        else:
            domain = testing
        x, s_1 = self.fc1(x1, domain.detach())
        x = F.selu(x)
        x, s_2 = self.fc2(x, domain.detach())
        x = F.selu(x)
        x, s_3 = self.fc3(x, domain)
        return x, x1

class SVD_Linear_MutiS(nn.Linear):
    def __init__(self, in_features, out_features, domain_num=3, num_classes=11, s_num=1, name='trainable_S', fc_weight=None, fc_bias=None):
        super(SVD_Linear_MutiS, self).__init__(out_features, in_features)
        self.s_num = s_num
        if fc_weight is not None:
            self.weight.data = fc_weight
        if fc_bias is not None:
            self.bias.data = fc_bias
        bias = nn.Parameter(torch.randn((domain_num, self.bias.shape[-1])), requires_grad=True)
        if domain_num > 1:
            bias.data[:-1, :] = self.bias.data
            self.bias = bias
        self.U, S, self.V = torch.svd(self.weight.T.detach())
        self.S = nn.Parameter(torch.randn((domain_num*self.s_num, self.weight.shape[0])), requires_grad=True)  # .cuda()
        # 对矩阵进行 QR 分解，并取转置得到行向量之间正交的矩阵，可以探索一下效果如何，上面那行是直接加载进去了。下面注释这两行是正交分解Q和R，这样可以让这两组参数隔离的比较开
        # q, r = torch.qr(self.S.t())
        # self.S.data.copy_(q.t().cuda())
        self.S.data.copy_(S)
        self.register_parameter(name, self.S)
        # self.bias = self.bias
        # if domain_num > 1:
        #     self.bias = torch.cat(self.bias, self.bias.resize((1, -1)))
        #     print(self.bias.shape)
        #     self.register_parameter('bias_'+str(domain_num-1), self.bias_list[-1])
        # self.bn1 = nn.LayerNorm(out_features*2)
        # self.bn2 = nn.LayerNorm(out_features*2)
        # self.bn = nn.LayerNorm(out_features)
        # 可以探索层的多少
        # self.bn1 = nn.BatchNorm1d(out_features*2, affine=True)
        # self.bn2 = nn.BatchNorm1d(out_features*2, affine=True)
        # self.bn3 = nn.BatchNorm1d(out_features, affine=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.5)
        # self.bottleneck1 = nn.Linear(out_features, out_features*2)
        # self.bottleneck2 = nn.Linear(out_features*2, out_features*2)
        # self.bottleneck3 = nn.Linear(out_features*2, out_features)

    def forward(self, x, domain_predict):
        '''
        x = self.bottleneck1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.bottleneck2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.bottleneck3(x)
        x = self.bn3(x)
        '''
        # x = self.bn(x)
        # print('bias:', self.bias.shape)
        st = domain_predict.argmax(dim=1) * self.s_num
        ed = domain_predict.argmax(dim=1) * self.s_num + self.s_num
        indices = [torch.arange(st[i], ed[i]).reshape(1, -1) for i in range(domain_predict.shape[0])]
        index = torch.cat(indices, dim=0)
        if len(self.bias.shape) > 1:
            Y = torch.mm(torch.mm(x, self.U.cuda()).mul(torch.mean(self.S[index, :], dim=1)), self.V.cuda()) + self.bias[domain_predict.argmax(dim=1), :]
        else:
            Y = torch.mm(torch.mm(x, self.U.cuda()).mul(torch.mean(self.S[index, :], dim=1)), self.V.cuda()) + self.bias
        # print(Y)
        # Y = F.linear(X, self.U.T.cuda())
        # Y = Y.mul(self.S[domain_predict.argmax(dim=1), :].cuda())
        # Y = F.linear(Y, self.V.cuda()) + self.bias.cuda()
        return Y, self.S
class MCLDNN_SVD_MutiS(MCLDNN):
    def __init__(self, fc_dict=None, classes=11, domain_classes=1, s_num=1):
        super(MCLDNN_SVD_MutiS, self).__init__()
        self.domain_classes = domain_classes
        if self.domain_classes >= 2:
            self.prompt_layer = nn.Sequential(nn.Linear(128, 128),
                                              # nn.LayerNorm(128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, 128),
                                              # nn.LayerNorm(128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, domain_classes))
        if fc_dict is None:
            self.fc1 = SVD_Linear(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=self.domain_classes)
            self.fc2 = SVD_Linear(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=self.domain_classes)
            self.fc3 = SVD_Linear(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=self.domain_classes)
        else:
            self.fc1 = SVD_Linear(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=self.domain_classes, fc_weight=fc_dict['fc1.weight'], fc_bias=fc_dict['fc1.bias'])
            self.fc2 = SVD_Linear(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=self.domain_classes, fc_weight=fc_dict['fc2.weight'], fc_bias=fc_dict['fc2.bias'])
            self.fc3 = SVD_Linear(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=self.domain_classes, fc_weight=fc_dict['fc3.weight'], fc_bias=fc_dict['fc3.bias'])

    def forward(self, input1, input2, input3, only_flag=False, energy=None):
        x = self.encoder(input1, input2, input3)
        if self.domain_classes == 1:
            domain = torch.ones((x.shape[0], self.domain_classes))
        else:
            domain = self.prompt_layer(x)
        if only_flag:
            domain = torch.zeros((x.shape[0], self.domain_classes))
            domain[:, -1] = 1
        if energy is not None:
            if isinstance(energy, int):
                domain = torch.zeros((x.shape[0], self.domain_classes))
                domain[:, energy] = 1
            elif isinstance(energy, torch.Tensor):
                domain = energy
        x, s_1 = self.fc1(x, domain.detach())
        # print('1:', x)
        x = F.selu(x)
        x = F.dropout(x, self.dr)
        # print('11:', x)
        x, s_2 = self.fc2(x, domain.detach())
        # x  = self.fc1(x)#, domain.detach())
        # x = F.selu(x)
        # x = F.dropout(x, self.dr)
        # x = self.fc2(x)#, domain.detach())
        x = F.selu(x)
        x = F.dropout(x, self.dr)
        x, s_3 = self.fc3(x, domain)
        # print('SS')
        return x, domain, s_1, s_2, s_3

class MCLDNN_SVD_GRL(MCLDNN):
    def __init__(self, fc_dict=None, classes=11, domain_classes=1, pro_class=1):
        super(MCLDNN_SVD_GRL, self).__init__()
        self.domain_classes = domain_classes
        self.grl = GradientReversalLayer()
        self.prompt_layer = nn.Sequential(nn.Linear(128, 128),
                                              # nn.LayerNorm(128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, 128),
                                              # nn.LayerNorm(128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, pro_class))
        if fc_dict is None:
            self.fc1 = SVD_Linear(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=self.domain_classes)
            self.fc2 = SVD_Linear(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=self.domain_classes)
            self.fc3 = SVD_Linear(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=self.domain_classes)
        else:
            self.fc1 = SVD_Linear(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=self.domain_classes, fc_weight=fc_dict['fc1.weight'], fc_bias=fc_dict['fc1.bias'])
            self.fc2 = SVD_Linear(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=self.domain_classes, fc_weight=fc_dict['fc2.weight'], fc_bias=fc_dict['fc2.bias'])
            self.fc3 = SVD_Linear(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=self.domain_classes, fc_weight=fc_dict['fc3.weight'], fc_bias=fc_dict['fc3.bias'])

    def forward(self, input1, input2, input3, only_flag=False, energy=None):
        x = self.encoder(input1, input2, input3)
        domain = self.prompt_layer(self.grl(x))
        domaincc = None
        if only_flag:
            domaincc = torch.zeros((x.shape[0], self.domain_classes))
            domaincc[:, -1] = 1
        x, s_1 = self.fc1(x, domaincc)
        # print('1:', x)
        x = F.selu(x)
        x = F.dropout(x, self.dr)
        # print('11:', x)
        x, s_2 = self.fc2(x, domaincc)
        # x  = self.fc1(x)#, domain.detach())
        # x = F.selu(x)
        # x = F.dropout(x, self.dr)
        # x = self.fc2(x)#, domain.detach())
        x = F.selu(x)
        x = F.dropout(x, self.dr)
        x, s_3 = self.fc3(x, domaincc)
        # print('SS')
        return x, domain, s_1, s_2, s_3

# 加入的新loss，可以使参数隔离的更开
def orthogonal_loss(matrix):
    # 计算矩阵的转置
    matrix_t = torch.transpose(matrix, 0, 1)
    # 计算矩阵和其转置的乘积
    product = torch.matmul(matrix, matrix_t)
    # print(product)
    # 计算 Frobenius 范数的平方
    frobenius_norm_square = torch.sum((product - torch.eye(matrix.shape[0]).cuda()) ** 2)
    # 返回 Frobenius 范数的平方作为损失
    return frobenius_norm_square


def cosine_similarity_loss(matrix):
    # Assume matrix is a torch tensor of shape (n+1, d)
    n = matrix.size(0) - 1  # Number of rows excluding the last one
    d = matrix.size(1)  # Dimensionality of each row vector

    # Extract the last row (target row) and other rows
    rn = matrix[-1, :]  # Last row
    r1_to_n_minus_1 = matrix[:-1, :]  # All rows except the last one

    # Calculate cosine similarity loss
    loss = 0
    for i in range(n):
        cos_sim = F.cosine_similarity(rn.unsqueeze(0), r1_to_n_minus_1[i].unsqueeze(0), dim=1)
        loss += cos_sim  # Cosine similarity loss (ReLU version)


    return loss


class SupConMCLDNN(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=128, feat_dim=256):
        super(SupConMCLDNN, self).__init__()
        self.encoder = MCLDNN_FE()
        self.head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim))
        
    def forward(self, input1, input2, input3):
        feat = self.encoder(input1, input2, input3)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

# PETCGDNN模型

class PETCGDNN_FE2(nn.Module):
    def __init__(self, input_shape=[2, 128], classes=11):
        super(PETCGDNN_FE2, self).__init__()
        self.dr = 0.2
        self.input = nn.Flatten()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=75, kernel_size=(2, 8), padding=0)
        self.conv2 = nn.Conv2d(in_channels=75, out_channels=25, kernel_size=(1, 5), padding=0)
        self.gru = nn.GRU(input_size=25, hidden_size=128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(input_shape[0]*input_shape[1], 1)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(128, classes)
        self.softmax = nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()

    def forward(self, input1, input2, input3):
        input2 = input2.squeeze(dim=1)
        input3 = input3.squeeze(dim=1)
        # print(input1.shape, input2.shape, input3.shape)  # torch.Size([512, 1, 2, 128]) torch.Size([512, 128]) torch.Size([512, 128])
        x1 = self.fc1(self.input(input1))
        # print('x1:', x1[0][0])  # torch.Size([512, 256])
        # print(x1.shape)  # torch.Size([512, 1])
        cos1 = torch.cos(x1)
        sin1 = torch.sin(x1)

        x11 = torch.mul(input2, cos1)
        x12 = torch.mul(input3, sin1)
        x21 = torch.mul(input3, cos1)
        x22 = torch.mul(input2, sin1)
        # print('x11-x22:', x11.shape, x12.shape, x21.shape, x22.shape)  # x11-x22: torch.Size([512, 128])

        y1 = torch.add(x11, x12)
        y2 = torch.sub(x21, x22)
        # print('y1 y2:', y1.shape, y2.shape)  # y1 y2: torch.Size([512, 128])
        y1 = y1.unsqueeze(1)
        y2 = y2.unsqueeze(1)
        # print('y1 y2:', y1.shape, y2.shape)  # y1 y2: torch.Size([512, 1, 128]) torch.Size([512, 1, 128])
        x11 = torch.cat((y1, y2), 1)
        # print('x11:', x11.shape)   # x11: torch.Size([512, 2, 128])
        x11 = x11.unsqueeze(1)
        # print('x11:', x11.shape)   # x11: torch.Size([512, 1, 2, 128])
        x3 = F.relu(self.conv1(x11))
        x3 = F.dropout(x3, self.dr)
        # print(x3.shape)
        x3 = F.relu(self.conv2(x3))
        # print('x33:', x3[0])
        x3 = F.dropout(x3, self.dr)
        # print('x3:', x3.shape)   # x3: torch.Size([512, 25, 1, 117])
        x3 = x3.squeeze(2)
        # print('x3:', x3.shape)   # x3: torch.Size([512, 25, 117])
        x3 = x3.permute(0, 2, 1)
        # print('x3:', x3.shape)    # x3: torch.Size([512, 117, 25])
        x4, hidden = self.gru(x3)
        # print('x4:', x4[0])
        # print('x4:', x4.shape)    # x4: torch.Size([512, 117, 128])
        x = x4[:, -1, :]
        # print(x.shape)   # torch.Size([512, 128])
        return x

class PETCGDNN2(nn.Module):
    def __init__(self, input_shape=[2, 128], classes=11):
        super(PETCGDNN2, self).__init__()
        self.encoder = PETCGDNN_FE2(input_shape=input_shape, classes=classes)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, classes)

    def forward(self, input1, input2, input3):
        x = self.encoder(input1, input2, input3)
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)

        return x

class PETCGDNN_FE2_SVD(nn.Module):
    def __init__(self, conv_2d_dict, conv_2d_bias_dict, encoder_fc_dict, domain_num=1, classes=11, input_shape=[2, 128]):
        super(PETCGDNN_FE2_SVD, self).__init__()
        self.dr = 0.2
        self.input = nn.Flatten()
        self.convS_1 = SVD_Conv2d(1, 75, kernel_size=(2, 8), stride=1, padding=0, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[0] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[0] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=16)
        self.convS_2 = SVD_Conv2d(75, 25, kernel_size=(1, 5), stride=1, padding=0, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[1] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[1] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=25)
        self.gru = nn.GRU(input_size=25, hidden_size=128, num_layers=1, batch_first=True)
        self.fc1 = SVD_Linear_new(input_shape[0]*input_shape[1], 1, domain_num=domain_num,
                              fc_weight=encoder_fc_dict['encoder.fc1.weight'], fc_bias=encoder_fc_dict['encoder.fc1.bias'])
        self.softmax = nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()

    def forward(self, input1, input2, input3, pos):
        input2 = input2.squeeze(dim=1)
        input3 = input3.squeeze(dim=1)
        # print(input1.shape, input2.shape, input3.shape)  # torch.Size([512, 1, 2, 128]) torch.Size([512, 128]) torch.Size([512, 128])
        x1 = self.fc1(self.input(input1), pos)
        # print(self.input(input1))  # torch.Size([512, 256])
        # print(x1.shape)  # torch.Size([512, 1])
        cos1 = torch.cos(x1)
        sin1 = torch.sin(x1)

        x11 = torch.mul(input2, cos1)
        x12 = torch.mul(input3, sin1)
        x21 = torch.mul(input3, cos1)
        x22 = torch.mul(input2, sin1)
        # print('x11-x22:', x11.shape, x12.shape, x21.shape, x22.shape)  # x11-x22: torch.Size([512, 128])

        y1 = torch.add(x11, x12)
        y2 = torch.sub(x21, x22)
        # print('y1 y2:', y1.shape, y2.shape)  # y1 y2: torch.Size([512, 128])
        y1 = y1.unsqueeze(1)
        y2 = y2.unsqueeze(1)
        # print('y1 y2:', y1.shape, y2.shape)  # y1 y2: torch.Size([512, 1, 128]) torch.Size([512, 1, 128])
        x11 = torch.cat((y1, y2), 1)
        # print('x11:', x11.shape)   # x11: torch.Size([512, 2, 128])
        x11 = x11.unsqueeze(1)
        # print('x11:', x11.shape)   # x11: torch.Size([512, 1, 2, 128])
        x3 = F.relu(self.convS_1(x11, pos))
        x3 = F.dropout(x3, self.dr)
        x3 = F.relu(self.convS_2(x3, pos))
        x3 = F.dropout(x3, self.dr)
        # print('x33:', x3[0]) 
        # print('x3:', x3.shape)   # x3: torch.Size([512, 25, 1, 117])
        x3 = x3.squeeze(2)
        # print('x3:', x3.shape)   # x3: torch.Size([512, 25, 117])
        x3 = x3.permute(0, 2, 1)
        # print('x3:', x3.shape)    # x3: torch.Size([512, 117, 25])
        x4, hidden = self.gru(x3)
        # print('x4:', x4.shape)    # x4: torch.Size([512, 117, 128])
        x = x4[:, -1, :]
        # print('x4:', x4[0]) 
        # print(x.shape)   # torch.Size([512, 128])
        return x

class PETCGDNN_SVD_Conv(PETCGDNN2):
    def __init__(self, conv_2d_dict, conv_2d_bias_dict, encoder_fc_dict, fc_dict, input_shape=[2, 128], domain_num=1, classes=11):
        super(PETCGDNN_SVD_Conv, self).__init__()
        self.encoder = PETCGDNN_FE2_SVD(input_shape=input_shape, conv_2d_dict=conv_2d_dict, conv_2d_bias_dict=conv_2d_bias_dict, encoder_fc_dict=encoder_fc_dict, domain_num=domain_num, classes=classes)
        self.fc1 = SVD_Linear_new(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=domain_num,
                                  fc_weight=fc_dict['fc1.weight'], fc_bias=fc_dict['fc1.bias'])
        self.fc2 = SVD_Linear_new(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=domain_num,
                                  fc_weight=fc_dict['fc2.weight'], fc_bias=fc_dict['fc2.bias'])
        self.fc3 = SVD_Linear_new(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=domain_num,
                                  fc_weight=fc_dict['fc3.weight'], fc_bias=fc_dict['fc3.bias'])

    def forward(self, input1, input2, input3, pos=-1):
        x = self.encoder(input1, input2, input3, pos)
        x = F.selu(self.fc1(x, pos))
        x = F.selu(self.fc2(x, pos))
        x = self.fc3(x, pos)

        return x

class PETCGDNN(nn.Module):
    def __init__(self, input_shape=[2, 128], input_shape2=[128], classes=11):
        super(PETCGDNN, self).__init__()
        self.encoder = PETCGDNN_FE(input_shape=input_shape, classes=classes)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, classes)

    def forward(self, input1, input2, input3):
        x = self.encoder(input1, input2, input3)
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)

        return x

class PETCGDNN_FE(nn.Module):
    def __init__(self, input_shape=[2, 128], classes=11):
        super(PETCGDNN_FE, self).__init__()
        self.dr = 0.2
        self.input = nn.Flatten()

        self.conv1 = nn.Sequential(
            FFTConv2d(in_channels=1, out_channels=75, kernel_size=(2, 8), padding=0),
            nn.ReLU(),
            nn.Dropout(self.dr),
            FFTConv2d(in_channels=75, out_channels=25, kernel_size=(1, 5), padding=0),
            nn.ReLU(),
            nn.Dropout(self.dr),
        )

        self.gru = nn.GRU(input_size=25, hidden_size=128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(input_shape[0]*input_shape[1], 1)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(128, classes)
        self.softmax = nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()

    def forward(self, input1, input2, input3):
        input2 = input2.squeeze(dim=1)
        input3 = input3.squeeze(dim=1)
        # print(input1.shape, input2.shape, input3.shape)  # torch.Size([512, 1, 2, 128]) torch.Size([512, 128]) torch.Size([512, 128])
        x1 = self.fc1(self.input(input1))
        # print(self.input(input1).shape)  # torch.Size([512, 256])
        # print(x1.shape)  # torch.Size([512, 1])
        cos1 = torch.cos(x1)
        sin1 = torch.sin(x1)

        x11 = torch.mul(input2, cos1)
        x12 = torch.mul(input3, sin1)
        x21 = torch.mul(input3, cos1)
        x22 = torch.mul(input2, sin1)
        # print('x11-x22:', x11.shape, x12.shape, x21.shape, x22.shape)  # x11-x22: torch.Size([512, 128])

        y1 = torch.add(x11, x12)
        y2 = torch.sub(x21, x22)
        # print('y1 y2:', y1.shape, y2.shape)  # y1 y2: torch.Size([512, 128])
        y1 = y1.unsqueeze(1)
        y2 = y2.unsqueeze(1)
        # print('y1 y2:', y1.shape, y2.shape)  # y1 y2: torch.Size([512, 1, 128]) torch.Size([512, 1, 128])
        x11 = torch.cat((y1, y2), 1)
        # print('x11:', x11.shape)   # x11: torch.Size([512, 2, 128])
        x11 = x11.unsqueeze(1)
        # print('x11:', x11.shape)   # x11: torch.Size([512, 1, 2, 128])
        x3 = self.conv1(x11)
        # print('x3:', x3.shape)   # x3: torch.Size([512, 25, 1, 117])
        x3 = x3.squeeze(2)
        # print('x3:', x3.shape)   # x3: torch.Size([512, 25, 117])
        x3 = x3.permute(0, 2, 1)
        # print('x3:', x3.shape)    # x3: torch.Size([512, 117, 25])
        x4, hidden = self.gru(x3)
        # print('x4:', x4.shape)    # x4: torch.Size([512, 117, 128])
        x = x4[:, -1, :]
        # print(x.shape)   # torch.Size([512, 128])
        return x




class DC(nn.Module):
    def __init__(self, domain_classes=2):
        super(DC, self).__init__()
        self.FC = nn.Sequential(nn.Linear(128, 128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, 128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, 128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, domain_classes))
    def forward(self, x):
        return self.FC(x)

class PETCGDNN_SVD_718(PETCGDNN):
    def __init__(self, input_shape=[2, 128], input_shape2=[128], fc_dict=None, classes=11, domain_classes=1, s_num=1):
        super(PETCGDNN_SVD_718, self).__init__()
        self.domain_classes = domain_classes
        self.encoder = PETCGDNN_FE(input_shape=input_shape, classes=classes)
        if fc_dict is None:
            self.fc1 = SVD_Linear(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=self.domain_classes)
            self.fc2 = SVD_Linear(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=self.domain_classes)
            self.fc3 = SVD_Linear(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=self.domain_classes)
        else:
            self.fc1 = SVD_Linear(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=self.domain_classes,
                                  fc_weight=fc_dict['fc1.weight'], fc_bias=fc_dict['fc1.bias'])
            self.fc2 = SVD_Linear(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=self.domain_classes,
                                  fc_weight=fc_dict['fc2.weight'], fc_bias=fc_dict['fc2.bias'])
            self.fc3 = SVD_Linear(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=self.domain_classes,
                                  fc_weight=fc_dict['fc3.weight'], fc_bias=fc_dict['fc3.bias'])

    def forward(self, input1, input2, input3, testing=None):
        x1 = self.encoder(input1, input2, input3)
        if testing is None:
            domain = torch.zeros((x1.shape[0], self.domain_classes))
            domain[:, -1] = 1
        else:
            domain = testing
        x, s_1 = self.fc1(x1, domain.detach())
        x = F.selu(x)
        x, s_2 = self.fc2(x, domain.detach())
        x = F.selu(x)
        x, s_3 = self.fc3(x, domain)
        return x, x1




class PETCGDNN_SVD(PETCGDNN):
    def __init__(self, input_shape=[2, 128], input_shape2=[128], fc_dict=None, classes=11, domain_classes=1, no_dc=False, s_num=1):
        super(PETCGDNN_SVD, self).__init__()
        self.domain_classes = domain_classes
        self.no_dc = no_dc
        self.classes = classes
        self.encoder = PETCGDNN_FE(input_shape=input_shape, classes=classes)
        if self.domain_classes >= 2 and not no_dc:
            self.prompt_layer = nn.Sequential(nn.Linear(128, 128),
                                              # nn.LayerNorm(128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, 128),
                                              # nn.LayerNorm(128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, 128),
                                              # nn.LayerNorm(128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, domain_classes))
        if fc_dict is None:
            self.fc1 = SVD_Linear(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=self.domain_classes)
            self.fc2 = SVD_Linear(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=self.domain_classes)
            self.fc3 = SVD_Linear(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=self.domain_classes)
        else:
            self.fc1 = SVD_Linear(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=self.domain_classes,
                                  fc_weight=fc_dict['fc1.weight'], fc_bias=fc_dict['fc1.bias'])
            self.fc2 = SVD_Linear(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=self.domain_classes,
                                  fc_weight=fc_dict['fc2.weight'], fc_bias=fc_dict['fc2.bias'])
            self.fc3 = SVD_Linear(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=self.domain_classes,
                                  fc_weight=fc_dict['fc3.weight'], fc_bias=fc_dict['fc3.bias'])

    def forward(self, input1, input2, input3, only_flag=False, energy=None):
        x = self.encoder(input1, input2, input3)
        if self.domain_classes == 1:
            domain = torch.ones((x.shape[0], self.domain_classes))
        elif self.no_dc == False:
            domain = self.prompt_layer(x)
        if only_flag:
            domain = torch.zeros((x.shape[0], self.domain_classes))
            domain[:, -1] = 1
        if energy is not None:
            if isinstance(energy, int):
                domain = torch.zeros((x.shape[0], self.domain_classes))
                domain[:, energy] = 1
            elif isinstance(energy, torch.Tensor):
                domain = energy
        if self.no_dc and not only_flag:
            x_list, s_1_list, s_2_list, s_3_list = [], [], [], []
            x_old = x
            for i in range(self.domain_classes):
                domain = torch.zeros((x.shape[0], self.domain_classes))
                domain[:, i] = 1
                x, s_1 = self.fc1(x_old, domain.detach())
                x = F.selu(x)
                x, s_2 = self.fc2(x, domain.detach())
                x = F.selu(x)
                x, s_3 = self.fc3(x, domain)
                x_list.append(x.reshape(-1, 1, self.classes))
                s_1_list.append(s_1)
                s_2_list.append(s_2)
                s_3_list.append(s_3)
            x = torch.cat(x_list, dim=1)
            # print(x.shape)
            x = F.softmax(x, dim=2)
            # print(x[0])
            max_positions = torch.argmax(x, dim=2)
            # print(max_positions)
            # 初始化一个新张量来存储每一行的众数，形状为[512, 1]
            mode_tensor = torch.zeros((x.size(0), 1), dtype=torch.long)

            # 遍历tensor的每一行
            for i in range(max_positions.size(0)):
                # 对每一行的5个元素进行排序
                row = max_positions[i].view(-1)
                # 找到当前行所有不同元素及其出现次数
                unique_elements, counts = torch.unique(row, return_counts=True)
                # 找到出现次数最多的元素
                mode_index = torch.argmax(counts)
                # 将众数存储在新张量中
                mode_tensor[i] = unique_elements[mode_index]

            # max_values, max_indices = torch.max(x, dim=1)
            #
            # # 使用torch.gather沿着正确的维度收集最大值所在的行
            # # 这里我们使用max_indices来索引tensor，并沿着第二个维度'dim=1'收集
            # x = x.gather(1, max_indices.view(x.size(0), 1, -1)).squeeze(1)
            # # print(x.shape, x[0])
            # # print(x.shape)
            # # x = torch.sum(x, dim=1)
            # # print(x.shape)
            s_1 = torch.cat(s_1_list)
            s_2 = torch.cat(s_2_list)
            s_3 = torch.cat(s_3_list)
            return mode_tensor, domain, s_1, s_2, s_3
            # return x, domain, s_1, s_2, s_3
        x, s_1 = self.fc1(x, domain.detach())
        x = F.selu(x)
        x, s_2 = self.fc2(x, domain.detach())
        x = F.selu(x)
        x, s_3 = self.fc3(x, domain)
        return x, domain, s_1, s_2, s_3

class PETCGDNN2_SVD(PETCGDNN):
    def __init__(self, input_shape=[2, 128], input_shape2=[128], fc_dict=None, classes=11, domain_classes=1, no_dc=False, s_num=1):
        super(PETCGDNN2_SVD, self).__init__()
        self.domain_classes = domain_classes
        self.no_dc = no_dc
        self.classes = classes
        self.encoder = PETCGDNN_FE2(input_shape=input_shape, classes=classes)
        if self.domain_classes >= 2 and not no_dc:
            self.prompt_layer = nn.Sequential(nn.Linear(128, 128),
                                              # nn.LayerNorm(128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, 128),
                                              # nn.LayerNorm(128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, 128),
                                              # nn.LayerNorm(128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, domain_classes))
        if fc_dict is None:
            self.fc1 = SVD_Linear(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=self.domain_classes)
            self.fc2 = SVD_Linear(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=self.domain_classes)
            self.fc3 = SVD_Linear(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=self.domain_classes)
        else:
            self.fc1 = SVD_Linear(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=self.domain_classes,
                                  fc_weight=fc_dict['fc1.weight'], fc_bias=fc_dict['fc1.bias'])
            self.fc2 = SVD_Linear(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=self.domain_classes,
                                  fc_weight=fc_dict['fc2.weight'], fc_bias=fc_dict['fc2.bias'])
            self.fc3 = SVD_Linear(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=self.domain_classes,
                                  fc_weight=fc_dict['fc3.weight'], fc_bias=fc_dict['fc3.bias'])

    def forward(self, input1, input2, input3, only_flag=False, energy=None):
        x = self.encoder(input1, input2, input3)
        if self.domain_classes == 1:
            domain = torch.ones((x.shape[0], self.domain_classes))
        elif self.no_dc == False:
            domain = self.prompt_layer(x)
        if only_flag:
            domain = torch.zeros((x.shape[0], self.domain_classes))
            domain[:, -1] = 1
        if energy is not None:
            if isinstance(energy, int):
                domain = torch.zeros((x.shape[0], self.domain_classes))
                domain[:, energy] = 1
            elif isinstance(energy, torch.Tensor):
                domain = energy
        if self.no_dc and not only_flag:
            x_list, s_1_list, s_2_list, s_3_list = [], [], [], []
            x_old = x
            for i in range(self.domain_classes):
                domain = torch.zeros((x.shape[0], self.domain_classes))
                domain[:, i] = 1
                x, s_1 = self.fc1(x_old, domain.detach())
                x = F.selu(x)
                x, s_2 = self.fc2(x, domain.detach())
                x = F.selu(x)
                x, s_3 = self.fc3(x, domain)
                x_list.append(x.reshape(-1, 1, self.classes))
                s_1_list.append(s_1)
                s_2_list.append(s_2)
                s_3_list.append(s_3)
            x = torch.cat(x_list, dim=1)
            # print(x.shape)
            x = F.softmax(x, dim=2)
            # print(x[0])
            max_positions = torch.argmax(x, dim=2)
            # print(max_positions)
            # 初始化一个新张量来存储每一行的众数，形状为[512, 1]
            mode_tensor = torch.zeros((x.size(0), 1), dtype=torch.long)

            # 遍历tensor的每一行
            for i in range(max_positions.size(0)):
                # 对每一行的5个元素进行排序
                row = max_positions[i].view(-1)
                # 找到当前行所有不同元素及其出现次数
                unique_elements, counts = torch.unique(row, return_counts=True)
                # 找到出现次数最多的元素
                mode_index = torch.argmax(counts)
                # 将众数存储在新张量中
                mode_tensor[i] = unique_elements[mode_index]

            # max_values, max_indices = torch.max(x, dim=1)
            #
            # # 使用torch.gather沿着正确的维度收集最大值所在的行
            # # 这里我们使用max_indices来索引tensor，并沿着第二个维度'dim=1'收集
            # x = x.gather(1, max_indices.view(x.size(0), 1, -1)).squeeze(1)
            # # print(x.shape, x[0])
            # # print(x.shape)
            # # x = torch.sum(x, dim=1)
            # # print(x.shape)
            s_1 = torch.cat(s_1_list)
            s_2 = torch.cat(s_2_list)
            s_3 = torch.cat(s_3_list)
            return mode_tensor, domain, s_1, s_2, s_3
            # return x, domain, s_1, s_2, s_3
        x, s_1 = self.fc1(x, domain.detach())
        x = F.selu(x)
        x, s_2 = self.fc2(x, domain.detach())
        x = F.selu(x)
        x, s_3 = self.fc3(x, domain)
        return x, domain, s_1, s_2, s_3
# CLDNN模型
class CLDNN_FE(nn.Module):
    def __init__(self, classes=11):
        super(CLDNN_FE, self).__init__()
        self.dr = 0.5
        self.conv1 = nn.Conv2d(1, 50, kernel_size=(1, 8))
        self.conv2 = nn.Conv2d(50, 50, kernel_size=(1, 8))
        self.conv3 = nn.Conv2d(50, 50, kernel_size=(1, 8))
        self.dropout = nn.Dropout(self.dr)
        if classes == 24:
            self.lstm = nn.LSTM(input_size=4072, hidden_size=50, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=488, hidden_size=50, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.pad(x, (0, 4), value=0)
        x = self.relu(self.conv1(x))
        x1 = self.dropout(x)
        
        x2 = F.pad(x1, (0, 4), value=0)
        x2 = self.relu(self.conv2(x2))
        x2 = self.dropout(x2)
        
        x3 = F.pad(x2, (0, 4), value=0)
        x3 = self.relu(self.conv3(x3))
        x3 = self.dropout(x3)

        xc = torch.cat([x1, x3], dim=3)
        xc = xc.reshape(-1, 50, xc.shape[2]*xc.shape[3])

        fea, _ = self.lstm(xc)
        fea = fea[:, -1, :]

        return fea

class CLDNN(nn.Module):
    def __init__(self, classes=11):
        super(CLDNN, self).__init__()
        self.dr = 0.5
        self.encoder = CLDNN_FE(classes=classes)
        self.fc1 = nn.Linear(50, 256)
        self.dropout = nn.Dropout(self.dr)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class Classifier_CLDNN_2(nn.Module):
    def __init__(self, classes=11):
        super(Classifier_CLDNN_2, self).__init__()
        self.fc1 = nn.Linear(50, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, self.dr)
        x = F.relu(self.fc2(x))
        # x = F.dropout(x, self.dr)
        x = self.fc3(x)
        return x

class Classifier_CLDNN(nn.Module):
    def __init__(self, classes=11):
        super(Classifier_CLDNN, self).__init__()
        self.fc1 = nn.Linear(50, 40)
        self.fc2 = nn.Linear(40, 30)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, self.dr)
        x = self.fc2(x)
        return x

class CLDNN_SVD(CLDNN):
    def __init__(self, fc_dict=None, classes=11, domain_classes=1, s_num=1):
        super(CLDNN_SVD, self).__init__()
        self.domain_classes = domain_classes
        if self.domain_classes >= 2:
            self.prompt_layer = nn.Sequential(nn.Linear(50, 128),
                                              # nn.LayerNorm(128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, 128),
                                              # nn.LayerNorm(128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, 128),
                                              # nn.LayerNorm(128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, domain_classes))
        if fc_dict is None:
            self.fc1 = SVD_Linear(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=self.domain_classes)
            self.fc2 = SVD_Linear(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=self.domain_classes)
            self.fc3 = SVD_Linear(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=self.domain_classes)
        else:
            self.fc1 = SVD_Linear(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=self.domain_classes,
                                  fc_weight=fc_dict['fc1.weight'], fc_bias=fc_dict['fc1.bias'])
            self.fc2 = SVD_Linear(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=self.domain_classes,
                                  fc_weight=fc_dict['fc2.weight'], fc_bias=fc_dict['fc2.bias'])
            self.fc3 = SVD_Linear(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=self.domain_classes,
                                  fc_weight=fc_dict['fc3.weight'], fc_bias=fc_dict['fc3.bias'])

    def forward(self, input1, only_flag=False, energy=None):
        x = self.encoder(input1)
        if self.domain_classes == 1:
            domain = torch.ones((x.shape[0], self.domain_classes))
        else:
            domain = self.prompt_layer(x)
        if only_flag:
            domain = torch.zeros((x.shape[0], self.domain_classes))
            domain[:, -1] = 1
        if energy is not None:
            if isinstance(energy, int):
                domain = torch.zeros((x.shape[0], self.domain_classes))
                domain[:, energy] = 1
            elif isinstance(energy, torch.Tensor):
                domain = energy
        x, s_1 = self.fc1(x, domain.detach())
        x = F.relu(x)
        x, s_2 = self.fc2(x, domain.detach())
        x = F.relu(x)
        x, s_3 = self.fc3(x, domain)
        return x, domain, s_1, s_2, s_3


class CLDNN_FE_SVD(nn.Module):
    def __init__(self, conv_2d_dict, conv_2d_bias_dict, domain_num=1, classes=11):
        super(CLDNN_FE_SVD, self).__init__()
        self.dr = 0.5
        self.convS_1 = SVD_Conv2d(1, 50, kernel_size=(1, 8), stride=1, padding=0, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[0] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[0] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=8)
        self.convS_2 = SVD_Conv2d(50, 50, kernel_size=(1, 8), stride=1, padding=0, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[1] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[1] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=50)
        self.convS_3 = SVD_Conv2d(50, 50, kernel_size=(1, 8), stride=1, padding=0, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[2] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[2] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=50)
        if classes == 24:
            self.lstm = nn.LSTM(input_size=4072, hidden_size=50, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=488, hidden_size=50, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, x, pos):
        x = F.pad(x, (0, 4), value=0)
        x1 = self.relu(self.convS_1(x, pos))

        x2 = F.pad(x1, (0, 4), value=0)
        x2 = self.relu(self.convS_2(x2, pos))

        x3 = F.pad(x2, (0, 4), value=0)
        x3 = self.relu(self.convS_3(x3, pos))

        xc = torch.cat([x1, x3], dim=3)
        xc = xc.reshape(-1, 50, xc.shape[2] * xc.shape[3])

        fea, _ = self.lstm(xc)
        fea = fea[:, -1, :]

        return fea


class CLDNN_SVD_Conv(CLDNN):
    def __init__(self, conv_2d_dict, conv_2d_bias_dict, fc_dict, domain_num=1, classes=11):
        super(CLDNN_SVD_Conv, self).__init__()
        self.dr = 0.5
        self.encoder = CLDNN_FE_SVD(conv_2d_dict=conv_2d_dict, conv_2d_bias_dict=conv_2d_bias_dict, domain_num=domain_num, classes=classes)
        self.fc1 = SVD_Linear_new(self.fc1.weight.shape[0], self.fc1.weight.shape[1], domain_num=domain_num,
                              fc_weight=fc_dict['fc1.weight'], fc_bias=fc_dict['fc1.bias'])

        self.fc2 = SVD_Linear_new(self.fc2.weight.shape[0], self.fc2.weight.shape[1], domain_num=domain_num,
                              fc_weight=fc_dict['fc2.weight'], fc_bias=fc_dict['fc2.bias'])
        self.fc3 = SVD_Linear_new(self.fc3.weight.shape[0], self.fc3.weight.shape[1], domain_num=domain_num,
                              fc_weight=fc_dict['fc3.weight'], fc_bias=fc_dict['fc3.bias'])
        self.relu = nn.ReLU()

    def forward(self, x, pos=-1):
        x = self.encoder(x, pos)
        x = self.fc1(x, pos)
        x = self.relu(x)
        x = self.fc2(x, pos)
        x = self.relu(x)
        x = self.fc3(x, pos)

        return x

# CLDNN模型
class CLDNN2_FE(nn.Module):
    def __init__(self, classes=11):
        super(CLDNN2_FE, self).__init__()
        self.dr = 0.3
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 3))
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(2, 3))
        self.conv3 = nn.Conv2d(256, 80, kernel_size=(1, 3))
        self.conv4 = nn.Conv2d(80, 80, kernel_size=(1, 3))
        self.dropout = nn.Dropout(self.dr)
        self.lstm = nn.LSTM(input_size=120, hidden_size=50, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.dropout(x)

        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        
        x = self.relu(self.conv4(x))
        x = self.dropout(x)

        x = x.squeeze(2)

        fea, _ = self.lstm(x)
        fea = fea[:, -1, :]

        return fea

class CLDNN2(nn.Module):
    def __init__(self, classes=11):
        super(CLDNN2, self).__init__()
        self.dr = 0.5
        self.encoder = CLDNN2_FE()
        self.fc1 = nn.Linear(50, 128)
        # self.dropout = nn.Dropout(self.dr)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, classes)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x
# DANN方法
class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, num_domains=3):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class feat_classifier(nn.Module):
    def __init__(self, bottleneck_dim=128, class_num=11):
        super(feat_classifier, self).__init__()
        self.fc = nn.Linear(bottleneck_dim, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x
    
class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=128):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.bn(x)
        return x


class ICAMC(nn.Module):
    def __init__(self, classes=11, l = 128):
        super(ICAMC, self).__init__()
        dr = 0.2
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 1), padding=(0, 0))
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(1, 1), padding=(0, 0))
        self.conv3 = nn.Conv2d(128, 64, kernel_size=(1, 1), padding=(0, 0))
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(1, 1), padding=(0, 0))
        self.dropout = nn.Dropout(dr)
        self.flatten = nn.Flatten()
        # self.dense1 = nn.Linear(1024 if classes == 11 else 8192, 128)
        self.dense1 = nn.Linear(8*l, 128)
        self.dense2 = nn.Linear(128, classes)

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.max_pool(F.relu(self.conv3(x)))
        x = self.max_pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x

class ICAMC_SVD_Conv(nn.Module):
    def __init__(self, conv_2d_dict, conv_2d_bias_dict, fc_dict, domain_num=1, classes=11):
        super(ICAMC_SVD_Conv, self).__init__()
        self.convS_1 = SVD_Conv2d(1, 256, kernel_size=(1, 1), stride=1, padding=0, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[0] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[0] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.convS_2 = SVD_Conv2d(256, 128, kernel_size=(1, 1), stride=1, padding=0, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[1] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[1] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=128)
        self.convS_3 = SVD_Conv2d(128, 64, kernel_size=(1, 1), stride=1, padding=0, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[2] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[2] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=64)
        self.convS_4 = SVD_Conv2d(64, 64, kernel_size=(1, 1), stride=1, padding=0, bias=True, dilation=1, groups=1,
                                  origin_weight=list(conv_2d_dict.values())[3] if domain_num == 1 else None,
                                  origin_bias=list(conv_2d_bias_dict.values())[3] if domain_num == 1 else None,
                                  domain_num=domain_num, rank=64)
        self.flatten = nn.Flatten()
        self.fc1 = SVD_Linear_new(1024 if classes == 11 else 8192, 128, domain_num=domain_num,
                       fc_weight=fc_dict['dense1.weight'], fc_bias=fc_dict['dense1.bias'])
        self.fc2 = SVD_Linear_new(128, classes, domain_num=domain_num,
                              fc_weight=fc_dict['dense2.weight'], fc_bias=fc_dict['dense2.bias'])

    def forward(self, x, pos=-1):
        x = self.max_pool(F.relu(self.convS_1(x, pos)))
        x = self.max_pool(F.relu(self.convS_2(x, pos)))
        x = self.max_pool(F.relu(self.convS_3(x, pos)))
        x = self.max_pool(F.relu(self.convS_4(x, pos)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x, pos))
        x = self.fc2(x, pos)
        return x
# IC-AMCNET
class ICAMC_FE(nn.Module):
    def __init__(self, classes=11):
        super(ICAMC_FE, self).__init__()
        dr = 0.4
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 8), padding='same')
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 4), padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 8), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 1))
        self.dropout = nn.Dropout(dr)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(1, 8), padding='same')
        self.flatten = nn.Flatten()

    def forward(self, input):
        x = F.relu(self.conv1(input))
        print(x.shape)
        x = self.pool1(x)
        print(x.shape)

        x = F.relu(self.conv2(x))
        print(x.shape)

        x = F.relu(self.conv3(x))
        print(x.shape)

        x = self.pool2(x)
        print(x.shape)

        x = self.dropout(x)
        print(x.shape)

        x = F.relu(self.conv4(x))
        print(x.shape)

        x = self.dropout(x)
        print(x.shape)

        x = self.flatten(x)
        print(x.shape)


        return x


class ICAMC_gai(nn.Module):
    def __init__(self, classes=11):
        super(ICAMC_gai, self).__init__()
        self.dr = 0.4
        self.encoder = ICAMC_FE(classes=classes)
        self.dense = nn.Sequential(
            nn.Linear(64 * 128 if classes==11 else 65536, 128),
            nn.ReLU(),
            nn.Dropout(self.dr),
            nn.Linear(128, classes), )

    def forward(self, input):
        x = self.encoder(input)
        x = self.dense(x)

        return x

class ICAMC_FE_LN(nn.Module):
    def __init__(self, classes=11):
        super(ICAMC_FE_LN, self).__init__()
        dr = 0.2
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 8), padding='same')
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 4), padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 8), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 1))
        self.dropout = nn.Dropout(dr)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(1, 8), padding='same')
        self.flatten = nn.Flatten()
        self.ln1 = nn.LayerNorm(128, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(64, elementwise_affine=True)
        self.ln3 = nn.LayerNorm(64, elementwise_affine=True)
        self.ln4 = nn.LayerNorm(64, elementwise_affine=True)

    def forward(self, input):
        x = self.ln1(self.pool1(F.relu(self.conv1(input))))
        x = F.relu(self.ln2(self.conv2(x)))
        x = self.ln3(self.pool2(F.relu(self.conv3(x))))
        x = self.dropout(x)
        x = F.relu(self.ln4(self.conv4(x)))
        x = self.dropout(x)
        x = self.flatten(x)

        return x
    
class ICAMC_FE_RMSN(nn.Module):
    def __init__(self, classes=11):
        super(ICAMC_FE_RMSN, self).__init__()
        dr = 0.2
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 8), padding='same')
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 4), padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 8), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 1))
        self.dropout = nn.Dropout(dr)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(1, 8), padding='same')
        self.flatten = nn.Flatten()
        self.rmsn1 = RMSNorm(128)
        self.rmsn2 = RMSNorm(64)
        self.rmsn3 = RMSNorm(64)
        self.rmsn4 = RMSNorm(64)
        
    def forward(self, input):
        x = self.pool1(F.relu(self.rmsn1(self.conv1(input))))
        x = F.relu(self.rmsn2(self.conv2(x)))
        x = self.pool2(F.relu(self.rmsn3(self.conv3(x))))
        x = self.dropout(x)
        x = F.relu(self.rmsn4(self.conv4(x)))
        x = self.dropout(x)
        x = self.flatten(x)

        return x

class ICAMC_FE_RevIN(nn.Module):
    def __init__(self, classes=11):
        super(ICAMC_FE_RevIN, self).__init__()
        dr = 0.2
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 8), padding='same')
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 4), padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 8), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 1))
        self.dropout = nn.Dropout(dr)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(1, 8), padding='same')
        self.flatten = nn.Flatten()
        self.RevIN1 = RevIN(128)
        self.RevIN2 = RevIN(64)
        self.RevIN3 = RevIN(64)
        self.RevIN4 = RevIN(64)
        
    def forward(self, input):
        x = self.RevIN1(input, 'norm')
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.RevIN2(x, 'norm')
        x = F.relu(self.conv2(x))
        x = self.RevIN3(x, 'norm')
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.RevIN4(self.conv4(x), 'norm'))
        x = self.RevIN4(x, 'denorm')
        x = self.dropout(x)
        x = self.flatten(x)

        return x

class ICAMC_FE_IN(nn.Module):
    def __init__(self, classes=11):
        super(ICAMC_FE_IN, self).__init__()
        dr = 0.2
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 8), padding='same')
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 4), padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 8), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 1))
        self.dropout = nn.Dropout(dr)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(1, 8), padding='same')
        self.flatten = nn.Flatten()
        self.In1 = nn.InstanceNorm2d(64)
        self.In2 = nn.InstanceNorm2d(64)
        self.In3 = nn.InstanceNorm2d(128)
        self.In4 = nn.InstanceNorm2d(128)

    def forward(self, input):
        x = self.pool1(F.relu(self.In1(self.conv1(input))))
        x = F.relu(self.In2(self.conv2(x)))
        x = self.pool2(F.relu(self.In3(self.conv3(x))))
        x = self.dropout(x)
        x = F.relu(self.In4(self.conv4(x)))
        x = self.dropout(x)
        x = self.flatten(x)

        return x

class ICAMC_FE_BN(nn.Module):
    def __init__(self, classes=11):
        super(ICAMC_FE_BN, self).__init__()
        dr = 0.2
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 8), padding='same')
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 4), padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 8), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 1))
        self.dropout = nn.Dropout(dr)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(1, 8), padding='same')
        self.flatten = nn.Flatten()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

    def forward(self, input):
        x = self.pool1(F.relu(self.bn1(self.conv1(input))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        x = self.flatten(x)

        return x
    
class ICAMC_FE_MSN(nn.Module):
    def __init__(self, classes=11):
        super(ICAMC_FE_MSN, self).__init__()
        dr = 0.2
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 8), padding='same')
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 4), padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 8), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 1))
        self.dropout = nn.Dropout(dr)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(1, 8), padding='same')
        self.flatten = nn.Flatten()
        self.msn1 = MSNorm(64)
        self.msn2 = MSNorm(64)
        self.msn3 = MSNorm(128)
        self.msn4 = MSNorm(128)

    def forward(self, input):
        x = self.pool1(F.relu(self.msn1(self.conv1(input))))
        x = F.relu(self.msn2(self.conv2(x)))
        x = self.pool2(self.msn3(F.relu(self.conv3(x))))
        x = self.dropout(x)
        x = F.relu(self.msn4(self.conv4(x)))
        x = self.dropout(x)
        x = self.flatten(x)

        return x

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x*torch.sigmoid(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(dim, hidden_dim)
        self.swish = Swish()
        self.fc3 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        f1 = x1 * self.swish(x2)
        f = self.fc3(f1)
        return f


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., talk_heads=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.talking_heads1 = nn.Conv2d(heads, heads, 1, bias=False) if talk_heads else nn.Identity()
        self.talking_heads2 = nn.Conv2d(heads, heads, 1, bias=False) if talk_heads else nn.Identity()

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = self.talking_heads1(dots)

        attn = self.attend(dots)
        attn = self.dropout(attn)
        attn = self.talking_heads2(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class GroupBlock(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(GroupBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, hidden_channel, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=3, padding=1, groups=in_channel),
            nn.ReLU(),
            nn.Conv1d(hidden_channel, in_channel, kernel_size=1, padding=0),
        )
        self.shortcut = nn.Sequential(
            nn.Identity(),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.shortcut(x)
        x3 = self.relu(x1 + x2)
        return x3

    def __call__(self, x):
        return self.forward(x)


class TransNet(nn.Module):
    def __init__(self, classes, dim=96):
        super(TransNet, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.We = nn.Parameter(torch.randn(1, dim, dim))
        self.dropout = nn.Dropout(0.1)
        self.preconv1 = nn.Sequential(
            nn.Conv1d(1, dim//2, kernel_size=32, stride=16),
            nn.ReLU()
        )
        self.preconv2 = nn.Sequential(
            nn.Conv1d(1, dim//2, kernel_size=32, stride=16),
            nn.ReLU()
        )
        # self.preconv3 = nn.Sequential(
        #     nn.Conv1d(1, dim//2, kernel_size=32, stride=16),
        #     nn.ReLU()
        # )
        self.coder = Transformer(dim=dim, depth=6, heads=8, dim_head=dim // 8, mlp_dim=dim * 2, dropout=0.3)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, classes)
        )

    def forward(self, x):
        x = x.squeeze(1).float()
        A = self.preconv1(x[:,0].reshape(x.shape[0],1,-1)).transpose(1, 2)
        P = self.preconv2(x[:,1].reshape(x.shape[0],1,-1)).transpose(1, 2)
        # F = self.preconv3(x[:,2].reshape(x.shape[0],1,-1)).transpose(1, 2)
        x = torch.cat((A,P), dim=2)
        B, N, _ = x.shape

        # class token & position coding
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=B)  # (B, 1, D)
        x = torch.cat((cls_tokens, torch.matmul(x, self.We)), dim=1)
        x += self.pos_embedding[:, :(N + 1)]
        x = self.dropout(x)

        codex = self.coder(x)[:, 0]
        fc = self.fc(codex.reshape(B, -1))

        return fc




class RMSNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.channels = channels
        self.scale = nn.Parameter(torch.ones(channels))
        
    def forward(self, x):
        
        xnorm = x.norm(2, dim=-1, keepdim=True)
        
        msn_x = xnorm * self.channels ** (-1. / 2)
        x_normed = x / (msn_x  + self.eps)
        
        out = x_normed * self.scale
            
        return out

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class MSNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(MSNorm, self).__init__()
        self.eps = eps
        
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))
        
    def forward(self, x):
        if x.dim() == 4: # 2D的情况
            # 计算 BatchNorm 的均值和方差
            batch_mean = x.mean(dim=[0, 2, 3], keepdims=True)
            batch_var = x.var(dim=[0, 2, 3], keepdims=True, unbiased=True)

            # 计算 InstanceNorm 的均值和方差
            instance_mean = x.mean(dim=[2, 3], keepdims=True)
            instance_var = x.var(dim=[2, 3], keepdims=True, unbiased=True)
            
            x_bn = (x - batch_mean) / (batch_var + self.eps) ** 0.5
            x_in = (x - instance_mean) / (instance_var + self.eps) ** 0.5
            
            out = (x_bn + x_in) * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
            
        if x.dim() == 3: # 1D的情况
            # 计算 BatchNorm 的均值和方差
            batch_mean = x.mean(dim=[0, 2], keepdims=True)
            batch_var = x.var(dim=[0, 2], keepdims=True, unbiased=True)
            
            # 计算 InstanceNorm 的均值和方差
            instance_mean = x.mean(dim=[2], keepdims=True)
            instance_var = x.var(dim=[2], keepdims=True, unbiased=True)

            x_bn = (x - batch_mean) / (batch_var + self.eps) ** 0.5
            x_in = (x - instance_mean) / (instance_var + self.eps) ** 0.5
            
            out = (x_bn + x_in) * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)
            
        if x.dim() == 2: # 1D的情况
            # 计算 BatchNorm 的均值和方差
            batch_mean = x.mean(dim=[0, 1], keepdims=True)
            batch_var = x.var(dim=[0, 1], keepdims=True, unbiased=True)
            
            # 计算 InstanceNorm 的均值和方差
            instance_mean = x.mean(dim=-1, keepdims=True)
            instance_var = x.var(dim=-1, keepdims=True, unbiased=True)

            x_bn = (x - batch_mean) / (batch_var + self.eps) ** 0.5
            x_in = (x - instance_mean) / (instance_var + self.eps) ** 0.5
            
            out = (x_bn + x_in) * self.gamma.view(1, -1) + self.beta.view(1, -1)
            
        return out + x
    
class MSNorm_learn(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super(MSNorm_learn, self).__init__()
        self.eps = eps
        
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))
        
        # 定义低通滤波器卷积核
        self.low_pass_filter_1d = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.low_pass_filter_2d = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        # 定义高通滤波器卷积核
        self.high_pass_filter_1d = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.high_pass_filter_2d = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        # 初始化卷积核参数
        nn.init.xavier_uniform_(self.low_pass_filter_1d.weight)
        nn.init.xavier_uniform_(self.low_pass_filter_2d.weight)
        nn.init.xavier_uniform_(self.high_pass_filter_1d.weight)
        nn.init.xavier_uniform_(self.high_pass_filter_2d.weight)
        
    def forward(self, x):
        if x.dim() == 4: # 2D的情况
            low_passed = self.low_pass_filter_2d(x)
            high_passed = self.high_pass_filter_2d(x)

            batch_mean = low_passed.mean(dim=[0, 2, 3], keepdims=True)
            batch_var = low_passed.var(dim=[0, 2, 3], keepdims=True, unbiased=True)

            # 计算 InstanceNorm 的均值和方差
            instance_mean = high_passed.mean(dim=[2, 3], keepdims=True)
            instance_var = high_passed.var(dim=[2, 3], keepdims=True, unbiased=True)
            
            x_bn = (low_passed - batch_mean) / (batch_var + self.eps) ** 0.5
            x_in = (high_passed - instance_mean) / (instance_var + self.eps) ** 0.5
            
            out = (x_bn + x_in) * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
            
        if x.dim() == 3: # 1D的情况
            low_passed = self.low_pass_filter_1d(x)
            high_passed = self.high_pass_filter_1d(x)
            
            batch_mean = low_passed.mean(dim=[0, 2], keepdims=True)
            batch_var = low_passed.var(dim=[0, 2], keepdims=True, unbiased=True)

            # 计算 InstanceNorm 的均值和方差
            instance_mean = high_passed.mean(dim=[2], keepdims=True)
            instance_var = high_passed.var(dim=[2], keepdims=True, unbiased=True)

            x_bn = (low_passed - batch_mean) / (batch_var + self.eps) ** 0.5
            x_in = (high_passed - instance_mean) / (instance_var + self.eps) ** 0.5
            
            out = (x_bn + x_in) * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)
            
        return out + x

# 梯度反转层
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class MCLDNN_FE_COSCL(nn.Module):
    def __init__(self, classes=11):
        super(MCLDNN_FE_COSCL, self).__init__()
        self.dr = 0.5
        self.encoder = MCLDNN_FE()
        self.encoder1 = MCLDNN_FE()
        self.encoder2 = MCLDNN_FE()

    def forward(self, input1, input2, input3):
        Experts = []
        x = self.encoder(input1, input2, input3)
        Experts.append(x.unsqueeze(0))
        x1 = self.encoder1(input1, input2, input3)
        Experts.append(x1.unsqueeze(0))
        x2 = self.encoder2(input1, input2, input3)
        Experts.append(x2.unsqueeze(0))

        h = torch.cat([h_result for h_result in Experts], 0)
        h = torch.sum(h, dim=0).squeeze(0)
        return h

class MCLDNN_COSCL(nn.Module):
    def __init__(self, classes=11):
        super(MCLDNN_COSCL, self).__init__()
        self.dr = 0.5
        self.encoder = MCLDNN_FE()
        self.encoder1 = MCLDNN_FE()
        self.encoder2 = MCLDNN_FE()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, classes)

    def forward(self, input1, input2, input3, return_expert=False):
        Experts = []
        Experts_y = []
        x = self.encoder(input1, input2, input3)
        Experts.append(x.unsqueeze(0))
        x1 = self.encoder1(input1, input2, input3)
        Experts.append(x1.unsqueeze(0))
        x2 = self.encoder2(input1, input2, input3)
        Experts.append(x2.unsqueeze(0))

        h = torch.cat([h_result for h_result in Experts], 0)
        h = torch.sum(h, dim=0).squeeze(0)

        h = F.selu(self.fc1(h))
        h = F.selu(self.fc2(h))
        h = self.fc3(h)
        if return_expert:
            for i in range(len(Experts)):
                h_exp = Experts[i].squeeze(0)
                h_exp = F.selu(self.fc1(h_exp))
                h_exp = F.selu(self.fc2(h_exp))
                h_exp = self.fc3(h_exp)
                Experts_y.append(h_exp)
            return h, Experts_y, Experts
        return h


class PETCGDNN_FE2_COSCL(nn.Module):
    def __init__(self, input_shape=[2, 128], classes=11):
        super(PETCGDNN_FE2_COSCL, self).__init__()
        self.encoder = PETCGDNN_FE2(input_shape=input_shape, classes=classes)
        self.encoder1 = PETCGDNN_FE2(input_shape=input_shape, classes=classes)
        self.encoder2 = PETCGDNN_FE2(input_shape=input_shape, classes=classes)

    def forward(self, input1, input2, input3):
        Experts = []
        x = self.encoder(input1, input2, input3)
        Experts.append(x.unsqueeze(0))
        x1 = self.encoder1(input1, input2, input3)
        Experts.append(x1.unsqueeze(0))
        x2 = self.encoder2(input1, input2, input3)
        Experts.append(x2.unsqueeze(0))

        h = torch.cat([h_result for h_result in Experts], 0)
        h = torch.sum(h, dim=0).squeeze(0)
        return h


class PETCGDNN2_COSCL(nn.Module):
    def __init__(self, input_shape=[2, 128], classes=11):
        super(PETCGDNN2_COSCL, self).__init__()
        self.encoder = PETCGDNN_FE2(input_shape=input_shape, classes=classes)
        self.encoder1 = PETCGDNN_FE2(input_shape=input_shape, classes=classes)
        self.encoder2 = PETCGDNN_FE2(input_shape=input_shape, classes=classes)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, classes)

    def forward(self, input1, input2, input3, return_expert=False):
        Experts = []
        Experts_y = []
        x = self.encoder(input1, input2, input3)
        Experts.append(x.unsqueeze(0))
        x1 = self.encoder1(input1, input2, input3)
        Experts.append(x1.unsqueeze(0))
        x2 = self.encoder2(input1, input2, input3)
        Experts.append(x2.unsqueeze(0))

        h = torch.cat([h_result for h_result in Experts], 0)
        h = torch.sum(h, dim=0).squeeze(0)

        h = F.selu(self.fc1(h))
        h = F.selu(self.fc2(h))
        h = self.fc3(h)
        if return_expert:
            for i in range(len(Experts)):
                h_exp = Experts[i].squeeze(0)
                h_exp = F.selu(self.fc1(h_exp))
                h_exp = F.selu(self.fc2(h_exp))
                h_exp = self.fc3(h_exp)
                Experts_y.append(h_exp)
            return h, Experts_y, Experts
        return h

class Wir_CNN_FE_COSCL(nn.Module):
    def __init__(self):
        super(Wir_CNN_FE_COSCL, self).__init__()
        self.encoder = Wir_CNN_FE()
        self.encoder1 = Wir_CNN_FE()
        self.encoder2 = Wir_CNN_FE()
    def forward(self, input):
        Experts = []
        x = self.encoder(input)
        Experts.append(x.unsqueeze(0))
        x1 = self.encoder1(input)
        Experts.append(x1.unsqueeze(0))
        x2 = self.encoder2(input)
        Experts.append(x2.unsqueeze(0))

        h = torch.cat([h_result for h_result in Experts], 0)
        h = torch.sum(h, dim=0).squeeze(0)
        return h


class Wir_CNN_COSCL(nn.Module):
    def __init__(self):
        super(Wir_CNN_COSCL, self).__init__()
        self.encoder = Wir_CNN_FE()
        self.encoder1 = Wir_CNN_FE()
        self.encoder2 = Wir_CNN_FE()

        # 全连接层
        self.fc1 = nn.Linear(8176, 25)
        self.fc2 = nn.Linear(25, 3)

    def forward(self, input, return_expert=False):
        Experts = []
        Experts_y = []
        x = self.encoder(input)
        Experts.append(x.unsqueeze(0))
        x1 = self.encoder1(input)
        Experts.append(x1.unsqueeze(0))
        x2 = self.encoder2(input)
        Experts.append(x2.unsqueeze(0))

        h = torch.cat([h_result for h_result in Experts], 0)
        h = torch.sum(h, dim=0).squeeze(0)

        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        if return_expert:
            for i in range(len(Experts)):
                h_exp = Experts[i].squeeze(0)
                h_exp = F.relu(self.fc1(h_exp))
                h_exp = self.fc2(h_exp)
                Experts_y.append(h_exp)
            return h, Experts_y, Experts
        return h

class SCF_CNN_16_FE_COSCL(nn.Module):
    def __init__(self):
        super(SCF_CNN_16_FE_COSCL, self).__init__()

        self.encoder = SCF_CNN_16_FE()
        self.encoder1 = SCF_CNN_16_FE()
        self.encoder2 = SCF_CNN_16_FE()

    def forward(self, input):
        Experts = []
        x = self.encoder(input)
        Experts.append(x.unsqueeze(0))
        x1 = self.encoder1(input)
        Experts.append(x1.unsqueeze(0))
        x2 = self.encoder2(input)
        Experts.append(x2.unsqueeze(0))

        h = torch.cat([h_result for h_result in Experts], 0)
        h = torch.sum(h, dim=0).squeeze(0)
        return h


class SCF_CNN_16_COSCL(nn.Module):
    def __init__(self):
        super(SCF_CNN_16_COSCL, self).__init__()
        self.encoder = SCF_CNN_16_FE()
        self.encoder1 = SCF_CNN_16_FE()
        self.encoder2 = SCF_CNN_16_FE()
        self.dense1 = nn.Linear(in_features=256, out_features=256)
        self.dense2 = nn.Linear(in_features=256, out_features=4)

    def forward(self, input, return_expert=False):
        Experts = []
        Experts_y = []
        x = self.encoder(input)
        Experts.append(x.unsqueeze(0))
        x1 = self.encoder1(input)
        Experts.append(x1.unsqueeze(0))
        x2 = self.encoder2(input)
        Experts.append(x2.unsqueeze(0))

        h = torch.cat([h_result for h_result in Experts], 0)
        h = torch.sum(h, dim=0).squeeze(0)

        h = F.relu(self.dense1(h))
        h = self.dense2(h)
        if return_expert:
            for i in range(len(Experts)):
                h_exp = Experts[i].squeeze(0)
                h_exp = F.relu(self.dense1(h_exp))
                h_exp = self.dense2(h_exp)
                Experts_y.append(h_exp)
            return h, Experts_y, Experts
        return h

class KLD(nn.Module):
    def __init__(self):
        super(KLD, self).__init__()
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x):
        KLD_loss = 0
        for k in range(len(x)):
            for l in range(len(x)):
                if l != k:
                    KLD_loss += self.criterion_KLD(F.log_softmax(x[k], dim=1), F.softmax(x[l], dim=1).detach())

        return KLD_loss