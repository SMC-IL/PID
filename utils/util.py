from __future__ import print_function
import random
import numpy as np
import torch
from types import FunctionType
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import Counter  
import torch.nn as nn
from scipy.spatial.distance import cdist

class TwoCropTransform:
    """Create two crops of the same signal"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
 
    def __call__(self, img):
        img = np.array(img)
        c, h, w = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(c, h, w))
        img = N + img
        img = torch.tensor(img)
        return img
    
class MyToTensor:
    def __init__(self) -> None:
        _log_api_usage_once(self)
    def __call__(self, pic):
        return torch.Tensor(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def discrepancy(self, out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.get_cmap("Blues"), labels=[],save_filename=None):
    plt.figure(figsize=(10, 6),dpi=600)
    plt.imshow(cm*100, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90,size=12)
    plt.yticks(tick_marks, labels,size=12)
    for i in range(len(tick_marks)):
        for j in range(len(tick_marks)):
            if i!=j:
                text=plt.text(j,i,int(np.around(cm[i,j]*100)),ha="center",va="center",fontsize=10)
            elif i==j:
                if int(np.around(cm[i,j]*100))==100:
                    text=plt.text(j,i,int(np.around(cm[i,j]*100)),ha="center",va="center",fontsize=7,color='darkorange')
                else:
                    text=plt.text(j,i,int(np.around(cm[i,j]*100)),ha="center",va="center",fontsize=10,color='darkorange')
            

    plt.tight_layout()
    plt.ylabel('True label',fontdict={'size':8,})
    plt.xlabel('Predicted label',fontdict={'size':8,})
    if save_filename is not None:
        plt.savefig(save_filename,dpi=600,bbox_inches = 'tight')
    plt.close()

def _log_api_usage_once(obj) -> None:
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")



def Entropylogits(input, redu='mean'):
    input_ = F.softmax(input, dim=1)
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    if redu == 'mean':
        entropy = torch.mean(torch.sum(entropy, dim=1))
    elif redu == 'None':
        entropy = torch.sum(entropy, dim=1)
    return entropy

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss
    
class FocalLoss_sample(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss_sample, self).__init__()
        self.gamma = gamma
        self.weight = weight  # 这是 类别权重，不是样本权重

    def forward(self, inputs, targets):
        # 关键：reduction='none' → 输出 [batch_size]，每个样本独立loss
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none') 
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss  # 返回 [batch_size]
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def set_dlabel(model, dbottleneck, dgenerator, loader, dataset):
    model.eval()
    dbottleneck.eval()
    dgenerator.eval()

    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            inputs = inputs.cuda().float()
            # inputs1 = inputs[:,:,0,:]
            # inputs2 = inputs[:,:,1,:]
            # feas = dbottleneck(model_2016A.encoder(inputs, inputs1, inputs2))
            feas = dbottleneck(model.encoder(inputs))
            index = data[-1]
            outputs = dgenerator(feas)
            all_index = index
    
    pred_label = outputs.argmax(dim=1, keepdim=True)
    pred_label = pred_label.squeeze().cpu().numpy()

    dataset.set_labels_by_index(pred_label, all_index, 'domain_label')
    print(Counter(pred_label))
    
    dgenerator.train()
    model.train()
    dbottleneck.train()

def get_params(model, backbone, module='domain') -> torch.Tensor:
    """
    Returns all the parameters concatenated in a single tensor.

    Returns:
        parameters tensor
    """
    params = []
    for name, pp in model.named_parameters():
        if backbone == 'PETCGDNN':
            # 防止PETCGDNN里面有未使用的参数
            if 'encoder' in name and ('fc2' in name or 'fc3' in name):
                continue
        elif module == 'class':
            if 'fc3' in name:
                continue
        params.append(pp.view(-1))

    return torch.cat(params)

def set_params(model, backbone, new_params: torch.Tensor) -> None:
    """
    Sets the parameters to a given value.

    Args:
        new_params: concatenated values to be set
    """
    progress = 0
    for name, pp in model.named_parameters():
        if backbone == 'PETCGDNN':
            # 防止PETCGDNN里面有未使用的参数
            if 'encoder' in name and ('fc2' in name or 'fc3' in name):
                continue
        cand_params = new_params[progress: progress +
                                 torch.tensor(pp.size()).prod()].view(pp.size())
        progress += torch.tensor(pp.size()).prod()
        pp.data = cand_params
    # for pp in list(model.parameters()):
    #     cand_params = new_params[progress: progress +
    #                              torch.tensor(pp.size()).prod()].view(pp.size())
    #     progress += torch.tensor(pp.size()).prod()
    #     pp.data = cand_params
    return model

def get_grads(model) -> torch.Tensor:
    """
    Returns all the gradients concatenated in a single tensor.

    Returns:
        gradients tensor
    """
    return torch.cat(get_grads_list(model))


def get_grads_list(model):
    """
    Returns a list containing the gradients (a tensor for each layer).

    Returns:
        gradients list
    """
    grads = []
    for pp in list(model.parameters()):
        if pp.grad is not None:
            grads.append(pp.grad.view(-1))
    return grads

