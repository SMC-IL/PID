import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Buffer(nn.Module):
    def __init__(self, b_s=44000, n_classes=11, input_size=(2, 128)):
        super().__init__()
        self.k = 0.03

        self.place_left = True  # 有剩余位置
        self.buffer_num = 0
        # buffer_size为缓冲区大小，即能容纳多少数据
        self.buffer_size = b_s
        print('buffer has %d slots' % self.buffer_size)
        # *input_size将变量转换为元组，和前面的buffer_size共同组成元组，也就是创建一个放置x的缓冲区存储器，类型FloatTensor,size为(buffer_size, *input_size)
        bx = torch.FloatTensor(self.buffer_size, *input_size).fill_(0)  # 放数据
        by = torch.LongTensor(self.buffer_size).fill_(0)  # 放label的缓冲区存储器
        bs = torch.LongTensor(self.buffer_size).fill_(0)  # snr标签
        bdl = torch.LongTensor(self.buffer_size).fill_(0) # dlabel
        logits = torch.FloatTensor(self.buffer_size, n_classes).fill_(0)  # 概率向量缓冲区
        feature = torch.FloatTensor(self.buffer_size, 512).fill_(0)  # 特征缓冲区

        bx = bx.cuda()
        by = by.cuda()
        bs = bs.cuda()
        bdl = bdl.cuda()
        logits = logits.cuda()
        feature = feature.cuda()
        self.save_logits = None

        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full = 0
        self.domains = []

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.register_buffer('bs', bs)
        self.register_buffer('bdl', bdl)
        self.register_buffer('logits', logits)
        self.register_buffer('feature', feature)
        # 将label转换为one-hot向量
        self.to_one_hot = lambda x: x.new(x.size(0), n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x: torch.arange(x.size(0)).to(x.device)
        # 将0-x.size(0)随机打乱
        self.shuffle = lambda x: x[torch.randperm(x.size(0))]

    @property
    def x(self):
        return self.bx[:self.current_index]

    @property
    def y(self):
        return self.to_one_hot(self.by[:self.current_index])

    @property
    def snr(self):
        return self.bs[:self.current_index]

    @property
    def dlabel(self):
        return self.bdl[:self.current_index]
    
    @property
    def valid(self):
        return self.is_valid[:self.current_index]

    # def display(self, gen=None, epoch=-1):
    #     from torchvision.utils import save_image
    #     from PIL import Image
    #
    #     if 'cifar' in self.args.dataset:
    #         shp = (-1, 3, 32, 32)
    #     elif 'tinyimagenet' in self.args.dataset:
    #         shp = (-1, 3, 64, 64)
    #     else:
    #         shp = (-1, 1, 28, 28)
    #
    #     if gen is not None:
    #         x = gen.decode(self.x)
    #     else:
    #         x = self.x
    #
    #     save_image((x.reshape(shp) * 0.5 + 0.5), 'samples/buffer_%d.png' % epoch, nrow=int(self.current_index ** 0.5))
    #     # Image.open('buffer_%d.png' % epoch).show()
    #     print(self.y.sum(dim=0))

    # 存储数据
    def add_reservoir(self, x, y, s=None, dl=None, logits=None):
        n_elem = x.size(0)
        self.buffer_num += n_elem
        save_logits = logits
        self.save_logits = logits
        # add whatever still fits in the buffer
        place_left = max(0, self.bx.size(0) - self.current_index)
        if place_left:
            offset = min(place_left, n_elem)
            self.bx[self.current_index: self.current_index + offset].data.copy_(x[:offset])
            self.by[self.current_index: self.current_index + offset].data.copy_(y[:offset])
            self.bdl[self.current_index: self.current_index + offset].data.copy_(dl[:offset])
            if save_logits is not None:
                # 新类进来后self.logits维度变大，之前任务的logits维度不够，全补0
                pad_0 = self.logits.shape[-1] - logits.shape[-1]
                logits = F.pad(logits, (0, pad_0))
                self.logits[self.current_index: self.current_index + offset].data.copy_(logits[:offset])
            if s is not None:
                self.bs[self.current_index: self.current_index + offset].data.copy_(s[:offset])

            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == x.size(0):
                return

        self.place_left = False

        # 删除缓冲区中已经存在的内容
        x, y, s, dl = x[place_left:], y[place_left:], s[place_left:], dl[place_left:]

        # 使用随机均匀分布抽取序号
        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, self.n_seen_so_far).long()
        # 筛选出满足条件的序号 < 满足结果对应元素为1， 否则为0
        valid_indices = (indices < self.bx.size(0)).long()
        # 把上一步满足条件（即元素为1）的序号挑出来，作为最终要替换的x中数据序号
        idx_new_data = valid_indices.nonzero().squeeze(-1)
        # 得到要被替换的buffer中的数据序号
        idx_buffer = indices[idx_new_data]

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return

        assert idx_buffer.max() < self.bx.size(0)
        assert idx_buffer.max() < self.by.size(0)
        assert idx_buffer.max() < self.bs.size(0)

        assert idx_new_data.max() < x.size(0)
        assert idx_new_data.max() < y.size(0)
        assert idx_new_data.max() < s.size(0)

        # 执行覆盖操作, 对应位置替换
        self.bx[idx_buffer] = x[idx_new_data]
        self.by[idx_buffer] = y[idx_new_data]
        self.bs[idx_buffer] = s[idx_new_data]
        self.bdl[idx_buffer] = dl[idx_new_data]

        if save_logits:
            self.logits[idx_buffer] = logits[idx_new_data]
        return idx_buffer

    # 非任务增量取数据
    def onlysample(self, amt, task=None, ret_ind=False):
        if task is not None:
            valid_indices = (self.dlabel == task)
            valid_indices = valid_indices.nonzero().squeeze()
            bx, by, bs, bdl, logits = self.bx[valid_indices], self.by[valid_indices], self.bs[valid_indices], self.bdl[valid_indices],\
                                 self.logits[valid_indices]
        else:
            bx, by, bs, bdl, logits = self.bx[:self.current_index], self.by[:self.current_index], \
                                 self.bs[:self.current_index], self.bdl[:self.current_index], self.logits[:self.current_index]

        if bx.size(0) < amt:
            if ret_ind:
                return bx, by, logits, bs, bdl, torch.from_numpy(np.arange(bx.size(0)))
            else:
                return bx, by, logits, bs, bdl
        else:
            indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False))
            indices = indices.cuda()

            if ret_ind:
                return bx[indices], by[indices], logits[indices], bs[indices], bdl[indices], indices
            else:
                return bx[indices], by[indices], logits[indices], bs[indices], bdl[indices]


    def measure_valid(self, generator, classifier):
        with torch.no_grad():
            # fetch valid examples
            valid_indices = self.valid.nonzero()
            valid_x, valid_y = self.bx[valid_indices], self.by[valid_indices]
            one_hot_y = self.to_one_hot(valid_y.flatten())

            hid_x = generator.idx_2_hid(valid_x)
            x_hat = generator.decode(hid_x)

            logits = classifier(x_hat)
            _, pred = logits.max(dim=1)
            one_hot_pred = self.to_one_hot(pred)
            correct = one_hot_pred * one_hot_y

            per_class_correct = correct.sum(dim=0)
            per_class_deno = one_hot_y.sum(dim=0)
            per_class_acc = per_class_correct.float() / per_class_deno.float()
            self.class_weight = 1. - per_class_acc
            self.valid_acc = per_class_acc
            self.valid_deno = per_class_deno

    def shuffle_(self):
        indices = torch.randperm(self.current_index).cuda()
        self.bx = self.bx[indices]
        self.by = self.by[indices]
        self.bs = self.bs[indices]
        self.bdl = self.bdl[indices]
        self.logits = self.logits[indices]

    def delete_up_to(self, remove_after_this_idx):
        self.bx = self.bx[:remove_after_this_idx]
        self.by = self.by[:remove_after_this_idx]
        self.bs = self.bs[:remove_after_this_idx]
        self.bdl = self.bdl[:remove_after_this_idx]
        self.logits = self.logits[:remove_after_this_idx]

    def sample(self, amt, exclude_task=None, ret_ind=False):
        if self.save_logits:
            if exclude_task is not None:
                # 筛选出不属于目前任务的存储数据
                valid_indices = (self.t != exclude_task)
                valid_indices = valid_indices.nonzero().squeeze()
                bx, by, bs, bdl, logits = self.bx[valid_indices], self.by[valid_indices], self.bs[valid_indices], self.bdl[valid_indices], \
                                     self.logits[valid_indices]
            else:
                bx, by, bs, bdl, logits = self.bx[:self.current_index], self.by[:self.current_index], \
                    self.bs[:self.current_index], self.bdl[:self.current_index], self.logits[:self.current_index]

            if bx.size(0) < amt:
                if ret_ind:
                    return bx, by, logits, bs, bdl, torch.from_numpy(np.arange(bx.size(0)))
                else:
                    return bx, by, logits, bs, bdl
            else:
                indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False))

                indices = indices.cuda()

                if ret_ind:
                    return bx[indices], by[indices], logits[indices], bs[indices], bdl[indices], indices
                else:
                    return bx[indices], by[indices], logits[indices], bs[indices], bdl[indices]
        else:
            if exclude_task is not None:
                valid_indices = (self.t != exclude_task)
                valid_indices = valid_indices.nonzero().squeeze()
                bx, by, bs, bdl = self.bx[valid_indices], self.by[valid_indices], self.bs[valid_indices], self.bdl[valid_indices]
            else:
                bx, by, bs, bdl = self.bx[:self.current_index], self.by[:self.current_index], self.bs[:self.current_index], self.bdl[:self.current_index]

            if bx.size(0) < amt:
                if ret_ind:
                    return bx, by, bs, bdl, torch.from_numpy(np.arange(bx.size(0)))
                else:
                    return bx, by, bs, bdl
            else:
                indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False)).long()

                indices = indices.cuda()

                if ret_ind:
                    return bx[indices], by[indices], bs[indices], bdl[indices], indices
                else:
                    return bx[indices], by[indices], bs[indices], bdl[indices]

    def split(self, amt):
        indices = torch.randperm(self.current_index).cuda()
        return indices[:amt], indices[amt:]

