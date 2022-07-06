import os
import matplotlib.pyplot as plt
import torchvision
import torch
import numpy as np
import random

plt.ion()


def mkdir(floder):
    if not os.path.exists(floder):
        os.makedirs(floder)


# 查看各个类别样本情况？效率很低；Subset bug!
def dataset_class_count(dataset):
    print('Total count: ', len(dataset))
    for key, value in dataset.class_to_idx.items():
        # custom_dataset.targets == 0 获取索引失败？
        class_num = len([x for x in dataset.imgs if x[1] == value])
        print('Class: {}, count: {}, percent: {:.2%}'.format(
            key, class_num, class_num / len(dataset)))


# 先记录下来，后面再写入文件. 45.6318, 18.3280
def inverse_PM(PMs, PM_mean=45.6318, PM_std=18.3280):
    return PMs * PM_std + PM_mean


def im_segment(input_):
    """ 截取图像中非天空区域 """
    w, h = input_.size
    box = (0, h * 2 // 3, w, h)  # 暂时粗略截
    input_ = input_.crop(box)   
    return input_


def value2class(PMs, pollution={'L0':35, 'L1':70, 'L2':100}):
    classes = np.zeros(len(PMs))
    for i, x in enumerate(PMs):
        if x <= pollution['L0']:
            classes[i] = int(0)
        elif x <= pollution['L1']:
            classes[i] = int(1)
        elif x <= pollution['L2']:
            classes[i] = int(2)
        else:
            classes[i] = int(2)  # 有剩余的也暂时归入最后一类

    return classes


def set_seed(seed):
    """ 需探究这里的 torch, numpy 是否与其他文件一致？ """
    torch.manual_seed(seed)       # cpu 
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True  
    np.random.seed(seed) 
    random.seed(seed)     


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    # plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def show_databatch(inputs, classes):
    inputs = batch_inverse_normalize(inputs)
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=classes)


def batch_inverse_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # represent mean and std to 1, C, 1, ... tensors for broadcasting
    reshape_shape = [1, -1] + ([1] * (len(x.shape) - 2))
    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).reshape(*reshape_shape)
    std = torch.tensor(std, device=x.device, dtype=x.dtype).reshape(*reshape_shape)
    return x * std + mean