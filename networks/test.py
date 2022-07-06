import os, sys
os.chdir(sys.path[0])  # 设置当前工作目录，放再import其他路径模块之前

import time
import torch
import torch.nn as nn
import yaml
from PIL import Image
import numpy as np
from scipy import stats

from networks import get_nets
from datasets import get_transform
from utils import inverse_PM, value2class

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Tester:
    def __init__(self, config, img: Image):
        self.config = config
        self.img = img

    def test(self):
        self._init_params()
        self.model.to(device)
        self.model.train(False)
        self._test_single_img(self.img)

    def _test_single_img(self, input_):
        time_start = time.time()
        PM_list, class_list = [], []
        for i in range(0, self.crop_img_blocks):
            X = self.transform(input_)
            with torch.no_grad():
                X = X.unsqueeze(0)
                X = X.to(device)
                outputs = self.model(X)  # 可处理多个
                PMs = inverse_PM(outputs.numpy().flatten())
                classes = value2class(PMs)

            PM_list.append(PMs[0])
            class_list.append(classes[0])
            print('Predict PM2.5: {}, classes: {}'.format(PMs, classes))

        print('Median PM2.5: {}, class: {}'.format(np.median(PM_list), np.median(class_list)))  # 会有 0.5
        class_mode = stats.mode(class_list)  # PM2.5 是浮点数不好计算众数
        print('Mode class: {}, count: {}/{}'.format(class_mode[0][0], class_mode[1][0], self.crop_img_blocks))

        time_end = time.time()
        print('Single image time cost {:.2f} s'.format(time_end - time_start))

    def _init_params(self):
        self.criterion = nn.MSELoss()  # get_loss 抽象
        self.model = get_nets(self.config['model'])
        self.model.load_state_dict(torch.load(self.config['test']['model_path'], map_location=device)['model'])
        self.transform = get_transform(self.config['image_size'], 'RandomCrop')
        self.crop_img_blocks = self.config['test']['crop_img_blocks']
        
        
if __name__ == '__main__':
    with open(r'config\config.yaml') as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)

    # img_path = r'F:\workplace\public_dataset\Heshan_imgset\morning\1\20191116上午1.jpg'
    img_path = r'D:\workplace\dataset\Heshan_imgset\Heshan_imgset\morning\1\20191116上午1.jpg'
    input_ = Image.open(img_path)
    tester = Tester(config_list, input_)
    tester.test()