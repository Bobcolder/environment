import time
import torch
import torch.nn as nn
import yaml
from PIL import Image
import numpy as np
from scipy import stats
import re
import torchvision.transforms as transforms

import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
print(sys.path)  # 必要时检查，有时要进入脚本所在目录运行

from networks import get_nets

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Tester:
    def __init__(self, config, img: Image):
        self.config = config
        self.img = img  # config, img: Image

    def test(self):
        self._init_params()
        self.model.to(device)
        self.model.train(False)
        self._test_single_img(self.img)

    def _test_single_img(self, input_):
        time_start = time.time()
        Xs = self.randomCrop(input_)
        Xs = Xs.unsqueeze(0)
        # 上面两句代码随机抽取一个块，下面再抽取剩下的
        for i in range(1, self.crop_img_blocks):
            X = self.randomCrop(input_)
            Xs = torch.cat((Xs, X.unsqueeze(0)), 0)

        with torch.no_grad():
            Xs = Xs.to(device)
            outputs = self.model(Xs)  # 可处理多个
            _, preds = torch.max(outputs.data, 1)

        class_list = preds.cpu().data.numpy()
        print('Median class: {}'.format(np.median(class_list)))
        class_mode = stats.mode(class_list)
        print('Mode class: {}, count: {}/{}'.format(class_mode[0][0], class_mode[1][0], self.crop_img_blocks))

        time_end = time.time()
        print('Single image time cost {:.2f} s'.format(time_end - time_start))

    def _init_params(self):
        self.criterion = nn.MSELoss()  # get_loss 抽象
        self.model = get_nets(self.config['model']['g_name'], self.config['model']['out_features'])
        self.model.load_state_dict(torch.load(self.config['test']['model_path'], map_location=device)['model'])
        self.randomCrop = transforms.RandomCrop(self.config['image_size'])
        self.crop_img_blocks = self.config['test']['crop_img_blocks']


class SkySegment:

    def __init__(self):
        pass

    def process(self, img, name):
        img = self.remove_sky(img, name)
        w, h = img.size
        if h < 128:
            img = self.scale_height(img, 180)  # bug
        return img

    @staticmethod
    def remove_sky(im, filename):
        # 1角度去掉原图的2/3 , 2和3角度去掉原图的2/3
        angleDic = {1: 3 / 4, 2: 2 / 3, 3: 2 / 3}
        width = im.size[0]
        height = im.size[1]
        angle_type = int(re.findall(r'.*(\d)\.*', filename)[0])
        im = im.crop((0, height * angleDic[angle_type], width, height))
        return im

    @staticmethod
    def scale_height(img, target_height):
        ow, oh = img.size
        if (oh == target_height):
            return img
        h = target_height
        w = int(target_height * ow / oh)
        return img.resize((w, h), Image.BICUBIC)


if __name__ == '__main__':
    with open(r'networks\config\config.yaml') as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)
        data_dir = config_list['nonsky_dir']
        imgsize = config_list['image_size']

    # img_path = r'F:\workplace\public_dataset\Heshan_imgset\morning\1\20191116上午1.jpg'
    img_path = r'D:\workplace\dataset\Heshan_imgset\Heshan_imgset\am_pm_123\filtering\test\L0\14.2-20191007上午2.jpg'
    img = Image.open(img_path)

    # 需要先分割天空区域，做归一化
    imgname = re.split(r'[\\/]', img_path)[-1]
    segSky = SkySegment()
    img_seg = segSky.process(img, imgname)

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    input_ = test_transform(img_seg)

    tester = Tester(config_list, input_)
    tester.test()

