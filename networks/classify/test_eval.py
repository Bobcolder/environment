import time
import torch
import torch.nn as nn
import yaml
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
print(sys.path)  # 必要时检查，有时要进入脚本所在目录运行

from networks import get_nets
from datasets import SegImageFolder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Tester:
    def __init__(self, config, dataLoader: DataLoader):
        self.config = config
        # self.img = img  # config, img: Image
        self.dataLoader = dataLoader

    def test(self):
        self._init_params()
        self.model.to(device)
        self.model.train(False)
        correct_all, cnt_all = 0, 0
        time_start = time.time()
        for i, data in enumerate(self.dataLoader):
            inputs, labels, names = data

            vote_pred = [0 for x in range(len(inputs))]  # 记录每张大图的投票预测值
            for j, input_ in enumerate(inputs):
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
                class_mode = stats.mode(class_list)  # 记录投票的众数
                print('Mode class: {}, count: {}/{}'.format(class_mode[0][0], class_mode[1][0], self.crop_img_blocks))
                print('True label & name: ', names[j])
                vote_pred[j] = class_mode[0][0]

            correct = np.sum(np.array(vote_pred) == labels.cpu().data.numpy())
            correct_all += correct
            cnt_all += labels.size(0)

        time_end = time.time()
        print('time cost', (time_end - time_start) / len(test_dataset), 's')  # 这里默认 batch=1
        acc = correct_all * 1. / cnt_all
        print('Acc: ', acc)

    def _init_params(self):
        self.criterion = nn.MSELoss()  # get_loss 抽象
        self.model = get_nets(self.config['model']['g_name'], self.config['model']['out_features'])
        self.model.load_state_dict(torch.load(self.config['test']['model_path'], map_location=device)['model'])
        self.randomCrop = transforms.RandomCrop(self.config['image_size'])
        self.crop_img_blocks = self.config['test']['crop_img_blocks']
        

if __name__ == '__main__':
    with open(r'networks\config\config.yaml') as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)
        data_dir = config_list['nonsky_dir']
        imgsize = config_list['image_size']

    test_dir = os.path.join(data_dir, 'test')
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    test_dataset = SegImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    tester = Tester(config_list, dataLoader=test_loader)
    tester.test()