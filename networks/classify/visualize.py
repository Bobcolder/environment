import torch
from torchvision.datasets import ImageFolder
import yaml
from torch.utils.data import DataLoader, random_split

import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
print(sys.path)  # 必要时检查，有时要进入脚本所在目录运行

from utils import show_databatch
from datasets import MyImageFolder, get_transform
from networks import get_nets

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def visualize_model(vgg, dataloader, out=16):
    """预测结果可视化

    参数说明：
        dataloaders: 载入测试数据
        out: 取多少个 batch
    """
    vgg.train(False)
    for i, data in enumerate(dataloader):
        inputs, labels, info = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        # predicted_labels = [preds[j] for j in range(inputs.size()[0])]

        print("Ground truth:", labels)
        print("Prediction:  ", preds)
        error_idx = (labels != preds).nonzero(as_tuple=True)[0]
        if len(error_idx) > 0:
            show_databatch(inputs, labels)
            show_databatch(inputs, preds)
        names, PMs = info[0], info[1]
        for idx in error_idx:
            print('Error idx: ', idx)
            print('\tLabels: ', labels[idx], ' ==> ', preds[idx])
            print('\tPM2.5: ', PMs[idx], ' & ', names[idx])
            pass
        
        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

        if i > out: break


if __name__ == "__main__":
    with open('../config/config.yaml', 'r') as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)
        data_dir = config_list['nonsky_dir']
        imgsize = config_list['image_size']
        val_trans_first = config_list['val']['transform_first']

    valid_dir = os.path.join(data_dir, "val")
    valid_transform = get_transform(imgsize, val_trans_first)  # 验证集不一定跟训练集的transform一致
    valid_dataset = MyImageFolder(valid_dir, transform=valid_transform)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=8, shuffle=True)

    model = get_nets(config_list['model']['g_name'], config_list['model']['out_features'])
    model.load_state_dict(torch.load(config_list['test']['model_path'], map_location=device)['model'])
    visualize_model(model, valid_loader, out=1000)


