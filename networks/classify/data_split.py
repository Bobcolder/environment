"""划分训练、验证集

并按照 datasets.ImageFolder 的方式组织目录
"""
import os
import re
import shutil
import yaml
import glob
import random

random.seed(88)

# 仅使用取整后的 PM2.5 值
with open('networks/config/config.yaml') as file:
    config_list = yaml.load(file, Loader=yaml.FullLoader)
    datax_dir = config_list['nonsky_dir']
    split_fig = config_list['data_split']
    split_date = split_fig['split_date']
    split_isback = split_fig['split_back']
    val_ratio = split_fig['val_ratio']
    test_ratio = split_fig['test_ratio']

pollution = {
    'L0': 35,
    'L1': 70,
    'L2': 100
}


def mkdir(floder):
    if not os.path.exists(floder):
        os.makedirs(floder)


def add_to_path(dst_path, pollution, PM25):
    """ 在目标路径之后追加标签目录 """
    labels = list(pollution.keys())
    if PM25 <= pollution[labels[0]]: 
        dst_path = os.path.join(dst_path, labels[0])
    elif PM25 <= pollution[labels[1]]:
        dst_path = os.path.join(dst_path, labels[1])
    elif PM25 <= pollution[labels[2]]:
        dst_path = os.path.join(dst_path, labels[2])
    else:
        dst_path = os.path.join(dst_path, labels[2])  # 有剩余的也暂时归入最后一类
    return dst_path


def split_train_val(split_date=1107):
    """ 11月?号前的作为训练集，往后的作为验证集 """
    train_folder = os.path.join(datax_dir, 'train')
    val_folder = os.path.join(datax_dir, 'val') 
    mkdir(train_folder)
    mkdir(val_folder)

    for y in pollution.keys():
        mkdir(os.path.join(train_folder, y))
        mkdir(os.path.join(val_folder, y))

    for filename in os.listdir(datax_dir):
        if not filename.endswith('.bmp'):
            continue
        date = int(filename[:4])
        PM25 = int(re.split('[-.]', filename)[-2])
        if date < split_date:
            dst_path = add_to_path(train_folder, pollution, PM25)
            shutil.move(os.path.join(datax_dir, filename), dst_path)
            print(os.path.join(datax_dir, filename), "==>", dst_path)
        else:
            dst_path = add_to_path(val_folder, pollution, PM25)
            shutil.move(os.path.join(datax_dir, filename), dst_path)
            print(os.path.join(datax_dir, filename), "==>", dst_path)


def split_train_val_random(val_ratio=0.1, test_ratio=0.2):
    """ 随机划分训练集，验证集 """
    train_folder = os.path.join(datax_dir, 'train')
    val_folder = os.path.join(datax_dir, 'val')
    test_folder = os.path.join(datax_dir, 'test')
    mkdir(train_folder)
    mkdir(val_folder)
    mkdir(test_folder)

    for y in pollution.keys():
        mkdir(os.path.join(train_folder, y))
        mkdir(os.path.join(val_folder, y))
        mkdir(os.path.join(test_folder, y))

    for filename in os.listdir(datax_dir):
        if not filename.endswith(('.bmp', 'png', 'jpg')):
            continue
        # PM25 = int(re.split('[-.]', filename)[-2])
        PM25 = float(re.split('[-]', filename)[0])
        odds = random.random()
        if odds < 1 - val_ratio - test_ratio:
            dst_path = add_to_path(train_folder, pollution, PM25)
        elif odds < 1 - test_ratio:
            dst_path = add_to_path(val_folder, pollution, PM25)
        else:
            dst_path = add_to_path(test_folder, pollution, PM25)

        shutil.move(os.path.join(datax_dir, filename), dst_path)
        print(os.path.join(datax_dir, filename), "==>", dst_path)


def back_split():
    """ 回滚数据集划分，将图块重归至一个文件夹 """
    paths_mask = datax_dir + '/*/*/*'  # *.bmp
    img_paths = glob.glob(paths_mask)
    for img_path in img_paths:
        shutil.move(img_path, datax_dir)
        print(img_path, "==>", datax_dir)


if __name__ == '__main__':
    if not split_isback:
        split_train_val_random(val_ratio, test_ratio)
    else:
        back_split()




