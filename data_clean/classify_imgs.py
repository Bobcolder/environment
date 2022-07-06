"""
根据 PM2.5 值对图像进行文件目录分类
并先剔除部分图片
"""
import os
import re
import shutil


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


def split_by_PM25(datax_dir, pollution):
    """ 根据 PM2.5 归类图片 """
    train_folder = os.path.join(datax_dir, 'train')
    mkdir(train_folder)

    for y in pollution.keys():
        mkdir(os.path.join(train_folder, y))

    for filename in os.listdir(datax_dir):
        if not filename.endswith(('.jpg', '.png')):
            continue
        PM25 = float(re.split('-', filename)[0])  # 根据特定文件名获取 PM 2.5 的值，自行更改
        dst_path = add_to_path(train_folder, pollution, PM25)
        print(os.path.join(datax_dir, filename), "==>", dst_path)
        shutil.move(os.path.join(datax_dir, filename), dst_path)


# 获取误差较大的图片名
def get_remove_file(filename):
    data = []
    f = open(filename, 'r', encoding="utf-8")
    file_data = f.readlines()
    # 去除多余的字符，需自行根据文件名定义
    for i in file_data:
        i = i.replace("\\", "")
        i = i.replace("\n", "")
        data.append(i)
    f.close()
    return data


def to_remove_dir(remove_txt, datax_dir):
    remove_list = get_remove_file(remove_txt)
    remove_folder = os.path.join(datax_dir, 'remove')
    mkdir(remove_folder)
    for filename in os.listdir(datax_dir):
        if not filename.endswith(('.jpg', '.png')):
            continue
        img_name = re.split('-', filename)[1]  # 去掉 PM2.5 的值，需自定义
        if img_name in remove_list:
            dst_path = os.path.join(remove_folder, filename)
            print(os.path.join(datax_dir, filename), "==>", dst_path)
            shutil.move(os.path.join(datax_dir, filename), dst_path)


if __name__ == '__main__':
    pollution = {
        'L0': 35,
        'L1': 70,
        'L2': 100
    }
    datax_dir = r'D:\workplace\dataset\Heshan_imgset\Heshan_imgset\am_pm_123'
    remove_txt = r'..\data\remove_image.txt'

    # 单独归类部分误差较大的图片
    to_remove_dir(remove_txt, datax_dir)

    # 根据 PM2.5 值归类剩余图片
    split_by_PM25(datax_dir, pollution)