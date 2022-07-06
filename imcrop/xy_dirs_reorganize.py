"""
============================
天空/非天空图像块和标签分开存储
============================

需要在 Results/ 目录下包含所有的原始图像块和单个.csv标签文件

TODO: 也可以把这个程序合并到 mk_dataset.py 里面

"""
import os
import re
import shutil
import pandas as pd
from deprecated.sphinx import deprecated
import warnings

warnings.filterwarnings("always")


def mkdir(floder):
    if not os.path.exists(floder):
        os.makedirs(floder)


# 将 Results/ 下的原始图像块归入 sky/ 和 non_sky/ 文件夹
def standardize_data_folder(patch_dir):
    sky_folder = os.path.join(patch_dir, 'sky') 
    nonsky_folder = os.path.join(patch_dir, 'non_sky') 
    mkdir(sky_folder)
    mkdir(nonsky_folder)
    skylabels_folder = os.path.join(patch_dir, 'sky_labels')
    nonsky_labels_folder = os.path.join(patch_dir, 'non_sky_labels')
    mkdir(skylabels_folder)
    mkdir(nonsky_labels_folder)

    for filename in os.listdir(patch_dir):
        if not filename.endswith(('.bmp', '.csv')):
            continue
        is_sky = re.split('[-]', filename)[-2]  # 0 表示天空，1 表示其他
        if filename.endswith('.bmp'):
            if '0' == is_sky:
                shutil.move(os.path.join(patch_dir, filename), sky_folder)
                print(filename, " ==> ", sky_folder)
            else:
                shutil.move(os.path.join(patch_dir, filename), nonsky_folder)
                print(filename, " ==> ", nonsky_folder)
        elif filename.endswith('csv'):
            if '0' == is_sky:
                shutil.move(os.path.join(patch_dir, filename), skylabels_folder)
                print(filename, " ==> ", skylabels_folder)
            else:
                shutil.move(os.path.join(patch_dir, filename), nonsky_labels_folder)
                print(filename, " ==> ", nonsky_labels_folder)
    

@deprecated(version='1.0', reason="This function will be removed soon")
def standardize_label_folder(patch_dir):
    """将 excel 里的标签切分为单个文本文件

    'crop_labels.xlsx' 默认保存在当前文件夹，由 mk_dataset.py 处理所得
    """
    skylabel_folder = os.path.join(patch_dir, 'sky_labels') 
    nonsky_labels_folder = os.path.join(patch_dir, 'non_sky_labels') 
    mkdir(skylabel_folder)
    mkdir(nonsky_labels_folder)

    pd_labels = pd.read_excel('crop_labels.xlsx', index_col=[0])
    for i in range(len(pd_labels)):
        row_data = pd_labels.iloc[i]
        img_id = row_data['IMG_ID']
        is_sky = re.split('[-]', img_id)[-2]
        # 0 表示天空，1 表示其他
        if '0' == is_sky:
            filename = '{}.csv'.format(os.path.join(skylabel_folder, img_id))
            row_data.to_csv(filename, index=True)
            print(img_id, " ==> ", skylabel_folder)
        else:
            filename = '{}.csv'.format(os.path.join(nonsky_labels_folder, img_id))
            row_data.to_csv(filename, index=True)
            print(img_id, " ==> ", nonsky_labels_folder)


if __name__ == '__main__':
    standardize_data_folder(r'D:\workplace\dataset\Heshan_imgset\Heshan_imgset\Results')