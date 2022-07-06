"""
==========================
直接在分好类的大图中切块
==========================
"""
import glob
import os
import numpy as np
import re
import sys
from PIL import Image


class ImageCropper():

    def __init__(self, box_w=256, box_h=256, stride_w=256, stride_h=256, epsilon=10):
        self.box_w = box_w
        self.box_h = box_h
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.epsilon = epsilon

    def crop(self, im):
        """Crop image to get patches.

        :param epsilon: 右下方边界容忍值，低于之则直接丢弃
        :return: 返回截取的 patches 以及其对应于原图的坐标
        """
        box_w = self.box_w
        box_h = self.box_h
        stride_w = self.stride_w
        stride_h = self.stride_h
        epsilon = self.epsilon

        width = im.size[0]
        height = im.size[1]
        if width < box_w or height < box_h:
            return

        patches, patches_idx = [], []
        iw = np.arange(0, width - box_w + 1, stride_w)
        jh = np.arange(0, height - box_h + 1, stride_h)
        for i in iw:
            for j in jh:
                box = (i, j, i + box_w, j + box_h)
                cm = im.crop(box)
                patches.append(cm)
                patches_idx.append(box)
        # repair x and y orientation's boundary
        if width - box_w - iw[-1] > epsilon:
            for j in jh:
                box = (width - box_w, j, width, j + box_h)
                cm = im.crop(box)
                patches.append(cm)
                patches_idx.append(box)
        if height - box_h - jh[-1] > epsilon:
            for i in iw:
                box = (i, height - box_h, i + box_w, height)
                cm = im.crop(box)
                patches.append(cm)
                patches_idx.append(box)
        # need only once
        if width - box_w - iw[-1] > epsilon and height - box_h - jh[-1] > epsilon:
            box = (width - box_w, height - box_h, width, height)
            cm = im.crop(box)
            patches.append(cm)
            patches_idx.append(box)

        return patches, patches_idx


def remove_sky(im, filename):
    """在切割图像之前去除天空区域

    :param im: PIL.Image 原图像
           filename: 图片名
    :return: 返回切除天空的图像
    """
    # 1角度去掉原图的2/3 , 2和3角度去掉原图的2/3
    angleDic = {1: 3 / 4, 2: 2 / 3, 3: 2 / 3}
    width = im.size[0]
    height = im.size[1]
    angle_type = int(re.findall(r'.*(\d)\.*', filename)[0])
    im = im.crop((0, height * angleDic[angle_type], width, height))
    return im


def mk_dataset(Heshan_imgset, imCropper):
    """ 制作数据集

    Params:
        Heshan_imgset: 原始图像路径
        imCropper: 图像切块方式
    Output:
        同目录下的 blocks: L0/L1/L2
    """
    types = ('*.jpg', '*.png', '*.jpeg')
    img_paths = []
    for files in types:
        paths_mask = Heshan_imgset + '/*/' + files
        img_paths.extend(glob.glob(paths_mask))

    result_folder = os.path.join(Heshan_imgset, 'blocks')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    for i, filename in enumerate(img_paths):
        im = Image.open(filename)
        im = remove_sky(im, filename)  # 切除整个天空区域
        patches, boxes = imCropper.crop(im)
        re_list = re.split(r'[\\/]', filename)
        imgname = re_list[-1]
        sub_folder = re_list[-2]

        sub_dir = os.path.join(result_folder, sub_folder)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        for i in range(len(patches)):
            im_patch = patches[i]
            box = boxes[i]
            ref_origin = "({},{})".format(box[0], box[1])
            patch_name = '-'.join([ref_origin, imgname]) + '.bmp'
            print(os.path.join(result_folder, sub_folder, patch_name))
            im_patch.save(os.path.join(result_folder, sub_folder, patch_name), 'bmp')


if __name__ == '__main__':
    Heshan_imgset = sys.argv[1]
    # 该目录下分好 L0/L1/L2 类别的大图
    # Heshan_imgset = r'D:\workplace\dataset\Heshan_imgset\Heshan_imgset\am_pm_123\filtering\train'
    imCropper = ImageCropper(box_w=256, box_h=256, stride_w=256, stride_h=256)
    mk_dataset(Heshan_imgset, imCropper)

