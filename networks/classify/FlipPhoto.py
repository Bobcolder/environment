"""图像增广
需先分好 0/1/2 三类文件夹
将 0 类变为原来的 2 倍， 2 类变为原来的 6 倍
"""
import cv2
import os
import numpy as np


def imread(path_img):
    """解决中文路径问题 """
    I = cv2.imdecode(np.fromfile(file=path_img, dtype=np.uint8), cv2.IMREAD_COLOR)
    return I


def imwrite(dst, image):
    """解决中文路径问题 """
    cv2.imencode(ext='.bmp', img=image)[1].tofile(dst)


def rotate_img(degree, img):
    """
    degree: 角度(int)
    """
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]

    # 定义一个旋转矩阵
    matRotate = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), degree, 1)  # mat rotate 1 center 2 angle 3 缩放系数
    new_img = cv2.warpAffine(img, matRotate, (width, height))

    return new_img


def main_func(data_dir):
    for root, dirs, _ in os.walk(data_dir):
        # 遍历类别
        for sub_dir in dirs:
            img_names = os.listdir(os.path.join(root, sub_dir))
            # img_names = list(filter(lambda x: x.endswith('.bmp'), img_names))  # 这一步耗时？

            # 遍历图片
            for i in range(len(img_names)):
                img_name = img_names[i]
                filter_list = ['rotate', 'flip']  # 避免重复增强图块
                if not img_name.endswith(('.bmp', 'jpg', 'png')) \
                        or any(key in img_name for key in filter_list):
                    continue

                path_img = os.path.join(root, sub_dir, img_name)
                grade = sub_dir

                # 分类 0 只需要翻转 2倍
                if 'L0' == grade:
                    continue
                    img = imread(path_img)
                    horizontal_img = cv2.flip(img, 1)
                    # new_img_name = img_name.replace(".bmp", "") + "-flip" + ".bmp"
                    new_img_name = 'flip-{}'.format(img_name)
                    path_new_img = os.path.join(root, sub_dir, new_img_name)
                    print(path_new_img)
                    imwrite(path_new_img, horizontal_img)

                # 分类 2 翻转后 + rotate1.5 + rotate-1.5
                elif 'L2' == grade:
                    img = imread(path_img)
                    horizontal_img = cv2.flip(img, 1)
                    # new_img_name = img_name.replace(".bmp", "") + "-flip" + ".bmp"
                    new_img_name = 'flip-{}'.format(img_name)
                    path_new_img = os.path.join(root, sub_dir, new_img_name)
                    print(path_new_img)
                    imwrite(path_new_img, horizontal_img)

                    degrees = [-1, 1, -1.5, 1.5, -2, 2]
                    for angle in degrees:
                        r_img = rotate_img(angle, img)
                        r_new_name = 'rotate{}-{}'.format(str(angle), img_name)
                        r_path_new = os.path.join(root, sub_dir, r_new_name)
                        imwrite(r_path_new, r_img)


if __name__ == '__main__':
    img_folder = r'D:\workplace\dataset\Heshan_imgset\Heshan_imgset\am_pm_123\filtering\train'
    # img_folder = sys.argv[1]
    print('该目录下需包含 3类 子文件（L0/L1/L2），可以嵌套：', img_folder)  # 命令行传入文件路径
    main_func(img_folder)
