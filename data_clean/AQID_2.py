""" 模仿 AQID 数据集，裁剪鹤山数据集"""
import os
import re
from PIL import Image


def remove_origin_sky(im, filename):
    """去除原始图像部分天空区域，使得天空区域占1/3~1/2

    :param im: PIL.Image 原图像
           filename: 图片名
    :return: 返回切除天空的图像
    """
    # 1角度去掉原图的2/3 , 2和3角度去掉原图的2/3 ?
    angleDic = {1: 3/5, 2: 1/2, 3: 3/5}
    width, height = im.size
    angle_type = int(re.findall(r'.*(\d)\.*', filename)[0])
    im = im.crop((0, height * angleDic[angle_type], width, height))
    return im


def process_sky_cut(img_dir):
    """ 批量处理天空区域去除 """
    names = img_dir.rsplit('\\', 1)
    top_dir, dir_name = names
    dst_dir = os.path.join(top_dir, '{}_sky_cut'.format(dir_name))
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for filename in os.listdir(datax_dir):
        if not filename.endswith(('.jpg', '.png', 'jpeg')):
            continue
        src = Image.open(os.path.join(datax_dir, filename))
        img = remove_origin_sky(src, filename)
        # img.show()
        bmp_name = filename.rsplit('.', 1)[0] + '.bmp'  # 去掉 .jpg 后缀
        img.save(os.path.join(dst_dir, bmp_name), 'bmp')
        print('{}.bmp'.format(os.path.join(dst_dir, bmp_name)))


def print_dataset_info(datax_dir):
    """输出数据集的信息，包括图像数目，大小等"""
    img_cnt = 0
    im_sizes = set()

    for filename in os.listdir(datax_dir):
        if not filename.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            continue
        im = Image.open(os.path.join(datax_dir, filename))
        img_cnt = img_cnt + 1
        im_sizes.add(im.size)

    print("Total num of images: ", img_cnt)
    print("All kinds of image size: ", im_sizes)


if __name__ == '__main__':
    datax_dir = r'D:\workplace\dataset\Heshan_imgset\Heshan_imgset\AQID_2\PM_imgs'
    print_dataset_info(datax_dir)  # 打印数据集信息
    # process_sky_cut(datax_dir)     # 批量去除部分天空区域

    sky_cut_dir = r'D:\workplace\dataset\Heshan_imgset\Heshan_imgset\AQID_2\PM_imgs_sky_cut'
    print_dataset_info(sky_cut_dir)

