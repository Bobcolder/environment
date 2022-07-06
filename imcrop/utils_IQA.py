import os
from sklearn.metrics.cluster import entropy
from skimage.measure import shannon_entropy
from skimage import io, color, img_as_ubyte
from matlab.AG import AG
from matlab.FADE import FADE


def get_FADE(rgbImg):
    """Fog Aware Density Evaluator (FADE)

    paper: L. K. Choi, J. You, and A. C. Bovik,
        "Referenceless Prediction of Perceptual Fog Density and Perceptual Image Defogging,"
        IEEE Transactions on Image Processing, vol. 24, no. 11
    site: http://live.ece.utexas.edu/research/fog/index.html
    """
    D, D_map = FADE(rgbImg)
    return D


def get_AG(rgbImg):
    grayImg = img_as_ubyte(color.rgb2gray(rgbImg))
    r = AG(grayImg)
    return r


def get_IE(rgbImg):
    """Calculate the Shannon entropy of an image.

    The Shannon entropy is defined as S = -sum(pk * log(pk)), where pk are frequency/probability of pixels of value k.
    """
    grayImg = img_as_ubyte(color.rgb2gray(rgbImg))
    # r = entropy(grayImg)
    r = shannon_entropy(grayImg)
    return r


if __name__ == '__main__':
    img_dir = r'D:\workplace\dataset\Heshan_imgset\Heshan_imgset\morning\1'
    img_list = [
        os.path.join(img_dir, '20190927上午1.jpg'),
        os.path.join(img_dir, '20190928上午1.jpg'),
        os.path.join(img_dir, '20190929上午1.jpg'),
        os.path.join(img_dir, '20190930上午1.jpg'),
        os.path.join(img_dir, '20191001上午1.jpg')
    ]
    # test IE
    for file in img_list:
        img = io.imread(file)
        # print((get_IE(img)))
        # print((get_AG(img)))
        print((get_FADE(img)))