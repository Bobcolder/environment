import os
import sys 
sys.path.append(os.getcwd()) 
from imcrop.mk_dataset import ImageCropper
from PIL import Image, ImageDraw
import random


def test_im_crop():
    im = Image.open('data/20190927am1.jpg')
    imCropper = ImageCropper(im)
    patches, boxes = imCropper.crop(box_w=512, box_h=512, stride_w=512, stride_h=512)
    draw = ImageDraw.Draw(im)
    for box in boxes:
        color = (random.randint(64,255), random.randint(64,255), random.randint(64,255))
        draw.rectangle(box, outline=color, width=5)
        # im.show()
    im.show()


test_im_crop()