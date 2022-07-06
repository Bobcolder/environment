"""
先手动将所有角度上下午的大图放至一个文件夹
此处给它们加上 PM2.5 的信息
"""
import os
import pandas as pd

img_dir = r'D:\workplace\dataset\Heshan_imgset\Heshan_imgset\am_pm_123'

df = pd.read_excel('big_images_labels.xlsx')
for i, row in df.iterrows():
    img_name = row['IMG_ID']
    PM25 = format(row['PM2.5'], '.1f')
    new_name = str(PM25) + '-' + img_name
    print(img_name, ' ==> ', new_name)
    os.rename(os.path.join(img_dir, img_name), os.path.join(img_dir, new_name))