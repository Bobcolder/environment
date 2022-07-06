"""
===============================
读取图片的文件头信息并标定主观评分
===============================

每个图片只需要这些信息：拍摄时间、图像大小（4k多x3k多那个分辨率）、
曝光时间、光圈、ISO感光度、焦距、测光模式这几个数据。

"""
import exifread
import glob
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import re


def pd_to_exel(data_df, name='env_data.xlsx' ):
    cols = list(data_df)
    cols.insert(0, cols.pop(cols.index('IMG_ID')))
    data_df = data_df.loc[:, cols]

    writer = pd.ExcelWriter(name)
    data_df.to_excel(writer, float_format='%.5f')
    writer.save()


# 排序，使得同一时间同一方向的图像在一块
def sort_key(item):
    if(re.match(r'.*上午', item)):
        key = 0
    else:
        key = 10
   
    key += int(re.findall(r"(\d)(\(.\))?.jpg", item)[0][0])
    return key


img_addrs = glob.glob(r'F:\workplace\public_dataset\环境数据\*\*.jpg')
img_addrs.sort(key=sort_key)
data_df = pd.DataFrame()
plt.ion()  # 打开交互模式，使用plt，由于直接 im.show() 会卡住进程

for i in range(len(img_addrs)):
    # 阶段性保存时重新进入的节点
    # if i <= 125:
    #     print(img_addrs[i])
    #     continue

    img = exifread.process_file(open(img_addrs[i], 'rb'))
    img_id = img_addrs[i].split('环境数据', 1 )[1]

    keys = list(img.keys())
    values = [str(x) for x in img.values()]
    rowdata = dict(zip(keys, values))
    del rowdata['JPEGThumbnail']
    rowdata['IMG_ID'] = img_id

    # human evaluation
    im_show = Image.open(img_addrs[i])
    plt.figure(img_id)   
    plt.imshow(im_show)
    illumination         = input("光照条件：有日光1 ———— 无日光0  {0,1}：")
    illumination_range   = input("光照好坏：好 5    ———— 坏 0    {0,1,2,3,4,5} ：")
    fog_density          = input("雾浓程度：浓 5    ———— 薄 0    {0,1,2,3,4,5} ：")
    # overall_color  = input("整体色彩：鲜明5   ———— 暗淡0    {0,1,2,3,4,5} ：")
    # definition     = input("清晰度：  清晰5   ———— 模糊0    {0,1,2,3,4,5} ：")
    rowdata['光照条件'] = illumination
    rowdata['光照好坏'] = illumination_range
    rowdata['雾浓程度'] = fog_density
    # rowdata['整体色彩'] = overall_color
    # rowdata['清晰度']   = definition

    plt.close()
    df = pd.DataFrame(rowdata, columns=rowdata.keys(), index=[0])
    data_df = data_df.append(df, ignore_index=True)

    # 阶段性保存
    if i % 5 == 0 and i > 0:
        pd_to_exel(data_df, name = 'env_data_temp' + str(i) + '.xlsx')

pd_to_exel(data_df)