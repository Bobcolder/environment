"""
==============================
计算图片主/客观指标与PM2.5的相关系数
==============================
分开不同方向的图片计算，如1角度（上|下午）

Pearson 相关系数只度量线性关系。Spearman 相关系数只度量单调关系。
因此，即使相关系数为 0，也可能存在有意义的关系。检查散点图可确定关系的形式。

"""
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats
import sys
sys.path.append('../')

matplotlib.rcParams['font.sans-serif'] = ['Source Han Sans TW', 'sans-serif']

# 9月26日无上午图片，10月26日无下午图片
dataframe = pd.read_excel(r'..\..\excel\temp_all.xlsx')
IQA_list = ['BIQME', 'FADE', 'AG', 'IE', '清晰度']
angle_list = ['上午1|下午1', '上午2|下午2', '上午3|下午3']

for IQA in IQA_list:
    print('IQA: {} {}'.format(IQA, '-' * 20))
    df_temp = dataframe[['IMG_ID', 'PM2.5', IQA]]
    for angle in angle_list:
        print('\tAngle: {} {}'.format(angle, '-' * 10))
        am_pm = df_temp[df_temp['IMG_ID'].str.contains(angle)]
        am_pm = am_pm.sort_values(by=['IMG_ID'])
        # print(am_pm.head())
        am_pm.plot(x='IMG_ID', y='PM2.5')  # 按时间排序
        plt.show()
        arr = np.array(am_pm[[IQA, 'PM2.5']])
        corr, p_val = stats.spearmanr(arr)
        print('\tcorr=%f, p_val=%f' % (corr, p_val))

plt.close('all')  # 关闭所有 figure windows