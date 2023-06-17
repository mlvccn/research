

from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import random
from matplotlib.pyplot import MultipleLocator


front_type = 'Times New Roman'
label_size = 40
tick_size = 35
text_size = 33


models = ['MFNet', 'RTFNet', 'PSTNet', 'FuseSeg','EGFNet', 'MTANet', 'Ours(R50-18)','Ours(D161-121)', 'Ours(R101-34)']
mIoUs = [39.7, 53.2, 48.6, 54.5, 54.8, 56.1, 55.1, 55.5, 56.2]
flops= [8.42, 336.69, 163.64, 193.4, 201.09, 264.96, 64.69, 95.02, 98.82]
# fpss = [205.47, 33.86, 102.82, 18.36, 81.36, 34.15, 57.31]
fpss = [137.39, 14.63, 64.76, 16.88, 8.12, 8.40, 56.42, 14.89, 32.08]

fig, axes = plt.subplots(1, 1, figsize=(12, 8))

# 设置最小刻度间隔
# axes.yaxis.set_minor_locator(MultipleLocator(3))
# axes.xaxis.set_major_locator(MultipleLocator(1))
#axes.xaxis.set_minor_locator(MultipleLocator(1))
# 画网格线
# axes.grid(which='minor', c='lightgrey')
# 设置x、y轴标签
axes.set_ylabel("mIoU (%)", size=label_size, fontproperties=front_type)
axes.set_xlabel("FLOPs (G)", size=label_size, fontproperties=front_type)

# 自定义刻度线方向
axes.tick_params(axis='x', which='both', direction='in')
axes.tick_params(axis='y', which='both', direction='in')

# 设置y轴的刻度
# axes.set_yticks([30, 35, 40, 45, 50, 55, 60])

plt.yticks(fontproperties=front_type, size=tick_size)
plt.xticks(fontproperties=front_type, size=tick_size)
plt.grid(linestyle='--', linewidth=1.2)  # 设置网格模式


for model, flop, mIoU, fps in zip(models, flops, mIoUs, fpss):
    # 画出 fps 的圆圈，圆圈越大，fps越大
    if 'Ours' in model:
        sca1 = plt.scatter(x=flop, y=mIoU, s=fps*30, color='#F76CB1')
    else:
        sca2 = plt.scatter(x=flop, y=mIoU, s=fps*30, color='#86CDF9')
    
    # 在圆圈旁边标记模型名称
    if model == 'MFNet':
        axes.text(flop+14, mIoU+0.3, model, size=text_size,fontproperties=front_type)
    elif model == 'RTFNet':
        axes.text(flop-44, mIoU-1.2, model, size=text_size,fontproperties=front_type)
    elif model == 'PSTNet':
        axes.text(flop+12, mIoU-0.5, model, size=text_size,fontproperties=front_type)
    elif model == 'FuseSeg':
        axes.text(flop-40, mIoU-1.2, model, size=text_size,fontproperties=front_type)
    elif model == 'EGFNet':
        axes.text(flop+6, mIoU-0.8, model, size=text_size,fontproperties=front_type)
    elif model == 'MTANet':
        axes.text(flop+6, mIoU-0.4, model, size=text_size,fontproperties=front_type)
    elif model == 'Ours(R50-18)':
        axes.text(flop-50, mIoU-1.8, model, size=text_size,fontproperties=front_type)
    elif model == 'Ours(D161-121)':
        axes.text(flop+10, mIoU-0.3, model, size=text_size,fontproperties=front_type)
    elif model == 'Ours(R101-34)':
        axes.text(flop+6, mIoU+0.5, model, size=text_size,fontproperties=front_type)

# (x,y) 散点图
axes.scatter(x=flops, y=mIoUs, s=10, color='blue', marker='o')

# set_yticks 要放在 axes.scatter之后使用才有效果
# axes.set_yticks((38.0, 40.0, 45.0, 50.0, 55.0, 60.0))
axes.set_yticks([38, 40, 45, 50, 55, 58])
axes.set_yticklabels(['', '40', '45', '50', '55', ''])


# legend 字体设置
# labelss=plt.legend(handles=[sca1], labels=["FPS"], loc="lower right",fontsize=text_size, edgecolor='black').get_texts()
# [label.set_fontname('Times New Roman') for label in labelss]


# 展示图片
plt.tight_layout(pad=0.2)
plt.savefig('drawing.png')
# plt.savefig('fps.pdf')



