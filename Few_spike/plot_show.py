import math
import numpy as np
import torch
from matplotlib import pyplot as plt

from fs_coding import fs_relu, fs_sigmoid

x = np.linspace(-5, 5, 1000, dtype=np.float32)
x = torch.Tensor(x)

y_real_relu = torch.nn.functional.relu(x)
y_real_sigmoid = torch.sigmoid(x)

y_fs_relu = fs_relu(x)
y_fs_sigmoid = fs_sigmoid(x)
# 设置新的坐标轴范围，以x=0为中心左右对称，这里示例设置x轴范围大致涵盖sigmoid函数主要变化区间，可根据实际调整
plt.xlim([-2, 2])
# 设置y轴范围，通常sigmoid函数的值域是(0, 1)，这里适当向两边扩展一点便于观察
plt.ylim([0, 1])
# 设置绘图的尺寸 (宽度, 高度)
# plt.figure(figsize=(4, 5))

plt.plot(x, y_fs_relu, label='ReLU-Spike')
plt.plot(x, y_real_relu, label='ReLU',color='red')

# plt.plot(x, y_fs_sigmoid, label='Sigmoid-Spike')
# plt.plot(x, y_real_sigmoid, label='Sigmoid',color='red')

# 设置横坐标间距小，纵坐标间距大
plt.xticks(np.linspace(-5, 5, 11))  # 横坐标的间距小（增加刻度数）
plt.yticks(np.linspace(0, 6, 10))    # 纵坐标的间距大（减少刻度数）

# 设置标签位置为左上角，并调整字体大小
plt.legend(loc='upper left', fontsize=15)
# 保存图表为SVG格式
plt.savefig('/e/wht_project/new_eeg_data/k_fold10/relu-spike.svg', format='svg')
# plt.savefig('/e/wht_project/new_eeg_data/k_fold10/sigmoid-spike.svg', format='svg')
plt.show()
