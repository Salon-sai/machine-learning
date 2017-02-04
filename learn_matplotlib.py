# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# 在[-pi , pi]之间取出256个点
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
# 通过该256个变量逐个计算出对应的cos和sin的值
C, S = np.cos(X), np.sin(X)

min_x, max_x = X.min(), X.max()
min_y, max_y = C.min(), C.max()

dx = (max_x - min_x) * 0.02
dy = (max_y - min_y) * 0.02

# 设置颜色，线条粗细，线条形状, 函数标签
plt.plot(X, C, color="blue", linewidth=2, linestyle='-', label='cosine')
plt.plot(X, S, color="red", linewidth=2, linestyle='-', label='sine')
# 函数标签放在 右下方
plt.legend(loc='lower right')

# 设置横坐标的标记号(-pi , -pi/2, 0, pi/2, pi)
plt.xlim(min_x - dx, max_x + dx)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
           [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

# 设置纵坐标的标记号(-1, 0, 1)
plt.ylim(min_y - dy, max_y + dy)
plt.yticks([-1, 0, 1],
           [r'$-1$', r'$0$', r'$+1$'])

# 设置右侧和顶部侧脊柱为透明
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# 设置底部脊柱在x轴的0处 即 x=0
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))

# 设置底部脊柱在y轴的0处 即 y=0
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(("data", 0))

t = 2 * np.pi / 3
plt.plot([t, t], [0, np.cos(t)], color='blue', linewidth=2, linestyle='--')
plt.scatter([t,], [np.cos(t), ], 50, color='blue')

plt.show()