'''
网格法计算面积 - 根据RGB颜色着色
'''
import random
import numpy as np
import pymeshlab as ml
import matplotlib.pyplot as plt
from matplotlib import cm

# 加载网格数据
data = '../data/srm.ply'
ms = ml.MeshSet()
ms.load_new_mesh(data)
ms.set_current_mesh(0)
vertices = ms.current_mesh().vertex_matrix()
rgb = ms.current_mesh().vertex_color_matrix()
# 随机采样10%的点
num_samples = int(len(vertices))
random_indices = random.sample(range(len(vertices)), num_samples)
sampled_vertices = vertices[random_indices]

# 获取坐标
x, y, z = sampled_vertices[:, 0], sampled_vertices[:, 1], sampled_vertices[:, 2]
rgbx, rgby, rgbz = rgb[:, 0], rgb[:, 1], rgb[:, 2]

'''
坐标转换
'''
zmin = z[np.argmin(z)]
zmax = z[np.argmax(z)]
depthz = (zmax - zmin) / 0.36 * 4.5
z = depthz * (z) / (zmax - zmin)

xmin = x[np.argmin(x)]
xmax = x[np.argmax(x)]
depthx = (xmax - xmin) / 0.36 * 4.5
x = depthx * (x) / (xmax - xmin)

ymin = y[np.argmin(y)]
ymax = y[np.argmax(y)]
depthy = (ymax - ymin) / 0.36 * 4.5
y = depthy * (y) / (ymax - ymin)

# 创建RGB颜色数组
# 假设RGB值在0-255范围内，需要归一化到0-1
colors = np.column_stack((rgbx, rgby, rgbz))
# 如果RGB值大于1，则归一化到0-1范围
if colors.max() > 1.0:
    colors = colors / 255.0
# 确保值在有效范围内
colors = np.clip(colors, 0, 1)

# 创建图形
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=26)

# 使用RGB值设置颜色
scatter = ax.scatter(
    x, y, z,
    c=colors,  # 使用RGB颜色数组
    s=0.01,  # 点的大小
    alpha=0.7  # 透明度
)

# 设置坐标轴标签和范围
ax.set_xlabel('X axis', fontdict={'family': 'Times New Roman', 'size': 20,})
ax.set_ylabel('Y axis', fontdict={'family': 'Times New Roman', 'size': 20,})
ax.set_zlabel('Depth(cm)', fontdict={'family': 'Times New Roman', 'size': 20,})
ax.set_zlim(-50, 50,)
ax.set_ylim(-50, 50)
ax.set_xlim(50, -50)
# 调整刻度标签的字体样式
tick_font = {
    'family': 'Times New Roman',
    'size': 16,
    'weight': 'normal',
    'style': 'normal'
}

# # 方法1：使用tick_params设置
# ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])

# 方法2：单独设置每个轴的刻度字体
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    # 设置刻度标签字体
    axis.set_tick_params(labelsize=tick_font['size'])

    # 获取当前刻度
    current_ticks = axis.get_ticklabels()

    # 设置每个刻度标签的字体属性
    for tick in current_ticks:
        tick.set_fontname(tick_font['family'])
        tick.set_fontsize(tick_font['size'])
        tick.set_fontweight(tick_font['weight'])
        tick.set_fontstyle(tick_font['style'])

# 自动调整视图角度以获得更好的3D效果
ax.view_init(elev=15, azim=-45)

plt.tight_layout()
plt.show()