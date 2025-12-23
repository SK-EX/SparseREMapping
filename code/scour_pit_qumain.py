'''
网格法计算面积 - 根据高程着色
'''
import random
import numpy as np
import pymeshlab as ml
import matplotlib.pyplot as plt
from matplotlib import cm

# 加载网格数据
data = 'scour_pit_red.ply'
ms = ml.MeshSet()
ms.load_new_mesh(data)
ms.set_current_mesh(0)
vertices = ms.current_mesh().vertex_matrix()

# 随机采样10%的点
num_samples = int(len(vertices))
random_indices = random.sample(range(len(vertices)), num_samples)
sampled_vertices = vertices[random_indices]

# 获取坐标
x, y, z = sampled_vertices[:, 0], sampled_vertices[:, 1], sampled_vertices[:, 2]

'''
坐标转换
'''
zmin = z[np.argmin(z)]
zmax = z[np.argmax(z)]
depthz = (zmax - zmin) / 0.34 * 4.5
z = depthz * (z) / (zmax - zmin)


xmin = x[np.argmin(x)]
xmax = x[np.argmax(x)]
depthx = (xmax - xmin) / 0.34 * 4.5
x = depthx * (x) / (xmax - xmin)


ymin = y[np.argmin(y)]
ymax = y[np.argmax(y)]
depthy = (ymax - ymin) / 0.34 * 4.5
y = depthy * (y) / (ymax - ymin)
# 创建图形

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=26)
# 根据Z值（高程）设置颜色
scatter = ax.scatter(
    x, y, z,
    c=z,  # 使用Z值作为颜色依据
    cmap=cm.viridis,  # 使用viridis颜色映射
    s=0.1,  # 点的大小
    alpha=1# 透明度
)

# 添加颜色条
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Elevation (Z)')  # 颜色条标签

# 设置坐标轴标签和范围
ax.set_xlabel('X axis',fontdict={'family': 'Arial', 'size': 12, 'weight': 'bold'})
ax.set_ylabel('Y axis',fontdict={'family': 'Arial', 'size': 12, 'weight': 'bold'})
ax.set_zlabel('Depth',fontdict={'family': 'Arial', 'size': 12, 'weight': 'bold'})
# ax.set_title('Elevation for case4',fontdict={'family': 'Arial', 'size': 16, 'weight': 'bold'},)
ax.set_zlim(-15, 15)
ax.set_ylim(-20, 20)
ax.set_xlim(20, -20)
# 自动调整视图角度以获得更好的3D效果
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()