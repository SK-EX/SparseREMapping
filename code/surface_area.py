import numpy as np
import pymeshlab as pyml
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def findScour(radius1: float, radius2: float, vertex_matrix):
    """
      找出冲刷坑点云索引
      :param radius1: 内半径
      :param radius2: 外半径
      :param vertex_matrix: 顶点矩阵
      :return: 冲刷坑点索引数组
      """
    # 计算到中心点(0,0)的水平距离
    xy_distances = np.linalg.norm(vertex_matrix[:, :2], axis=1)
    # 筛选条件：环形区域内且z<0的点
    scour_mask = (xy_distances > radius1) & (xy_distances < radius2) & (vertex_matrix[:, 2] < 0)
    return np.where(scour_mask)[0]


def depth(radius, bottom_points):
    bottom = bottom_points[2]
    d = 2 * radius
    return np.abs(bottom) / d * 9

def xoy(scour_vertex,radius, scale):
    '''
    xoy:冲刷坑向xoy平面投影，得到其xoy面上的投影面积
    :return:
    '''
    xoyarea = 0

    scour_xy_vertex = scour_vertex[:,:2]
    x_min = scour_xy_vertex[np.argmin(scour_xy_vertex[:,0])]
    x_max = scour_xy_vertex[np.argmax(scour_xy_vertex[:,0])]
    y_min = scour_xy_vertex[np.argmin(scour_xy_vertex[:,1])]
    y_max = scour_xy_vertex[np.argmax(scour_xy_vertex[:,1])]
    #计算蓝色像素点个数
    #桩径r = 0.42
    pit_area = np.pi * radius**2
    shiji = np.pi * 4.5 **2

    area = ConvexHull(scour_xy_vertex[:,:2]).volume
    area = area / pit_area * shiji
    plt.figure(figsize = [5,5])
    plt.scatter(scour_xy_vertex[:,0], scour_vertex[:,1],s = 0.11,)
    plt.axis('equal')
    plt.show()
    return  area


if __name__ == '__main__':
    # 1. 加载原始网格
    data_path = 'meshed-poisson(seg3).ply'
    ms = pyml.MeshSet()
    ms.load_new_mesh(data_path)
    ms.set_current_mesh(0)

    # 2. 获取网格信息
    print(f'顶点数: {ms.current_mesh().vertex_number()}')
    bounding_box = ms.current_mesh().bounding_box()
    print(f'边界框最大值: {bounding_box.max()}')
    print(f'边界框最小值: {bounding_box.min()}')

    # 3. 找到桩顶点（最高点）
    vertex_matrix = ms.current_mesh().vertex_matrix()
    z_values = vertex_matrix[:, 2]
    top_index = np.argmax(z_values)
    bottom_index = np.argmin(z_values)
    top_point = vertex_matrix[top_index]
    bottom_point = vertex_matrix[bottom_index]
    print(f'桩顶坐标: {top_point}, 高程: {top_point[2]:.2f}')
    print(f'冲刷坑底坐标: {bottom_point}, 高程: {bottom_point[2]:.2f}')

    depth = depth(0.42, bottom_points=bottom_point)
    print(f'冲刷坑深{depth:.4f}')
    # 4. 识别冲刷坑点
    scour_indices = findScour(0.42, 1.75, vertex_matrix)
    print(f'找到冲刷坑点数: {len(scour_indices)}')

    scour_vertex= vertex_matrix[scour_indices]
    xoyarea = xoy(scour_vertex, 0.42, 1)

    print(f'冲刷坑xoy投影面积: {xoyarea}')

    # ms.set_color_per_vertex( vertex_matrix = vertex_matrix[scour_indices], )