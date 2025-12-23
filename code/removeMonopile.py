'''
用于数据后处理剔除桩本身，成图好看
'''
import numpy as np
import pymeshlab as pyml
from plyfile import PlyData, PlyElement


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
    scour_mask = (xy_distances > radius1)
    return np.where(scour_mask)[0]


def save_colored_ply(scour_vertices, sava_path, color):
      vertex_data = np.zeros(len(scour_filter), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('alpha','u1')
      ])
      vertex_data['x'] = scour_vertices['x']
      vertex_data['y'] = scour_vertices['y']
      vertex_data['z'] = scour_vertices['z']

      vertex_data['red'] = color[0]
      vertex_data['green'] = color[1]
      vertex_data['blue'] = color[2]
      vertex_data['alpha'] = color[3]

      vertex_element = PlyElement.describe(vertex_data, 'vertex')

      PlyData([vertex_element], text=True).write(sava_path)

def depth(radius, bottom_points):
        bottom = bottom_points[2]
        d =  2 * radius
        return np.abs(bottom) / d * 9

def top(r, top_points):
    top = top_points[2]
    d  = 2 * r
    return np.abs(top) / d *9
if __name__ == '__main__':
    # 1. 加载原始网格
    data_path = './data/srm.ply'
    ms = pyml.MeshSet()
    ms.load_new_mesh(data_path)
    ms.set_current_mesh(0)

    # # 2. 获取网格信息
    # print(f'顶点数: {ms.current_mesh().vertex_number()}')
    # bounding_box = ms.current_mesh().bounding_box()
    # print(f'边界框最大值: {bounding_box.max()}')
    # print(f'边界框最小值: {bounding_box.min()}')

    # 3. 找到桩顶点（最高点）
    vertex_matrix = ms.current_mesh().vertex_matrix()
    z_values = vertex_matrix[:, 2]
    top_index = np.argmax(z_values)
    bottom_index = np.argmin(z_values)
    top_point = vertex_matrix[top_index]
    bottom_point = vertex_matrix[bottom_index]
    print(f'桩顶坐标: {top_point}, 高程: {top_point[2]:.2f}')
    print(f'冲刷坑底坐标: {bottom_point}, 高程: {bottom_point[2]:.2f}')

    depth = depth(0.35, bottom_points = bottom_point)

    top = top(0.35, top_points  = top_point)
    print(f'冲刷坑深{depth:.4f}')
    print(f'淤积高度{top:.4f}')
    # 4. 识别冲刷坑点
    scour_indices = findScour(0.34, 1.8, vertex_matrix)
    print(f'找到冲刷坑点数: {len(scour_indices)}')

   # ms.set_color_per_vertex( vertexcolor = vertex_matrix[scour_indices])

    vertx = PlyData.read(data_path)['vertex'].data
    print(vertx)
    scour_filter = vertx[scour_indices]
    print(scour_filter)
    print(len(scour_filter))
    save_path =  'removeMonopile.ply'
    save_colored_ply(scour_filter,save_path,[255,0,0,30])
    print(f'save successfully!save to:{save_path}')