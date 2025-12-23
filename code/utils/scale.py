'''
将点云数据按比例放大到实际海床场
'''

import numpy as np
import open3d as o3d
from plyfile import PlyElement, PlyData


def read_ply_ascii(file_path):
    """读取ASCII格式的PLY文件"""
    with open(file_path, 'r') as f:
        # 读取头部信息
        while True:
            line = f.readline().strip()
            if line == "end_header":
                break
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
        vertices = []
        for _ in range(vertex_count):
            parts = f.readline().strip().split()
            vertex_data = [float(x) for x in parts]
            vertices.append(vertex_data)
        ans = np.array(vertices)
    return ans

def scale(points1, points2):
    points1 = points1[:,:3]
    points2 = points2[:,:3]
    alpha = 0.40
    #假设工况1中4.5cm对应但是桩径半径的距离差0.36
    #对于
    points1[:,0] = 4.5 * points1[:,0]/0.40
    points1[:,1] = 4.5 * points1[:,1]/0.40
    points1[:,2] = 4.5 * points1[:,2]/0.40
    #把分割到[-40,40]之内，[-30,50]到
    im_windows = (
        (points1[:, 0] <= 50) &
        (points1[:, 0] >= -30)&
        (points1[:, 1] <= 40 ) &
        (points1[:, 1] >=-40)
                  )
    indices = np.where(im_windows)[0]
    points1 = points1[indices]
    return points1, points2

def save_colored_ply(points, sava_path, color):
    vertex_data = np.zeros(len(points), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('alpha', 'u1')
    ])
    vertex_data['x'] = points[:,0]
    vertex_data['y'] = points[:,1]
    vertex_data['z'] = points[:,2]

    vertex_data['red'] = color[0]
    vertex_data['green'] = color[1]
    vertex_data['blue'] = color[2]
    vertex_data['alpha'] = color[3]

    vertex_element = PlyElement.describe(vertex_data, 'vertex')

    PlyData([vertex_element], text=True).write(sava_path)

if __name__ == '__main__':
    path = '../data/model - 副本.ply'
    points = read_ply_ascii(path)
    points,points2 = scale(points,points)
    save_path = '../data/model_to_actualWord2.ply'
    save_colored_ply(points, save_path, [255, 0, 0, 30])
