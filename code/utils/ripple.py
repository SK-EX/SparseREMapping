'''
计算沙纹信息。
Xu【2024】 Hr[cm],Lr[cm], Hr/Lr , Hr/hi, Lr/hi
'''
import numpy as np
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

def seg10(points1,i):
    points1 = points1[:, :3]
    im_windows = (
        (points1[:, 0] <= 50) &
        (points1[:, 0] >= -30)&
        (points1[:, 1] <= -40 + 8 * i + 8) &
        (points1[:, 1] >=(-40 + 8 * i)
         )
    )
    indices = np.where(im_windows)[0]
    points = points1[indices]
    return points
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
    path1 = '../data/model_to_actualWord2.ply'
    points1 = read_ply_ascii(path1)
    for i in range(10):
        save_path = f'../data/model_to_actualWord2_seg{i}.ply'
        points = seg10(points1, i)
        save_colored_ply(points, save_path, [255,0,0,30])
