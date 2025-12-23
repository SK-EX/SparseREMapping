import numpy as np
from scipy.spatial import KDTree
from plyfile import PlyData, PlyElement

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

        # 读取顶点数据
        vertices = []
        color = []
        for _ in range(vertex_count):
            parts = f.readline().strip().split()
            x, y, z = map(float, parts[:3])
            r,g,b = map(int, parts[6:9])
            vertices.append([x, y, z])
            color.append([r,g,b])
        ans = np.append(vertices, color, axis = 1)
        ans = ans[ans[:, 0].argsort()]

    return ans


def read_ply_ascii2(file_path):
    """读取ASCII格式的PLY文件"""
    with open(file_path, 'r') as f:
        # 读取头部信息
        while True:
            line = f.readline().strip()
            if line == "end_header":
                break
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[2])

        # 读取顶点数据
        vertices = []
        for _ in range(vertex_count):
            parts = f.readline().strip().split()
            # 将字符串转换为浮点数并添加到列表
            vertex_data = [float(x) for x in parts]
            vertices.append(vertex_data)

        # 转换为numpy数组
        ans = np.array(vertices)
    return ans

def findNeastNabor(vertices):
    tree = KDTree(vertices[:,:3])
    mid_array = np.zeros((vertices.shape[0],10))
    distances, indices = tree.query(vertices[:,:3], 2)
    indices = indices[:,1]
    new_emu = []
    for i in range(vertices.shape[0]):
        mid_x = (vertices[i,0] + vertices[indices[i],0] )/ 2
        mid_y = (vertices[i,1] + vertices[indices[i],1] )/ 2
        mid_z = (vertices[i,2] + vertices[indices[i],2]) / 2
        color_r = int((vertices[i,3] + vertices[indices[i],3]) / 2)
        color_g = int((vertices[i,4] + vertices[indices[i],4]) / 2)
        color_b = int((vertices[i,5] + vertices[indices[i],5]) / 2)
        mid_array[i] = [mid_x, mid_y,mid_z,0,0,0,color_r,color_g,color_b,255]
    return mid_array


def save_colored_ply(scour_vertices, mid_array, sava_path):

    ans = np.append(scour_vertices,mid_array,axis = 0)

    vertex_data = np.zeros(len(ans), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'),('ny', 'f4'),('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('alpha', 'u1')
    ])

    vertex_data['x'] = ans[:,0]
    vertex_data['y'] = ans[:,1]
    vertex_data['z'] = ans[:,2]

    vertex_data['nx'] = ans[:, 3]
    vertex_data['ny'] = ans[:, 4]
    vertex_data['nz'] = ans[:, 5]

    vertex_data['red'] = ans[:,6]
    vertex_data['green'] = ans[:,7]
    vertex_data['blue'] =  ans[:,8]
    vertex_data['alpha'] = ans[:,9]

    vertex_element = PlyElement.describe(vertex_data, 'vertex')

    PlyData([vertex_element], text=True).write(sava_path)




if __name__ == '__main__':

    path = '../data/model - 副本.ply'
    vertices = read_ply_ascii(path)
    print(vertices)
    mid_array=  findNeastNabor(vertices)
    print(mid_array)

    vertx = read_ply_ascii2(path)
    save_colored_ply(vertx, mid_array,sava_path='../data/linear.ply')
    print(1)