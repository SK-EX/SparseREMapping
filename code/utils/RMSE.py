import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


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
        return np.array(vertices)


def find_corresponding_points(reference_points, target_points):
    """
    为target_points中的每个点在reference_points中找到最近的点
    返回对应点的索引
    """
    kdtree = KDTree(reference_points[:, :3])
    distances, indices = kdtree.query(target_points[:, :3], k=1)
    return indices.flatten(), distances.flatten()


def calculate_rmse(reference_points, target_points):
    """
    计算RMSE（均方根误差）
    假设两个点云已经对齐，计算对应点之间的欧氏距离
    """
    # 找到对应点
    corr_indices, distances = find_corresponding_points(reference_points, target_points)

    # 如果点云大小不同，使用所有找到的对应点
    if len(reference_points) != len(target_points):
        print(f"注意：参考点云({len(reference_points)})和目标点云({len(target_points)})点数不同")
        print(f"使用{len(distances)}个对应点计算")

    # 计算RMSE
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse

def calculate_r_squared(reference_points, target_points):
    """
    计算R²（决定系数）
    衡量目标点云对参考点云的解释程度
    """
    # 找到对应点
    corr_indices, distances = find_corresponding_points(reference_points, target_points)

    # 获取对应的参考点
    corresponding_ref_points = reference_points[corr_indices]

    # 提取Z坐标（假设高程是主要比较维度）
    ref_z = corresponding_ref_points[:, 2]
    target_z = target_points[:, 2]

    # 计算总平方和
    ss_total = np.sum((ref_z - np.mean(ref_z)) ** 2)

    # 计算残差平方和
    ss_residual = np.sum((ref_z - target_z) ** 2)

    # 计算R²
    if ss_total == 0:
        print("警告：总平方和为0，R²无法计算")
        return 0
    r_squared = 1 - (ss_residual / ss_total)

    return r_squared


if __name__ == '__main__':
    path1 = '../data/model - 副本.ply'
    path2 = '../data/linear.ply'
    path3 = '../data/gradient.ply'
    path4 = '../data/srm.ply'
    path5 = '../data/fused - 副本.ply'
    model = read_ply_ascii(path1)
    linear = read_ply_ascii(path2)
    gradient = read_ply_ascii(path3)
    srm = read_ply_ascii(path4)
    density = read_ply_ascii(path5)

    # rmse1 = calculate_rmse(density, model)
    # rmse2 = calculate_rmse(density, linear)
    # rmse3 = calculate_rmse(density, gradient)
    # rmse4 = calculate_rmse(density, srm)

    rmse1 = calculate_rmse( model,density)
    rmse2 = calculate_rmse( linear,density)
    rmse3 = calculate_rmse(gradient,density)
    rmse4 = calculate_rmse(srm,density)

    # r2 = calculate_r_squared(density, model)
    # r22 = calculate_r_squared(density, linear)
    # r23 = calculate_r_squared(density, gradient)
    # r24 = calculate_r_squared(density, srm)

    r2 = calculate_r_squared(model,density)
    r22 = calculate_r_squared(linear,density)
    r23 = calculate_r_squared(gradient,density)
    r24 = calculate_r_squared(srm,density)

    print(rmse1)
    print(rmse2)
    print(rmse3)
    print(rmse4)

    print(r2)
    print(r22)
    print(r23)
    print(r24)