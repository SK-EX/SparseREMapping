import pandas as pd
import numpy as np
from pandas import read_fwf
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

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


def find_corresponding_points(reference_points, target_points):
    """
    为target_points中的每个点在reference_points中找到最近的点
    返回对应点的索引和距离
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

    # 统计信息
    if len(reference_points) != len(target_points):
        print(f"注意：参考点云({len(reference_points)})和目标点云({len(target_points)})点数不同")
        print(f"使用{len(distances)}个对应点计算")

    # 计算RMSE
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse, distances


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


def chamfer_distance(points1, points2,method='symmetric'):
    points1 = points1[:,:3]
    points2 = points2[:,:3]
    tree1 = KDTree(points1)
    tree2 = KDTree(points2)
    # 计算points1中每个点到points2的最近距离
    distances1_to_2, _ = tree2.query(points1)
    cd_1_to_2 = np.mean(distances1_to_2)


    if method == 'symmetric':
        # 计算points2中每个点到points1的最近距离
        distances2_to_1, _ = tree1.query(points2)
        cd_2_to_1 = np.mean(distances2_to_1)
        cd_distance = (cd_1_to_2 + cd_2_to_1) /2

        details = {
            'chamfer_distance': cd_distance,
            'forward_distance': cd_1_to_2,  # points1 -> points2
            'backward_distance': cd_2_to_1,  # points2 -> points1
            'forward_std': np.std(distances1_to_2),
            'backward_std': np.std(distances2_to_1),
            'forward_max': np.max(distances1_to_2),
            'backward_max': np.max(distances2_to_1)
        }
    else:
        cd_distance = cd_1_to_2
        details = {
            'chamfer_distance': cd_distance,
            'forward_distance': cd_1_to_2,
            'forward_std': np.std(distances1_to_2),
            'forward_max': np.max(distances1_to_2)
        }

    return cd_distance, details

if __name__ == '__main__':
    path1 = '../data/model - 副本.ply'
    path2 = '../data/liner.ply'
    path3 = '../data/gradient.ply'
    path4 = '../data/srm.ply'
    path5 = '../data/fused - 副本.ply'
    model = read_ply_ascii(path1)
    liner = read_ply_ascii(path2)
    gradient = read_ply_ascii(path3)
    srm = read_ply_ascii(path4)
    density = read_ply_ascii(path5)
    cd_model,details = chamfer_distance(model,density,method='symmetric')
    cd_liner, details2 = chamfer_distance(liner, density, method='symmetric')
    cd_gradient, details3 = chamfer_distance(gradient, density, method='symmetric')
    cd_srm, details4 = chamfer_distance(srm, density, method='symmetric')
    print(cd_model)
    print(cd_liner)
    print(cd_gradient)
    print(cd_srm)