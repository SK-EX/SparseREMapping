import pandas as pd
import numpy as np
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


def to2dimension(spase_points):
    '''
    :param spase_points:
    : 1、把单桩及其沙床的三维坐标向一个坐标平面投影，可以看出散点的分布程度
      2、将这个二维压缩的模型进行窗口分块，计算每一块的散点数，
      3、散点数少的块按一定比例用稠密点云进行重投影，找出并补充缺失的模型散点
      4、涉及原三维模型可能有一定的位移和旋转，不能简单的将z坐标设为0,用pca做
    '''
    spase_points = spase_points[:, :3]
    pca = PCA(n_components=3)
    pca.fit(spase_points)

    min_variance_idx = np.argmin(pca.explained_variance_)
    normal_vector = pca.components_[min_variance_idx]

    # 投影至平面，同时返回投影信息
    project_spase_points, projection_info_sparse = project_to_plane_with_info(spase_points, normal_vector)
    # 合并投影信息
    projection_info = {
        'normal_vector': normal_vector,
        'plane_origin': projection_info_sparse['plane_origin'],  # 使用稀疏点云的原点
        'x_axis': projection_info_sparse['x_axis'],
        'y_axis': projection_info_sparse['y_axis']
    }

    return project_spase_points, projection_info


def project_to_plane_with_info(points, normal_vector):
    """
    将三维点投影到指定法向量平面上，并返回投影信息
    """
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # 构建投影坐标系
    arbitrary_vector = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(arbitrary_vector, normal_vector)) > 0.9:
        arbitrary_vector = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(arbitrary_vector, normal_vector)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(normal_vector, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # 计算投影平面的原点
    plane_origin = np.mean(points, axis=0)

    # 将3D点投影到2D平面
    points_relative = points - plane_origin
    points_2d = np.column_stack([
        np.dot(points_relative, x_axis),
        np.dot(points_relative, y_axis)
    ])

    # 保存投影信息用于反投影
    projection_info = {
        'plane_origin': plane_origin,
        'x_axis': x_axis,
        'y_axis': y_axis,
        'normal_vector': normal_vector
    }

    return points_2d, projection_info


def project_to_plane(points, normal_vector):
    """
    简化的投影函数（保持向后兼容）
    """
    points_2d, _ = project_to_plane_with_info(points, normal_vector)
    return points_2d


def calculate_density_per_windows(spase_points_2d,windows_num):
    """
    计算窗口密度并进行梯度插值
    """
    points_2d = np.array(spase_points_2d)

    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError("输入必须是形状为(N, 2)的二维数组")

    # 计算坐标范围
    min_x = np.min(points_2d[:, 0])
    min_y = np.min(points_2d[:, 1])
    max_x = np.max(points_2d[:, 0])
    max_y = np.max(points_2d[:, 1])

    # 计算窗口大小
    windows_width = (max_x - min_x) / windows_num
    windows_length = (max_y - min_y) / windows_num

    density_per_windows = []
    for i in range(windows_num):
        current_min_y = min_y + i * windows_length
        current_max_y = current_min_y + windows_length

        for j in range(windows_num):
            current_min_x = min_x + j * windows_width
            current_max_x = current_min_x + windows_width

            # 筛选窗口内的点
            in_window = (
                    (points_2d[:, 0] >= current_min_x) &
                    (points_2d[:, 0] < current_max_x) &
                    (points_2d[:, 1] >= current_min_y) &
                    (points_2d[:, 1] < current_max_y)
            )

            window_density = np.sum(in_window)
            density_per_windows.append(window_density)

    return np.reshape(density_per_windows, (windows_num, windows_num))


def process_nested_lists(nested_lists):
    """
    处理嵌套列表，计算所有元素数量并合并到一个大列表中
    """
    total_count = 0
    all_elements = []

    for i, sublist in enumerate(nested_lists):
        count = len(sublist)
        total_count += count
        all_elements.extend(sublist)
    all_elements = np.array(all_elements, dtype=np.int32)
    return all_elements

def back_project_to_3d(points_2d, projection_info, reference_points_3d):
    """
    将2D点反投影回3D空间，保持原始高度信息
    """
    if len(points_2d) == 0:
        return np.array([])

    plane_origin = projection_info['plane_origin']
    x_axis = projection_info['x_axis']
    y_axis = projection_info['y_axis']
    normal_vector = projection_info['normal_vector']

    # 计算参考点云的2D投影
    reference_points_2d = project_to_plane(reference_points_3d[:, :3], normal_vector)

    if len(reference_points_2d) <= 1:
        return simple_back_projection(points_2d, projection_info, reference_points_3d)

    # 构建KD树用于最近邻搜索
    kdtree_2d = KDTree(reference_points_2d)

    points_3d = []

    for point_2d in points_2d:
        try:
            # 找到最近的参考点
            distances, indices = kdtree_2d.query(point_2d.reshape(1, -1), k=min(5, len(reference_points_2d)))

            # 获取对应的3D参考点
            nearest_3d_points = reference_points_3d[indices[0], :3]

            # 使用加权平均高度
            weights = 1.0 / (distances[0] + 1e-8)
            weights = weights / np.sum(weights)

            # 计算参考点在法向量方向上的投影高度
            reference_heights = []
            for ref_point in nearest_3d_points:
                height = np.dot(ref_point - plane_origin, normal_vector)
                reference_heights.append(height)

            # 加权平均高度
            avg_height = np.sum(np.array(reference_heights) * weights)

            # 构建3D点
            point_on_plane = plane_origin + point_2d[0] * x_axis + point_2d[1] * y_axis
            point_3d = point_on_plane + avg_height * normal_vector

            points_3d.append(point_3d)

        except Exception as e:
            # 使用简单方法
            point_on_plane = plane_origin + point_2d[0] * x_axis + point_2d[1] * y_axis
            avg_height = np.mean([np.dot(ref_point - plane_origin, normal_vector)
                                  for ref_point in reference_points_3d[:, :3]])
            point_3d = point_on_plane + avg_height * normal_vector
            points_3d.append(point_3d)

    return np.array(points_3d)


def simple_back_projection(points_2d, projection_info, reference_points_3d):
    """
    简单的反投影方法
    """
    plane_origin = projection_info['plane_origin']
    x_axis = projection_info['x_axis']
    y_axis = projection_info['y_axis']
    normal_vector = projection_info['normal_vector']

    # 计算平均高度
    heights = [np.dot(point - plane_origin, normal_vector) for point in reference_points_3d[:, :3]]
    avg_height = np.mean(heights)

    points_3d = []
    for point_2d in points_2d:
        point_on_plane = plane_origin + point_2d[0] * x_axis + point_2d[1] * y_axis
        point_3d = point_on_plane + avg_height * normal_vector
        points_3d.append(point_3d)

    return np.array(points_3d)

if __name__ == '__main__':
    path2 = '../data/liner.ply'
    path3 = '../data/'
    path4 = '../data/srm.ply'

    points2 = read_ply_ascii(path2)
    points3 = read_ply_ascii(path3)
    points4 = read_ply_ascii(path4)

    # 获取投影信息
    spase_points_2d2, projection_info2 = to2dimension(points2)
    spase_points_2d3, projection_info3 = to2dimension(points3)
    spase_points_2d4, projection_info4 = to2dimension(points4)


    density_per_windows2= calculate_density_per_windows(
        spase_points_2d2,windows_num=30,
    )
    density_per_windows3= calculate_density_per_windows(
        spase_points_2d3,  windows_num=30,
    )
    density_per_windows4= calculate_density_per_windows(
        spase_points_2d4,  windows_num=30,
    )
    print("密度分布:")
    print(density_per_windows2)
    print(density_per_windows3)
    print(density_per_windows4)
    data2 = pd.DataFrame(density_per_windows2)
    data3 = pd.DataFrame(density_per_windows3)
    data4 = pd.DataFrame(density_per_windows4)

    writer = pd.ExcelWriter('../data/density.xlsx')
    data2.to_excel(writer,'Sheet1')
    data3.to_excel(writer,'Sheet2')
    data4.to_excel(writer,'Sheet3')
    writer._save()
    writer.close()
