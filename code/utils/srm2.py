'''
1、遍历稀疏点云的每个点，找到当前点的梯度（变化率最大）,连线，找到改线段上的点在稠密点云中的具体坐标，并去重

2、计算每个滑动窗口内的点云密度，如果密度较低，则重新进行第一步
'''
import time

import numpy as np
from plyfile import PlyElement, PlyData
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


def to2dimension(spase_points, density_points):
    '''
    :param spase_points:
    : 1、把单桩及其沙床的三维坐标向一个坐标平面投影，可以看出散点的分布程度
      2、将这个二维压缩的模型进行窗口分块，计算每一块的散点数，
      3、散点数少的块按一定比例用稠密点云进行重投影，找出并补充缺失的模型散点
      4、涉及原三维模型可能有一定的位移和旋转，不能简单的将z坐标设为0,用pca做
    '''
    spase_points = spase_points[:, :3]
    density_points = density_points[:, :3]

    pca = PCA(n_components=3)
    pca.fit(spase_points)

    min_variance_idx = np.argmin(pca.explained_variance_)
    normal_vector = pca.components_[min_variance_idx]

    # 投影至平面，同时返回投影信息
    project_spase_points, projection_info_sparse = project_to_plane_with_info(spase_points, normal_vector)
    project_density_points, projection_info_dense = project_to_plane_with_info(density_points, normal_vector)

    # 合并投影信息
    projection_info = {
        'normal_vector': normal_vector,
        'plane_origin': projection_info_sparse['plane_origin'],  # 使用稀疏点云的原点
        'x_axis': projection_info_sparse['x_axis'],
        'y_axis': projection_info_sparse['y_axis']
    }

    return project_spase_points, project_density_points, projection_info


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


def calculate_density_per_windows(spase_points_2d, density_points_2d, windows_num, threshold, projection_info,
                                  spase_points_3d):
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

    all_interpolated_points_3d = []  # 存储所有插值点的3D坐标
    density_per_windows = []
    all_insert_indices = []
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

            # 只在密度不足的窗口进行插值
            if 0 < window_density < 9000:
                current_windows_scale = [current_min_x, current_max_x, current_min_y, current_max_y]
                # 使用梯度插值方法生成2D插值点
                interpolated_points_2d = gradient_based_interpolation(
                    current_windows_scale,
                    spase_points_2d,
                    spase_points_2d,
                    alpha=0.7
                )
                if len(interpolated_points_2d) > 0:
                    # 将2D插值点反投影到3D
                    interpolated_points_3d = back_project_to_3d(
                        interpolated_points_2d,
                        projection_info,
                        spase_points_3d
                    )

                    if len(interpolated_points_3d) > 0:
                        all_interpolated_points_3d.append(interpolated_points_3d)
                        print(f"窗口 [{i},{j}] 原本有{window_density}个点，生成 {len(interpolated_points_3d)} 个3D插值点")
            # (1 <i <5 and 1 < j < 4)
            # if 0 < window_density < 1300:
            #     current_windows_scale = [current_min_x, current_max_x, current_min_y, current_max_y]
            #     insert_indices = SPMREMapping(current_windows_scale, density_points_2d, alpha = 0.1)
            #     if len(insert_indices) > 0 :
            #         #转成三d坐标，并且包含颜色信息
            #         all_insert_indices.append(insert_indices)
    # 合并所有插值点
    if all_interpolated_points_3d:
        all_interpolated_points_3d = np.vstack(all_interpolated_points_3d)
        print(f"总共生成 {len(all_interpolated_points_3d)} 个3D插值点")
    else:
        all_interpolated_points_3d = np.array([])
        print("没有生成任何插值点")
    if all_insert_indices:
        all_insert_indices = process_nested_lists(all_insert_indices)
    return np.reshape(density_per_windows, (windows_num, windows_num)), all_interpolated_points_3d, all_insert_indices

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
    all_elements = np.array(all_elements,dtype = np.int32)
    return all_elements


def SPMREMapping(current_windows_scale, density_points_2d, alpha):
    minx = current_windows_scale[0]
    maxx = current_windows_scale[1]
    miny = current_windows_scale[2]
    maxy = current_windows_scale[3]

    #获取稠密窗口点云
    in_window = (
            (density_points_2d[:,0] >= minx) &
            (density_points_2d[:,0] <= maxx) &
            (density_points_2d[:,1] >= miny) &
            (density_points_2d[:,1] <= maxy)
    )

    sum = np.sum(in_window)
    indices = np.where(in_window)[0]

    insert_indices = np.random.choice(indices, size = int(alpha * sum ),replace = False)
    #反回的是下标
    return insert_indices

def gradient_based_interpolation(current_window_scale, spase_points_2d, density_points_2d, alpha=0.7):
    """
    基于梯度信息的2D插值
    """
    current_windows_min_x, current_windows_max_x, current_windows_min_y, current_windows_max_y = current_window_scale

    # 获取当前窗口内的稀疏点
    in_sparse_window = (
            (spase_points_2d[:, 0] >= current_windows_min_x) &
            (spase_points_2d[:, 0] < current_windows_max_x) &
            (spase_points_2d[:, 1] >= current_windows_min_y) &
            (spase_points_2d[:, 1] < current_windows_max_y)
    )

    sparse_window_indices = np.where(in_sparse_window)[0]
    sparse_window_points = spase_points_2d[sparse_window_indices]

    if len(sparse_window_points) < 5:
        return np.array([])

    try:
        # 使用当前窗口内的所有点计算梯度方向
        pca = PCA(n_components=2)
        pca.fit(sparse_window_points)
        gradient_direction = pca.components_[0]  # 主梯度方向
        perpendicular_direction = np.array([-gradient_direction[1], gradient_direction[0]])  # 垂直梯度方向

        # 计算当前窗口的中心点
        center_point = np.mean(sparse_window_points, axis=0)

        # 计算稀疏点在梯度方向上的投影
        gradient_projections = np.dot(sparse_window_points - center_point, gradient_direction)
        perpendicular_projections = np.dot(sparse_window_points - center_point, perpendicular_direction)

        # 获取投影范围
        min_gradient_proj = np.min(gradient_projections)
        max_gradient_proj = np.max(gradient_projections)
        min_perp_proj = np.min(perpendicular_projections)
        max_perp_proj = np.max(perpendicular_projections)

        interpolated_points = []

        # 在梯度方向进行插值
        num_gradient_steps = max(3, int((max_gradient_proj - min_gradient_proj) / 0.03))
        for i in range(num_gradient_steps):
            gradient_proj = min_gradient_proj + i * (max_gradient_proj - min_gradient_proj) / (num_gradient_steps - 1)

            # 在垂直梯度方向进行插值
            num_perp_steps = max(3, int((max_perp_proj - min_perp_proj) / 0.03))
            for j in range(num_perp_steps):
                perp_proj = min_perp_proj + j * (max_perp_proj - min_perp_proj) / (num_perp_steps - 1)

                # 生成插值点
                interp_point = center_point + gradient_proj * gradient_direction + perp_proj * perpendicular_direction

                # 检查插值点是否在当前窗口内
                if (current_windows_min_x <= interp_point[0] < current_windows_max_x and
                        current_windows_min_y <= interp_point[1] < current_windows_max_y):
                    interpolated_points.append(interp_point)

        # 从稠密点云中选择额外的点来补充
        in_density_window = (
                (density_points_2d[:, 0] >= current_windows_min_x) &
                (density_points_2d[:, 0] < current_windows_max_x) &
                (density_points_2d[:, 1] >= current_windows_min_y) &
                (density_points_2d[:, 1] < current_windows_max_y)
        )

        density_window_indices = np.where(in_density_window)[0]
        if len(density_window_indices) > 0:
            # 沿着梯度方向选择稠密点
            density_window_points = density_points_2d[density_window_indices]
            density_projections = np.dot(density_window_points - center_point, gradient_direction)

            # 选择梯度方向上的极值点
            if len(density_projections) > 0:
                extreme_indices = [
                    np.argmax(density_projections),
                    np.argmin(density_projections)
                ]
                for idx in extreme_indices:
                    if idx < len(density_window_points):
                        interpolated_points.append(density_window_points[idx])


        # # 控制插值点数量
        # if interpolated_points:
        #     interpolated_points = np.array(interpolated_points)
        #     target_count = min(len(interpolated_points), int(len(sparse_window_points) * alpha))
        #
        #     if len(interpolated_points) > target_count:
        #         selected_indices = np.random.choice(len(interpolated_points), size=target_count, replace=False)
        #         final_points = interpolated_points[selected_indices]
        #     else:
        #         final_points = interpolated_points
                final_points = interpolated_points
                return final_points

    except Exception as e:
        print(f"梯度插值失败: {e}")

    return np.array([])


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


def save_colored_ply_with_interpolation(spase_points_3d, interpolated_points_3d, density_points_3d, insert_indices, save_path):
    """
    保存包含插值点的PLY文件
    """


        #合并三维点云
    density_points_3d = density_points_3d[insert_indices, :]
    ans1 =  density_points_3d

    if len(interpolated_points_3d) == 0:
        ans2 = spase_points_3d
        print("没有插值点，只保存原始点云")
    else:
        # 为插值点添加颜色信息（红色）
        interpolated_with_color = add_color_to_points(interpolated_points_3d, color=[255, 0, 0])
        ans2 = np.append(spase_points_3d, interpolated_with_color, axis=0)
        print(f"合并后点云总数: {len(ans2)} (原始: {len(spase_points_3d)}, 插值: {len(interpolated_points_3d)})")

    # 创建顶点数据
    vertex_data = create_vertex_data(ans1, ans2)
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    PlyData([vertex_element], text=True).write(save_path)
    print(f"PLY文件已保存到: {save_path}")


def add_color_to_points(points_3d, color=[255, 0, 0]):
    """
    为点添加颜色信息
    """
    n_points = len(points_3d)
    if points_3d.shape[1] >= 6:
        colored_points = points_3d.copy()
        if colored_points.shape[1] >= 9:
            colored_points[:, 6:9] = color
        else:
            extended_points = np.zeros((n_points, 10))
            extended_points[:, :points_3d.shape[1]] = points_3d
            extended_points[:, 6:9] = color
            extended_points[:, 9] = 255
            colored_points = extended_points
    else:
        colored_points = np.zeros((n_points, 10))
        colored_points[:, :3] = points_3d[:, :3]
        colored_points[:, 6:9] = color
        colored_points[:, 9] = 255

    return colored_points


def create_vertex_data(points1, points2):
    """
    创建PLY顶点数据
    """
    length = len(points1) + len(points2)
    if np.shape(points1)[1] == 7:
        app = np.zeros((len(points1),3))
        temp = np.zeros((len(points1),10))
        temp[:,:3] = points1[:,:3]
        temp[:,3:6] = app
        temp[:,6:10] = points1[:,6:10]
        points1 = temp
    points = np.append(points1, points2, axis= 0)
    vertex_data = np.zeros(length, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('alpha', 'u1')
    ])

    vertex_data['x'] = points[:, 0]
    vertex_data['y'] = points[:, 1]
    vertex_data['z'] = points[:, 2]

    if points.shape[1] >= 6:
        vertex_data['nx'] = points[:, 3]
        vertex_data['ny'] = points[:, 4]
        vertex_data['nz'] = points[:, 5]

    if points.shape[1] >= 9:
        vertex_data['red'] = points[:, 6]
        vertex_data['green'] = points[:, 7]
        vertex_data['blue'] = points[:, 8]

    vertex_data['alpha'] = 255

    return vertex_data

import pandas as pd

if __name__ == '__main__':

    path = '../data/model - 副本.ply'
    path_density  = '../data/meshed-poisson2.ply'
    save_path = '../data/srm.ply'

    spase_points = read_ply_ascii(path)
    density_points = read_ply_ascii(path_density)
    start_time = time.time()
    # 获取投影信息
    spase_points_2d, density_points_2d, projection_info = to2dimension(spase_points, density_points)
    print('稀疏点云形状:', spase_points_2d.shape)
    print('稠密点云形状:', density_points_2d.shape)

    # 计算密度并进行梯度插值
    density_per_windows, interpolated_points_3d ,all_insert_indices = calculate_density_per_windows(
        spase_points_2d, density_points_2d, windows_num=10, threshold=5000,
        projection_info=projection_info, spase_points_3d=spase_points
    )

    # 保存结果
    save_colored_ply_with_interpolation(spase_points, interpolated_points_3d,  density_points, all_insert_indices,save_path)
    print("密度分布:")
    print(density_per_windows)
    end_time = time.time()
    print(start_time-end_time)
