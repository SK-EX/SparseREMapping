import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
            x, y, z = map(float, parts[:3])
            vertices.append([x, y, z])

    return np.array(vertices)


def calculate_surface_area(vertices):
    """计算点云的表面面积"""
    # 对x,y坐标进行Delaunay三角剖分
    tri = Delaunay(vertices[:, :2])

    # 计算每个三角形的面积
    triangles = vertices[tri.simplices]
    a = triangles[:, 1, :] - triangles[:, 0, :]
    b = triangles[:, 2, :] - triangles[:, 0, :]
    cross = np.cross(a, b)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    total_area = np.sum(areas)
    return total_area


def visualize_point_cloud(vertices):
    """可视化点云"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
               c='r', marker='o', s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Scour Pit Point Cloud')
    plt.show()


def visualize_surface(vertices, tri):
    """可视化重建的表面"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=tri.simplices, cmap='viridis', alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Reconstructed Scour Pit Surface')
    plt.show()


def main():

    ply_file = "scour_pit_red.ply.txt"
    vertices = read_ply_ascii(ply_file)
    print(f"Loaded {len(vertices)} vertices")

    visualize_point_cloud(vertices)

    # 使用Delaunay三角剖分
    tri = Delaunay(vertices[:, :2])
    surface_area = calculate_surface_area(vertices)
    print(f"{surface_area:.4f} square units")

    # 可视化重建的表面
    visualize_surface(vertices, tri)


if __name__ == "__main__":
    main()