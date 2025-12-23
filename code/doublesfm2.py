import exifread
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

file_path = f'D:/wyh/ply/data/15.jpg'
file_path2 = f'D:/wyh/ply/data/21.jpg'


def get_focalLength(path):
    with open(path, "rb") as file:
        tags = exifread.process_file(file, details=False)
        focal_length = str(tags.get('EXIF FocalLength'))
        focal_length = focal_length.split('/')
        focal_length = float(focal_length[0]) / 100
        print(f"焦距: {focal_length}")
    return focal_length


def get_imgLW(path):
    with open(path, "rb") as file:
        tags = exifread.process_file(file, details=False)
        imgLength = str(tags.get('EXIF ExifImageLength'))
        imgWidth = str(tags.get('EXIF ExifImageWidth'))
        imgWidth = int(imgWidth)
        imgLength = int(imgLength)
        return imgLength, imgWidth


def get_innerCamerParamter(path):
    focal_length = get_focalLength(path)
    K = np.zeros([3, 3])
    dx, dy = 0.00000345, 0.00000345  # 更合理的像素尺寸（3.45μm）
    imglength, imgwidth = get_imgLW(path)

    # 正确的内参矩阵构建
    K[0, 0] = focal_length / dx  # fx
    K[1, 1] = focal_length / dy  # fy
    K[0, 2] = imgwidth / 2.0  # cx
    K[1, 2] = imglength / 2.0  # cy
    K[2, 2] = 1.0

    print(f"内参矩阵K:\n{K}")
    return K


def detectAndDescribe(imagepath):
    img = cv2.imread(imagepath)
    if img is None:
        raise ValueError(f"无法读取图像: {imagepath}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(gray, None)
    kps = np.float32([kp.pt for kp in kps])
    return kps, features


def Bmatch(kps1, kps2, feature1, feature2, ratio=0.75):
    bfmatcher = cv2.BFMatcher()
    rawMatches = bfmatcher.knnMatch(feature1, feature2, k=2)

    matches = []
    good = []
    for m, n in rawMatches:
        if m.distance < n.distance * ratio:
            matches.append((m.trainIdx, m.queryIdx))
            good.append(m)

    print(f"匹配点数量: {len(matches)}")
    return matches, good


def xymatch(kps1, kps2, matches):
    pts1 = np.float32([kps1[i] for (_, i) in matches])
    pts2 = np.float32([kps2[i] for (i, _) in matches])
    return pts1, pts2


def essentialMatrix(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    return E, mask


def select_correct_pose(E, pts1, pts2, K):
    """选择正确的相机姿态组合"""
    # 分解本质矩阵得到4种可能的解
    R1, R2, t = cv2.decomposeEssentialMat(E)
    t = t.reshape(3, 1)

    # 4种可能的相机姿态组合
    poses = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]

    # 选择在相机前方的点最多的姿态
    max_front_points = 0
    best_pose = None
    best_points_3d = None

    for R, t in poses:
        # 三角化一组点来测试
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])

        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]

        # 点在第二个相机坐标系中的坐标
        points_cam2 = R @ points_3d + t

        # 统计在相机前方的点（z坐标为正）
        front_points_cam1 = np.sum(points_3d[2] > 0)
        front_points_cam2 = np.sum(points_cam2[2] > 0)

        total_front_points = front_points_cam1 + front_points_cam2

        if total_front_points > max_front_points:
            max_front_points = total_front_points
            best_pose = (R, t)
            best_points_3d = points_3d

    print(f"选择姿态，前方点数量: {max_front_points}")
    return best_pose[0], best_pose[1], best_points_3d


def triangulate_points(R, t, pts1, pts2, K, mask=None):
    """使用OpenCV的三角化函数"""
    if mask is not None:
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

    # 构建投影矩阵
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])

    # 三角化
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3] / points_4d[3]

    # 过滤无效点
    valid_mask = (points_4d[3] != 0) & (points_3d[2] > 0)  # 深度为正

    return points_3d[:, valid_mask], valid_mask


def drawMatches(img1_path, img2_path, kps1, kps2, matches):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("无法读取图像")
        return

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 创建关键点对象
    kp1 = [cv2.KeyPoint(x, y, 1) for x, y in kps1]
    kp2 = [cv2.KeyPoint(x, y, 1) for x, y in kps2]

    match_img = cv2.drawMatches(img1_rgb, kp1, img2_rgb, kp2, matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(15, 10))
    plt.imshow(match_img)
    plt.title(f'特征匹配 ({len(matches)} 个匹配点)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def showCameraTrack(R, t, points_3d=None):
    """显示相机姿态和点云"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 相机1位置和坐标系（世界坐标系）
    cam1_pos = np.zeros(3)
    cam1_x = np.array([1, 0, 0])
    cam1_y = np.array([0, 1, 0])
    cam1_z = np.array([0, 0, 1])

    # 相机2位置和坐标系
    cam2_pos = -R.T @ t.reshape(3)  # 从[R|t]恢复相机位置
    cam2_x = R.T @ np.array([1, 0, 0])
    cam2_y = R.T @ np.array([0, 1, 0])
    cam2_z = R.T @ np.array([0, 0, 1])

    # 绘制相机位置
    ax.scatter(*cam1_pos, c='red', s=100, label='Camera 1')
    ax.scatter(*cam2_pos, c='blue', s=100, label='Camera 2')

    # 绘制相机坐标系
    axis_length = 0.5
    ax.quiver(*cam1_pos, *cam1_x, color='r', length=axis_length, normalize=True, label='Cam1 X')
    ax.quiver(*cam1_pos, *cam1_y, color='g', length=axis_length, normalize=True, label='Cam1 Y')
    ax.quiver(*cam1_pos, *cam1_z, color='b', length=axis_length, normalize=True, label='Cam1 Z')

    ax.quiver(*cam2_pos, *cam2_x, color='r', length=axis_length, normalize=True, linestyle='--')
    ax.quiver(*cam2_pos, *cam2_y, color='g', length=axis_length, normalize=True, linestyle='--')
    ax.quiver(*cam2_pos, *cam2_z, color='b', length=axis_length, normalize=True, linestyle='--')

    # 绘制点云
    if points_3d is not None:
        ax.scatter(points_3d[0], points_3d[1], points_3d[2],
                   c='green', s=1, alpha=0.6, label='3D Points')

    # 设置坐标轴范围
    if points_3d is not None:
        max_range = max(points_3d.max() - points_3d.min(), 5)
        mid_x = (points_3d[0].max() + points_3d[0].min()) * 0.5
        mid_y = (points_3d[1].max() + points_3d[1].min()) * 0.5
        mid_z = (points_3d[2].max() + points_3d[2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('相机姿态和三维重建点云')
    plt.show()


def main():
    try:
        # 1. 获取相机内参
        K = get_innerCamerParamter(file_path)

        # 2. 特征检测和匹配
        kps1, feature1 = detectAndDescribe(file_path)
        kps2, feature2 = detectAndDescribe(file_path2)

        matches, good = Bmatch(kps1, kps2, feature1, feature2)
        pts1, pts2 = xymatch(kps1, kps2, matches)

        # 3. 绘制匹配结果
        drawMatches(file_path, file_path2, kps1, kps2, good)

        # 4. 计算本质矩阵
        E, mask = essentialMatrix(pts1, pts2, K)
        print(f"本质矩阵E:\n{E}")

        # 5. 选择正确的相机姿态
        R, t, test_points = select_correct_pose(E, pts1, pts2, K)
        print(f"旋转矩阵R:\n{R}")
        print(f"平移向量t:\n{t}")

        # 6. 三角化所有点
        points_3d, valid_mask = triangulate_points(R, t, pts1, pts2, K)
        print(f"重建点云数量: {points_3d.shape[1]}")

        # 7. 显示结果
        showCameraTrack(R, t, points_3d)

        # 8. 检查重建质量
        if points_3d.shape[1] > 0:
            depths = points_3d[2]  # Z坐标（深度）
            print(f"深度范围: {depths.min():.2f} ~ {depths.max():.2f}")
            print(f"平均深度: {depths.mean():.2f}")

            # 检查点云分布
            x_range = points_3d[0].max() - points_3d[0].min()
            y_range = points_3d[1].max() - points_3d[1].min()
            z_range = points_3d[2].max() - points_3d[2].min()
            print(f"点云范围: X({x_range:.2f}), Y({y_range:.2f}), Z({z_range:.2f})")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()