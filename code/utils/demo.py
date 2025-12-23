import exifread
import numpy as np
import cv2
import matplotlib.pyplot as plt

file_path = 'D:/wyh/ply/data/15.jpg'
file_path2 = 'D:/wyh/ply/data/21.jpg'


def get_focalLength(path):
    """从EXIF数据获取焦距（单位：毫米）"""
    try:
        with open(path, "rb") as file:
            tags = exifread.process_file(file, details=False)
            focal_length_tag = tags.get('EXIF FocalLength')

            if focal_length_tag is None:
                print(f"警告: {path} 中未找到焦距信息，使用默认值50mm")
                return 50.0

            focal_length = float(focal_length_tag.values[0])
            print(f"焦距: {focal_length}mm")
            return focal_length
    except Exception as e:
        print(f"读取焦距时出错: {e}, 使用默认值50mm")
        return 50.0


def get_imgLW(path):
    """获取图像尺寸"""
    try:
        with open(path, "rb") as file:
            tags = exifread.process_file(file, details=False)
            imgLength = tags.get('EXIF ExifImageLength')
            imgWidth = tags.get('EXIF ExifImageWidth')

            if imgLength is None or imgWidth is None:
                # 如果EXIF中没有尺寸信息，直接从图像读取
                img = cv2.imread(path)
                height, width = img.shape[:2]
                return height, width

            imgWidth = int(imgWidth.values[0])
            imgLength = int(imgLength.values[0])
            return imgLength, imgWidth
    except Exception as e:
        print(f"读取图像尺寸时出错: {e}")
        # 直接读取图像获取尺寸
        img = cv2.imread(path)
        height, width = img.shape[:2]
        return height, width


def get_innerCamerParamter(path):
    """计算相机内参矩阵"""
    focal_length_mm = get_focalLength(path)
    height, width = get_imgLW(path)

    # 假设传感器尺寸（单位：毫米）
    sensor_width_mm = 36.0  # 全画幅传感器宽度
    sensor_height_mm = 24.0  # 全画幅传感器高度

    # 计算焦距像素值
    fx = focal_length_mm * width / sensor_width_mm
    fy = focal_length_mm * height / sensor_height_mm

    # 主点坐标（图像中心）
    cx = width / 2
    cy = height / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    return K


def detectAndDescribe(imagepath):
    """检测特征点并计算描述子"""
    img = cv2.imread(imagepath)
    if img is None:
        raise ValueError(f"无法读取图像: {imagepath}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用SIFT特征检测器
    descriptor = cv2.SIFT_create()
    kps, features = descriptor.detectAndCompute(gray, None)

    # 将关键点转换为NumPy数组
    kps = np.float32([kp.pt for kp in kps])

    return kps, features


def match_features(feature1, feature2, ratio=0.75):
    """匹配特征点"""
    bfmatcher = cv2.BFMatcher()
    rawMatches = bfmatcher.knnMatch(feature1, feature2, k=2)

    matches = []
    good = []

    for m, n in rawMatches:
        if m.distance < ratio * n.distance:
            matches.append((m.trainIdx, m.queryIdx))
            good.append(m)

    return matches, good


def get_matched_points(kps1, kps2, matches):
    """获取匹配点的坐标"""
    pts1 = np.float32([kps1[i] for (_, i) in matches])
    pts2 = np.float32([kps2[i] for (i, _) in matches])
    return pts1, pts2


def draw_matches(img1_path, img2_path, kps1, kps2, matches):
    """绘制匹配结果"""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 转换为RGB用于matplotlib显示
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 绘制匹配结果
    fig, ax = plt.subplots(figsize=(15, 10))
    match_img = cv2.drawMatches(
        img1_rgb, [cv2.KeyPoint(x, y, 1) for x, y in kps1],
        img2_rgb, [cv2.KeyPoint(x, y, 1) for x, y in kps2],
        matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    ax.imshow(match_img)
    ax.set_title(f'特征匹配 ({len(matches)} 个匹配点)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 计算相机内参
    K1 = get_innerCamerParamter(file_path)
    K2 = get_innerCamerParamter(file_path2)
    print("相机1内参矩阵:")
    print(K1)
    print("相机2内参矩阵:")
    print(K2)

    # 检测和匹配特征点
    kps1, feature1 = detectAndDescribe(file_path)
    kps2, feature2 = detectAndDescribe(file_path2)

    matches, good_matches = match_features(feature1, feature2)
    pts1, pts2 = get_matched_points(kps1, kps2, matches)

    print(f"找到 {len(matches)} 个匹配点")

    # 绘制匹配结果
    draw_matches(file_path, file_path2, kps1, kps2, good_matches)

    # 计算基础矩阵
    if len(pts1) >= 8:
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3, 0.99)
        print("基础矩阵:")
        print(F)

        # 筛选内点
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        print(f"RANSAC后保留 {len(pts1)} 个内点")
    else:
        print("匹配点不足，无法计算基础矩阵")