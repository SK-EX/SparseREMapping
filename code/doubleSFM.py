import exifread
import numpy  as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

file_path  = f'D:/wyh/ply/data/15.jpg'
file_path2 = f'D:/wyh/ply/data/21.jpg'
def get_focalLength(path):
    with open(file_path,"rb") as file:
        tags = exifread.process_file(file,details=False)
        focal_length = str(tags.get('EXIF FocalLength'))
        focal_length = focal_length.split('/')
        focal_length = float(focal_length[0]) / 100
        print(focal_length)

    return focal_length

def get_imgLW(path):
    with open(file_path, "rb") as file:
        tags = exifread.process_file(file, details=False)
        imgLength = str(tags.get('EXIF ExifImageLength'))
        imgWidth = str(tags.get('EXIF ExifImageWidth'))
        imgWidth = int(imgWidth)
        imgLength = int(imgLength)
        return imgLength, imgWidth

def get_innerCamerParamter(path):
        focal_length = get_focalLength(path)
        K = np.zeros([3,3])
        dx ,dy = 0.0000008,0.0000008
        imglength, imgwidth = get_imgLW(path)
        K[0][2], K[1][2] , K[2][2] = imglength / 2, imgwidth / 2, 1
        K[0][0] , K[1][1] = focal_length / (1000 * dx),  focal_length / (1000 * dy)
        return K

#获取特征点匹配对
#调库实现特征点检测
def detectAndDescribe(imagepath):
        # 将彩色图片转换成灰度图
        img = cv2.imread(imagepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 建立SIFT生成器
        descriptor = cv2.SIFT_create()
        # 检测SIFT特征点，并计算描述子
        (kps, features) = descriptor.detectAndCompute(gray, None)

        # 将结果转换成NumPy数组
        kps = np.float32([kp.pt for kp in kps])

        # 返回特征点集，及对应的描述特征
        return kps, features

#匹配
def Bmatch(kps1,kps2,feature1, feature2):
    bfmatcher = cv2.BFMatcher()

    rawMatches = bfmatcher.knnMatch(feature1, feature2, k=2)
    # print(matches)
    # print(len(matches))
    # 调整ratio
    matches = []
    good = []
    for m,n in rawMatches:
        # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        if m.distance < n.distance * 0.75:
            # 存储两个点在featuresA, featuresB中的索引值
            matches.append((m.trainIdx, m.queryIdx))
            good.append(m)
    return matches,good

def xymatch(kps1,kps2,matches):
    '''
    :return:img1、img2匹配对的xy坐标 ，（x1,y1） / (x2,y2)
    '''
    x1y1 = []
    x2y2 = []
    for trainIdx, queryIdx in matches:
        pta = (int(kps1[queryIdx][0]), int(kps1[queryIdx][1]))
        ptb = (int(kps2[trainIdx][0]), int(kps2[trainIdx][1]))
        x1y1.append(pta)
        x2y2.append(ptb)
    #equals
    pts1 = np.float32([kps1[i] for (_,i ) in matches])
    pts2 = np.float32([kps2[i] for (i, _) in matches])
    x1y1 = np.array(x1y1)
    x2y2 = np.array(x2y2)
    return x1y1,x2y2,pts1,pts2

def fundamentalMatrix(x1y1,x2y2):

    F,mask = cv2.findFundamentalMat(x1y1, x2y2,cv2.FM_RANSAC, 3, 0.99)

    if F is not None:
        return F
    return None


def essentialMatrix(x1y1,x2y2,K):

    E,mask = cv2.findEssentialMat(x1y1,x2y2,K,cv2.FM_RANSAC,0.999,2.0)

    if E is not None:
        return E, mask
    return None

def drawMatches(img1, img2,kps1, kps2, matches):
    # 转换为RGB用于matplotlib显示
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
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

def RT(x1y1,x2y2,E, mask, K, u,s,v):
    #t = -+u3
    x1y1_inliers = x1y1[mask.ravel() == 1]
    x2y2_inliers = x2y2[mask.ravel() == 1]

    retval, R,t, mask_pose = cv2.recoverPose(E, x1y1_inliers, x2y2_inliers, cameraMatrix = K)
    t1 = u[:, 2]
    t1 = np.array([t1])
    t2 = -u[:, 2]
    t2 = np.array([t2])
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    if np.linalg.det(u) < 0:
        u = -u
    if np.linalg.det(v) < 0:
        v = -v

    R1 = u @ W @ np.transpose(v)
    np.array(R1)
    R2 = u @ np.transpose(W) @ np.transpose(v)
    R2 = np.array(R2)
    if np.linalg.det(R1) <  0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    RTsolution = [
        np.append(R1, t1.T,axis=1),
        np.append(R1, t2.T, axis=1),
        np.append(R2, t1.T, axis=1),
        np.append(R2, t2.T, axis=1)
    ]

    return R,t,RTsolution


def sanjiaohua(R,T, x1y1,x2y2,K,pts1,pts2):
        I = np.eye(3)
        IO = np.append(I,np.zeros((3,1)),axis = 1 )
        x1y1 = np.append(x1y1, np.zeros((x1y1.shape[0], 1)),axis = 1)
        x2y2 = np.append(x2y2, np.zeros((x1y1.shape[0], 1)),axis = 1)
        all_pts= []
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # P1 = K[I|0]
        P2 = K @ np.hstack((R, t.reshape(3, 1)))  # P2 = K[R|t]

        for i in range(min(x1y1.shape[0],x2y2.shape[0])):
            x1 = x1y1[i,0]
            x2 = x2y2[i,0]
            y1 = x1y1[i, 1]
            y2 = x2y2[i, 1]

            A = np.zeros((4,4))
            A[0] = x1 * P1[2] - P1[0]
            A[1] = y1 * P1[2] - P1[1]
            A[2] = x2 * P2[2] - P2[0]
            A[3] = y2 * P2[2] - P2[1]

            u,s,v  = np.linalg.svd(A)
            p = v[-1]
            p = p/p[3]

            all_pts.append(p)

        return all_pts

        #
        # for RT in RT:
        #     pts = []
        #     count = 0
        #
        #     for i in range(max(x1y1.shape[0],x2y2.shape[0])):
        #         #对每个点进行三角化
        #         x1 = x1y1[i,0]
        #         x2 = x2y2[i,0]
        #         y1 = x1y1[i,1]
        #         y2 = x2y2[i,1]
        #         A2 = np.array([[0, -1, y2],
        #                        [1, 0, -x2],
        #                        [-y2, x2,0]]) @  K @ RT
        #
        #         A1 = np.array([[0, -1, y1],
        #                        [1, 0 , -x1],
        #                        [-y1,x1, 0]]) @ K @ IO
        #
        #         A = np.append(A1, A2, axis = 0)
        #         u,s,v = np.linalg.svd(A)
        #         p = v[:,2]
        #         p = p / p[3]
        #         pts.append(p)
        #         if p[2] > 0:
        #             count += 1
        #
        #     all_pts.append(pts)
        #     counts.append(count)

        # index = np.argmax(counts)
        # return  RTsolution[index],all_pts[index]


def showCameraTrack(R,t):
        x_axis2 = R[:,0]
        y_axis2 = R[:,1]
        z_axis2 = R[:,2]
        #列向量正交，可近似为坐标轴向量

        position1 = np.zeros((3,1))

        position2 = t


        x_axis1 = np.diag((1,1,1))[:,0]
        y_axis1 = np.diag((1,1,1))[:,1]
        z_axis1 = np.diag((1,1,1))[:,2]

        figure = plt.figure(figsize=(10,8))
        ax = figure.add_subplot(111,projection='3d')
        ax.scatter(*position1, c = 'red',s = 50)
        ax.scatter(*position2 , c = 'red', s = 50)
        ax.quiver(*position2, *x_axis2, color = 'r', label = 'x_axis')
        ax.quiver(*position2, *y_axis2, color = 'g', label = 'y_axis')
        ax.quiver(*position2, *z_axis2, color = 'b', label = 'z_axis')

        ax.quiver(*position1, *x_axis1, color='r', )
        ax.quiver(*position1, *y_axis1, color='g',)
        ax.quiver(*position1, *z_axis1, color='b')


        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.legend()
        plt.show()

def create_camera_frustum(RTsolution):
    return None



def BuddleAdjustment():

    return None


def ThreeDpoints(all_pts,):
    all_pts = np.array(all_pts)
    x = all_pts[:,0]
    y = all_pts[:,1]
    z = all_pts[:,2]
    fig= plt.figure(figsize = (5,5))
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(x, y, z , c = 'red', s = 0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    return None



if __name__ == '__main__':
    K = get_innerCamerParamter(path=file_path)
    kps1, feature1 = detectAndDescribe(file_path)
    kps2, feature2 = detectAndDescribe(file_path2)
    matches,good = Bmatch(kps1,kps2,feature1, feature2)
    #print(matches)
    drawMatches(file_path, file_path2, kps1,kps2, good)
    x1y1,x2y2,pts1,pts2 = xymatch(kps1,kps2,matches)
    f = fundamentalMatrix(x1y1,x2y2)
    e,mask = essentialMatrix(x1y1, x2y2,  K)
    u,s,v = np.linalg.svd(e,full_matrices=True)
    # R1,R2,t1,t2, RTsolution = RT(u,s,v)

    r,t,RTsolution = RT(x1y1, x2y2, e,mask, K ,u,s,v)

    print(r,t)
    pts = sanjiaohua(r,t,x1y1, x2y2, K,pts1,pts2)
    # print(RT)
    showCameraTrack(r,t)

    ThreeDpoints(all_pts = pts)


