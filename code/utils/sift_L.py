import cv2
import numpy as np

def convolve(filter, mat, padding, strides):
    result = None
    filter_size = filter.shape
    mat_size = mat.shape
    if len(filter_size) == 2:
        if len(mat_size) == 3:
            channel = []
            for i in range(mat_size[-1]):
                pad_mat = np.pad(mat[:, :, i], ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
                temp = []
                for j in range(0, pad_mat.shape[0] - filter_size[0] + 1, strides[1]):
                    temp.append([])
                    for k in range(0, pad_mat.shape[1] - filter_size[1] + 1, strides[0]):
                        val = (filter * pad_mat[j:j + filter_size[0], k:k + filter_size[1]]).sum()
                        temp[-1].append(val)
                channel.append(np.array(temp))

            channel = tuple(channel)
            result = np.dstack(channel)
        elif len(mat_size) == 2:
            channel = []
            pad_mat = np.pad(mat, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
            for j in range(0, pad_mat.shape[0] - filter_size[0] + 1, strides[1]):
                channel.append([])
                for k in range(0, pad_mat.shape[1] - filter_size[1] + 1, strides[0]):
                    val = (filter * pad_mat[j:j + filter_size[0], k:k + filter_size[1]]).sum()
                    channel[-1].append(val)
            result = np.array(channel)
    return result

def downsample(img, step = 2):
    return img[::step, ::step]


def GaussianKernel(sigma, dim):
    '''
    :param sigma: Standard deviation
    :param dim: dimension(must be positive and also an odd number)
    :return: return the required Gaussian kernel.
    '''
    temp = [t - (dim // 2) for t in range(dim)]
    assistant = []
    for i in range(dim):
        assistant.append(temp)
    assistant = np.array(assistant)
    temp = 2 * sigma * sigma
    result = (1.0 / (temp * np.pi)) * np.exp(-(assistant ** 2 + (assistant.T) ** 2) / temp)
    # Normalize the kernel to sum to 1
    return result / np.sum(result)


def getDoG(img, n, sigma0, S=None, O=None):
    '''
    :param img: the original img.
    :param sigma0: sigma of the first stack of the first octave. default 1.52 for complicate reasons.
    :param n: how many stacks of feature that you wanna extract.
    :param S: how many stacks does every octave have. S must bigger than 3.
    :param O: how many octaves do we have.
    :return: the DoG Pyramid
    '''
    if S is None:
        S = 4
    if O is None:
        O = int(np.log2(min(img.shape[0], img.shape[1]))) - 3

    k = 2 ** (1.0 / n)
    sigma = [[(k ** s) * sigma0 * (1 << o) for s in range(S)] for o in range(O)]
    samplePyramid = [downsample(img, 1 << o) for o in range(O)]

    GaussianPyramid = []
    for i in range(O):
        GaussianPyramid.append([])
        for j in range(S):
            dim = int(6 * sigma[i][j] + 1)
            if dim % 2 == 0:
                dim += 1
            kernel = GaussianKernel(sigma[i][j], dim)
            GaussianPyramid[-1].append(
                convolve(kernel, samplePyramid[i], [dim // 2, dim // 2, dim // 2, dim // 2], [1, 1]))

    DoG = [[GaussianPyramid[o][s + 1] - GaussianPyramid[o][s] for s in range(S - 1)] for o in range(O)]
    return DoG, GaussianPyramid


def normalize_for_display(img):
    '''Normalize image to 0-255 for display'''
    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img_normalized.astype(np.uint8)


if __name__ == '__main__':
    path = '61.jpg'
    img = cv2.imread(path)

    if img is None:
        print(f"Error: Could not load image from {path}")
        exit(1)

    # Convert to grayscale for SIFT
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray_img = img.astype(np.float32)

    SIFT_SIGMA = 1.6
    SIFT_INIT_SIGMA = 0.5
    sigma0 = np.sqrt(SIFT_SIGMA ** 2 - SIFT_INIT_SIGMA ** 2)
    n = 2

    DoG, GaussianPyramid = getDoG(gray_img, n, sigma0)

    # Display Gaussian Pyramid
    for o, octave in enumerate(GaussianPyramid):
        for s, layer in enumerate(octave):
            display_img = normalize_for_display(layer)
            cv2.imshow(f'Gaussian Octave {o}, Layer {s}', display_img)
            cv2.imwrite(f'../data/Gaussian Octave {o} Layer {s}.jpg',display_img)
            cv2.waitKey(100)  # Short delay to see images

    # Display DoG Pyramid
    for o, octave in enumerate(DoG):
        for s, layer in enumerate(octave):
            display_img = normalize_for_display(layer)
            cv2.imshow(f'DoG Octave {o}, Layer {s}', display_img)
            cv2.imwrite(f'../data/DoG Octave {o} Layer {s}.jpg', display_img)
            cv2.waitKey(100)  # Short delay to see images

    cv2.waitKey(0)
    cv2.destroyAllWindows()



