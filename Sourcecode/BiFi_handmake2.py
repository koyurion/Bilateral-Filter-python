import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import math

# 双边滤波器：灰度图片
def bifiGray(A, N, sigma):
    sigma_d, sigma_r = sigma[0], sigma[1]
    # kernel distance matrix
    Y, X = np.mgrid[-N:N + 1, -N:N + 1]
    # domain kernel
    G = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma_d ** 2))
    # bilateral filter
    dim = A.shape
    B = np.zeros(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            iMin = max(i - N, 0)
            iMax = min(i + N + 1, dim[0])
            jMin = max(j - N, 0)
            jMax = min(j + N + 1, dim[1])
            I = A[iMin:iMax, jMin:jMax]
            # value kernel
            H = np.exp(-(I - A[i, j]) ** 2 / (2 * sigma_r ** 2))
            # BiFi
            itmp = - i + N
            jtmp = - j + N
            G_tmp = G[iMin + itmp:iMax + itmp, jMin + jtmp:jMax + jtmp]
            F = np.multiply(H, G_tmp)
            B[i, j] = sum(sum(np.multiply(F, I))) / sum(sum(F))
    return B

# 双边滤波器：彩色图片
def bifiColor(A, N, sigma):
    A = cv2.cvtColor(A, cv2.COLOR_BGR2LAB)
    sigma_d, sigma_r = sigma[0], sigma[1]
    # kernel distance matrix
    Y, X = np.mgrid[-N:N + 1, -N:N + 1]
    # domain kernel
    G = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma_d ** 2))

    sigma_r = 100 * sigma_r
    # bilateral filter
    dim = A.shape
    B = np.zeros(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            iMin = max(i - N, 0)
            iMax = min(i + N + 1, dim[0])
            jMin = max(j - N, 0)
            jMax = min(j + N + 1, dim[1])
            I = A[iMin:iMax, jMin:jMax]
            # value kernel
            d_list = []
            for l in range(len(dim)):
                d = I[:, :, l] - A[i, j, l]
                d_list.append(d)
            d_list = np.array(d_list) ** 2
            H = np.exp(-(sum(d_list)) / (2 * sigma_r ** 2))
            # BiFi
            itmp = - i + N
            jtmp = - j + N
            G_tmp = G[iMin + itmp:iMax + itmp, jMin + jtmp:jMax + jtmp]
            F = np.multiply(H, G_tmp)
            F_tmp = sum(sum(F))
            for l in range(len(dim)):
                B[i, j, l] = sum(sum(np.multiply(F, I[:, :, l]))) / F_tmp
    B = cv2.cvtColor(B.astype(np.float32), cv2.COLOR_Lab2BGR)
    return B

# 参数处理，异常处理
def params_process(N, sigma):
    # deal with N
    if N < 1 or N is None:
        N = 5
    N = math.ceil(N)

    # deal with sigma
    if len(sigma) != 2 or sigma[0] <= 0 or sigma[1] <= 0:
        print("The sigma is invalid.So sigma is changed into [3, 0.1]")
        sigma = [3, 0.1]

    return N, sigma

# 画图（原图+处理后）、保存图片（可选）
def draw_img(I, I_new, sigma, saveimg=False, imgpath=None, img_cmap="gray"):
    high, wide = 1, 2
    plt.subplot(high, wide, 1)
    plt.imshow(I, cmap=img_cmap)
    plt.axis('off')
    plt.title("Original")

    plt.subplot(high, wide, 2)
    plt.imshow(I_new, cmap=img_cmap)
    plt.axis('off')
    plt.title("sigma_d: " + str(sigma[0]) + " sigma_r:" + str(sigma[1]))
    if saveimg:
        if imgpath is not None:
            plt.savefig(imgpath)
            print("tte image is saved at: " + imgpath)
        else:
            print("Error: the saved image path is Nonetype")
    plt.show()

# 主函数：读取图片，选择不同的双边滤波器，异常处理
# 默认选择： 不保存图片
def main(image, image_flag, saveimg=False, imgpath=None, N=5, sigma_d=3, sigma_r=0.1):
    I = cv2.imread(image, image_flag)
    if I is None:
        print("Error: The image is NoneType")
    else:
        print("bilateral filter is working...")

        N, sigma = params_process(N, [sigma_d, sigma_r])
        I = I.astype(np.float32) / 255
        # choose the bilateral filter
        if image_flag == 0:
            I_new = bifiGray(I, N, sigma)
        else:
            I_new = bifiColor(I, N, sigma)
        # draw the images
        draw_img(I, I_new, sigma, saveimg, imgpath)
        print("bilateral filter done")

if __name__ == "__main__":
    image = "Images/c256_001/cara1.ppm"   # 图片地址
    image_flag = 1  # 0                   # 灰度图片： 0 ；彩色图片 ：1
    main(image, image_flag)
