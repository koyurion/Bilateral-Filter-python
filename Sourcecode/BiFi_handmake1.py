import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import math


def bfltGray(A, N, sigma):
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
            iMax = min(i + N, dim[0])
            jMin = max(j - N, 0)
            jMax = min(j + N, dim[1])
            I = A[iMin:iMax, jMin:jMax]
            # value kernel
            H = np.exp(-(I - A[i, j]) ** 2 / (2 * sigma_r ** 2))
            # BiFi
            itmp = - i + N + 1
            jtmp = - j + N + 1
            F = np.multiply(H, G[iMin + itmp:iMax + itmp, jMin + jtmp:jMax + jtmp])
            B[i, j] = sum(sum(np.multiply(F, I))) / sum(sum(F))
    return B


def bfltColor(A, N, sigma):
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
            iMax = min(i + N, dim[0])
            jMin = max(j - N, 0)
            jMax = min(j + N, dim[1])
            I = A[iMin:iMax, jMin:jMax]
            # value kernel
            dL = I[:, :, 0] - A[i, j, 0]
            da = I[:, :, 1] - A[i, j, 1]
            db = I[:, :, 2] - A[i, j, 2]
            H = np.exp(-(dL ** 2 + da ** 2 + db ** 2) / (2 * sigma_r ** 2))

            # BiFi
            itmp = - i + N + 1
            jtmp = - j + N + 1
            F = np.multiply(H, G[iMin + itmp:iMax + itmp, jMin + jtmp:jMax + jtmp])
            B[i, j, 0] = sum(sum(np.multiply(F, I[:, :, 0]))) / sum(sum(F))
            B[i, j, 1] = sum(sum(np.multiply(F, I[:, :, 1]))) / sum(sum(F))
            B[i, j, 2] = sum(sum(np.multiply(F, I[:, :, 2]))) / sum(sum(F))

    B = cv2.cvtColor(B.astype(np.float32), cv2.COLOR_Lab2BGR)
    return B


def bfilter2(A, N, sigma):
    # deal with N
    if N < 1 or N is None:
        N = 5
    N = math.ceil(N)

    # deal with sigma
    if len(sigma) != 2 or sigma[0] <= 0 or sigma[1] <= 0:
        print("The sigma is invalid.So sigma is changed into [3, 0.1]")
        sigma = [3, 0.1]

    if len(A.shape) == 3:
        B = bfltColor(A, N, sigma)
    else:
        B = bfltGray(A, N, sigma)
    return B


if __name__ == "__main__":
    path = "Images/gnocuad_001/"
    imgs_parent = []
    high, wide = 3,4
    count=1
    for i in os.listdir(path):


        img = path + i
        I = cv2.imread(img)
        # print(I.shape)  # 256 256 3

        if I is None:
            print("Error: The image is NoneType")
        else:
            I = I.astype(np.float32) / 255
            '''img = I[:, :, (2, 1, 0)]
            plt.subplot(high, wide, count)
            count+=1
            plt.imshow(img,cmap="gray")
            plt.axis('off')
            plt.title("Original")'''

            N = [5]  # bilateral filter half-width
            sigma = [3, 0.1]  # bilateral filter standard deviations: sigma_d, sigma_r
            sigma_d = [3]
            sigma_r = [0.1]
            # count = 1
            for w1 in N:
                for i in sigma_d:
                    for j in sigma_r:
                        sigma = [i, j]
                        I1 = bfilter2(I, w1, sigma)
                        img1 = I1[:, :, (2, 1, 0)]
                        plt.subplot(high, wide, count)
                        count += 1
                        plt.imshow(img1,cmap="gray")
                        plt.axis('off')
                        if count==1:
                            plt.title("After bilateral filter")
    plt.savefig("cb5.png",dpi=500)
    plt.show()


