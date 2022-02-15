import cv2
import numpy as np
from matplotlib import pyplot as plt


def img_show(name, image):
    """matplotlib图像显示函数
    name：字符串，图像标题
    img：numpy.ndarray，图像
    """
    #if len(image.shape) == 3:
    #   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(name, fontproperties='Times new roman', fontsize=12)


def hisEqulColor2(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)

    # 以下代码详细注释见官网：
    # https://docs.opencv.org/4.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])

    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


if __name__ == "__main__":
    img = cv2.imread("../test_data/1.jpg", 0)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    # print(hist)

    # 查看累积直方图
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    # print(cdf_normalized)

    # 均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equ = clahe.apply(img)

    hist1, bins1 = np.histogram(equ.flatten(), 256, [0, 256])
    # print(hist)

    # 查看均衡化后的累积直方图
    cdf1 = hist1.cumsum()
    cdf_normalized1 = cdf1 * hist1.max() / cdf1.max()

    # 显示图像结果
    plt.figure(figsize=(10, 8), dpi=100)
    plt.subplot(221)
    img_show('The original image', img)
    plt.subplot(222)
    img_show('Adaptive equalization', equ,)
    plt.subplot(223)
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlabel('histogram (red) and cumulative histogram (blue)', fontproperties='Times new roman', fontsize=15)
    plt.xlim([0, 256])
    plt.legend(('cdf', 'historgram'), loc='upper left')
    plt.subplot(224)
    plt.plot(cdf_normalized1, color='b')
    plt.hist(equ.flatten(), 256, [0, 256], color='r')
    plt.xlabel('Adaptive equalization histogram', fontproperties='Times new roman', fontsize=15)
    plt.xlim([0, 256])
    plt.legend(('cdf', 'historgram'), loc='upper left')
    plt.show()

    cv2.imshow("dst1", equ)
    cv2.waitKey(0)
