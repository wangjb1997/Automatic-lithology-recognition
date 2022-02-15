import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
img = cv2.imread("imagedata/3/3_rockslice92.jpg", 1)
cv2.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst", result)

cv2.waitKey(0)
'''


# 绘制整幅图像的直方图
def whole_hist(image):
    plt.hist(image.ravel(), 256, [0, 256])  # numpy的ravel函数功能是将多维数组降为一维数组
    plt.show()


# 画三通道图像的直方图
def channel_hist(image):
    color = ('b', 'g', 'r')  # 这里画笔颜色的值可以为大写或小写或只写首字母或大小写混合
    for i, color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])  # 计算直方图
        plt.plot(hist, color)
        plt.xlim([0, 256])
    plt.show()


# 彩色图像全局直方图均衡化
def hisEqulColor1(img):
    # 将RGB图像转换到YCrCb空间中
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # 将YCrCb图像通道分离
    channels = cv2.split(ycrcb)
    # 对第1个通道即亮度通道进行全局直方图均衡化并保存
    cv2.equalizeHist(channels[0], channels[0])
    # 将处理后的通道和没有处理的两个通道合并，命名为ycrcb
    cv2.merge(channels, ycrcb)
    # 将YCrCb图像转换回RGB图像
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


# 彩色图像进行自适应直方图均衡化，代码同上的地方不再添加注释
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


def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    # cv2.imshow("custom_blur_demo", dst)
    return dst


img = cv2.imread("../imagedata/6/8_220.jpg", 1)
cv2.imshow("src", img)

# img1 = img.copy()
# img2 = img.copy()
res = hisEqulColor2(img)
cv2.imshow("dst", res)
# res2 = hisEqulColor2(img2)
res1 = custom_blur_demo(res)

cv2.imshow("dst1", res1)

cv2.waitKey(0)

whole_hist(img)
channel_hist(img)
whole_hist(res1)
channel_hist(res1)



