import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


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


def img_cc(img):
    # 定义纵向过滤器
    vertical_filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    # 定义横向过滤器
    horizontal_filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

    # 读取纸风车的示例图片“pinwheel.jpg”

    print(img.shape)

    # 得到图片的维数
    n, m, d = img.shape

    # 初始化边缘图像
    edges_img = hisEqulColor2(img)

    # 循环遍历图片的全部像素
    for row in range(3, n - 2):
        for col in range(3, m - 2):
            # 在当前位置创建一个 3x3 的小方框
            local_pixels = img[row - 1:row + 2, col - 1:col + 2, 0]

            # 应用纵向过滤器
            vertical_transformed_pixels = vertical_filter * local_pixels
            # 计算纵向边缘得分
            vertical_score = vertical_transformed_pixels.sum() / 4

            # 应用横向过滤器
            horizontal_transformed_pixels = horizontal_filter * local_pixels
            # 计算横向边缘得分
            horizontal_score = horizontal_transformed_pixels.sum() / 4

            # 将纵向得分与横向得分结合，得到此像素总的边缘得分
            edge_score = (vertical_score ** 2 + horizontal_score ** 2) ** .5

            # 将边缘得分插入边缘图像中
            edges_img[row, col] = [edge_score] * 3

    # 对边缘图像中的得分值归一化，防止得分超出 0-1 的范围
    edges_img = edges_img / edges_img.max()

    return edges_img


def load_data(data_dir):
    """Load the data."""
    print('train_dir', data_dir)
    train_source = data_dir
    save_dir = os.path.join(os.getcwd(), 'saved_images')
    print(train_source, os.listdir(train_source))
    for class_name in os.listdir(train_source):
        print('class_name', class_name)
        # class_name=u(class_name)
        if os.path.isdir(os.path.join(train_source, class_name)):
            for img_name in os.listdir(os.path.join(train_source, class_name)):
                print('img_name:', os.path.join(train_source, class_name, img_name))
                upath = os.path.join(train_source, class_name, img_name)
                img_data = cv2.imread(upath)  # 读取图像
                img_data = img_cc(img_data)
                img_data = img_data * 225
                plt.imshow(img_data)
                plt.show()
                filepath = os.path.join(save_dir, class_name, img_name)
                print(filepath)

                cv2.imwrite(filepath, img_data)

    return save_dir


ds = load_data('/imagedata')  # 加载岩石薄片数据
print(ds)
