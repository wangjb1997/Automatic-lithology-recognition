# 加载薄片图像数据：训练数据和验证数据
import random
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

num_crop = 30
size_img = 692


# 随机截取一个图片
def random_crop_img(img):
    crop_img = np.zeros([0, 0, 0])
    limit_img = [random.randint(1, img.shape[0]), random.randint(1, img.shape[1])]     # 随机生成剪切区域
    IMG_Exists = (limit_img[0] + size_img <= img.shape[0] and limit_img[1] + size_img <= img.shape[1])
    while not IMG_Exists:  # 判断是否成功截取
        limit_img = [random.randint(1, img.shape[0]), random.randint(1, img.shape[1])]
        IMG_Exists = (limit_img[0] + size_img <= img.shape[0] and limit_img[1] + size_img <= img.shape[1])
    crop_img = img[limit_img[0]:limit_img[0] + size_img, limit_img[1]:limit_img[1] + size_img]

    return crop_img

# 图像加载
class ImageLoader:
    """Load images in arrays without batches."""
    def __init__(self, data_dir):
        """Create class."""
        self.data_dir = data_dir

    def load_data(self):
        """Load the data."""
        print('self.train_dir', self.data_dir)

        for train_source in [self.data_dir]:
            img, labels = [], []
            print(train_source, os.listdir(train_source))
            for class_name in os.listdir(train_source):
                print('class_name', class_name)
                # class_name=u(class_name)
                if os.path.isdir(os.path.join(train_source, class_name)):
                    for img_name in os.listdir(os.path.join(train_source, class_name)):
                        print('img_name:', os.path.join(train_source, class_name, img_name))
                        upath = os.path.join(train_source, class_name, img_name)
                        img = cv2.imread(upath)  # np.fromfile(upath,dtype=np.uint8))
                        for i in range(num_crop):  # 随机截取num_crop个图片
                            crop_img = random_crop_img(img)
                            # new_image = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)
                            new_image = cv2.resize(crop_img, (224, 224))
                            if new_image.shape[0] > 0:
                                # crop_img.save('d:/temp/crop'+class+)
                                plt.imshow(new_image)
                                img.append(new_image)
                                train_b = [int(class_name)]
                                labels.append(train_b)  # or other method to convert label

        # Shuffle labels.
        combine1 = list(zip(img, labels))  # zip as list for Python 3
        np.random.shuffle(combine1)
        img, labels = zip(*combine1)

        return [[np.array(img),
                 np.array(labels)],]
