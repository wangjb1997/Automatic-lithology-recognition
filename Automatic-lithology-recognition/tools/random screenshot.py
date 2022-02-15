import random
from PIL import Image
import os
import cv2
"""
num_crop = 10
size_img = 692

dir = "C:\\Users\AUSU\\Desktop\\coop"  #目标图片文件夹的路径
path = []    #图片的路径列表
path_key = 0  #每张图片的路径下标
count = 1  #总共裁剪的图片
num = 4  ##从每张图片裁剪的数量
cut_size=448

for root, dirs, files in os.walk(dir):
    for file in files:
        path.append((os.path.join(root,file)))
        image_path = path[path_key]
        img = Image.open(image_path)  # 打开当前路径图像
        for num_eve in range(num):
            limit_img = [random.randint(1, img.size[0]), random.randint(1, img.size[1])]
            #随机生成剪切区域
            Picture_Exists = (limit_img[0] + size_img <= img.shape[0] and limit_img[1] + size_img <= img.shape[1])
            while not Picture_Exists:  # 判断是否成功截取
                limit_img = [random.randint(1, img.shape[0]), random.randint(1, img.shape[1])]
                Picture_Exists = (limit_img[0] + size_img <= img.shape[0] and limit_img[1] + size_img <= img.shape[1])
            crop_img = img[limit_img[0]:limit_img[0] + size_img, limit_img[1]:limit_img[1] + size_img]


            box1 = (limit_img[0], limit_img[1], limit_img[0] + cut_size, limit_img[1] + cut_size)
                # 设置图像裁剪区域 (x左上，y左上，x右下,y右下)
            image1 = img.crop(box1)  # 图像裁剪
            image1.save('D:/Python_project/Deep_learning/rock_slice/imagedata/train/0/0_%s.jpg' % str(count))
            # 存储裁剪得到的图像
            print("successfully save %sth pic" % str(count))  #
            count += 1
        path_key += 1



def random_crop_img(img):
    # crop_img = np.zeros([0, 0, 0])
    limit_img = [random.randint(1, img.shape[0]), random.randint(1, img.shape[1])]
    # 随机生成剪切区域
    Picture_Exists = (limit_img[0] + size_img <= img.shape[0] and limit_img[1] + size_img <= img.shape[1])
    while not Picture_Exists:  # 判断是否成功截取
        limit_img = [random.randint(1, img.shape[0]), random.randint(1, img.shape[1])]
        Picture_Exists = (limit_img[0] + size_img <= img.shape[0] and limit_img[1] + size_img <= img.shape[1])
    crop_img = img[limit_img[0]:limit_img[0] + size_img, limit_img[1]:limit_img[1] + size_img]

    return crop_img


count = 1
dir = "C:\\Users\AUSU\\Desktop\\coop"  # 目标图片文件夹的路径
for img_name in os.listdir(dir):
    print('img_name:', os.path.join(dir, img_name))
    upath = os.path.join(dir, img_name)
    img = cv2.imread(upath)  # np.fromfile(upath,dtype=np.uint8))
    for i in range(num_crop):  # 随机截取num_crop个图片
        crop_img = random_crop_img(img)
        crop_img.save('D:/Python_project/Deep_learning/rock_slice/imagedata/train/0/0_%.jpg' % str(count))
        count += 1


import os
def ranamesJPG(filepath, kind):
    images = os.listdir(filepath)
    count=1
    for name in images:
        count = count+1
        os.rename(filepath+name, filepath+kind+'_'+'aux'+ count +'.jpg')
        print(name)
        #print(name.split('.')[0])
ranamesJPG('C:\\Users\\jianbang\\Desktop\\Data\\train\\1','1')



import os
# path为批量文件的文件夹的路径
path = 'C:\\Users\\jianbang\\Desktop\\Data\\test\\4'

# 文件夹中所有文件的文件名
file_names = os.listdir(path)

# 外循环遍历所有文件名，内循环遍历每个文件名的每个字符
for name in file_names:
    for s in name:
        if s == '_':
            index_num = name.index(s)  # index_num为要删除的位置索引

            # 采用字符串的切片方式删除编号
            os.renames(os.path.join(path, name), os.path.join(path, name[index_num + 1:]))
            break  # 重命名成功，跳出内循环
"""
import os

# 统一图片类型
def ranamesJPG(filepath, kind):
    images = os.listdir(filepath)
    i = 1
    for name in images:
        rename = str(i)
        os.rename(filepath+name, filepath+kind+'_'+rename+'.jpg')
        print(name)
        print(name.split('.')[0])
        i += 1
ranamesJPG("/imagedata/5\\", '5')



