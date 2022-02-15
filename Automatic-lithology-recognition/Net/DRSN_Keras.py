#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 23:24:05 2019

Implemented using TensorFlow 1.0.1 and Keras 2.2.1
 
M. Zhao, S. Zhong, X. Fu, et al., Deep Residual Shrinkage Networks for Fault Diagnosis, 
IEEE Transactions on Industrial Informatics, 2019, DOI: 10.1109/TII.2019.2943898

There might be some problems in the Keras code. The weights in custom layers of models created using the Keras functional API may not be optimized.
https://www.reddit.com/r/MachineLearning/comments/hrawam/d_theres_a_flawbug_in_tensorflow_thats_preventing/

TensorFlow被曝存在严重bug，搭配Keras可能丢失权重
https://cloud.tencent.com/developer/news/661458

The TFLearn code is recommended for usage.
https://github.com/zhao62/Deep-Residual-Shrinkage-Networks/blob/master/DRSN_TFLearn.py

@author: super_9527
"""

from __future__ import print_function
import os
import random
import cv2
import tensorflow as tf
import keras
import numpy as np
from keras.utils import np_utils
from tensorflow.keras.datasets import cifar10
# from tflearn.datasets import cifar10
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

K.set_learning_phase(1)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# Training parameters
batch_size = 2000  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = True
num_classes = 10

# Input image dimensions
img_rows, img_cols = 32, 32

# 加载薄片图像数据：训练数据和验证数据
# 裁剪数量和裁剪尺寸
num_crop = 5
size_img = 400


# 随机截取一个图片
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


# 图像加载
class ImageLoader:
    """Load images in arrays without batches."""

    def __init__(self, data_dir):
        """Create class."""
        self.data_dir = data_dir

    def load_data(self):
        """Load the data."""
        print('self.train_dir', self.data_dir)
        train_source = self.data_dir
        img, labels = [], []
        print(train_source, os.listdir(train_source))
        for class_name in os.listdir(train_source):
            print('class_name', class_name)
            # class_name=u(class_name)
            if os.path.isdir(os.path.join(train_source, class_name)):
                for img_name in os.listdir(os.path.join(train_source, class_name)):
                    print('img_name:', os.path.join(train_source, class_name, img_name))
                    upath = os.path.join(train_source, class_name, img_name)
                    img_data = cv2.imread(upath)  # 读取图像
                    # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
                    # (b, g, r) = cv2.split(img_data)
                    # bH = cv2.equalizeHist(b)
                    # gH = cv2.equalizeHist(g)
                    # rH = cv2.equalizeHist(r)
                    # img_data = cv2.merge((bH, gH, rH))  # 合并通道
                    for i in range(num_crop):  # 随机截取num_crop个图片
                        crop_img = random_crop_img(img_data)
                        # new_image = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)
                        new_image = cv2.resize(crop_img, (img_rows, img_cols))
                        if new_image.shape[0] > 0:
                            # crop_img.save('d:/temp/crop'+class+)
                            img.append(new_image)
                            train_b = [int(class_name)]
                            labels.append(train_b)  # or other method to convert label

        # plt.imshow(new_image)
        # plt.show()
        # Shuffle labels.
        combine1 = list(zip(img, labels))  # zip as list for Python 3
        np.random.shuffle(combine1)
        img, labels = zip(*combine1)

        return [np.array(img),
                np.array(labels)]




# The data, split between train and test sets
# ds = ImageLoader('D:\\Python_project\\Deep_learning\\rock_slice\\imagedata', )  # 加载岩石薄片数据
# (X, Y) = ds.load_data()
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

# Noised data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


def abs_backend(inputs):
    return K.abs(inputs)


def expand_dim_backend(inputs):
    return K.expand_dims(K.expand_dims(inputs, 1), 1)


def sign_backend(inputs):
    return K.sign(inputs)


def pad_backend(inputs, in_channels, out_channels):
    pad_dim = (out_channels - in_channels) // 2
    inputs = K.expand_dims(inputs, -1)
    inputs = K.spatial_3d_padding(inputs, ((0, 0), (0, 0), (pad_dim, pad_dim)), 'channels_last')
    return K.squeeze(inputs, -1)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


# Residual Shrinakge Block
def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                             downsample_strides=2):
    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    for i in range(nb_blocks):

        identity = residual

        if not downsample:
            downsample_strides = 1

        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(out_channels, 3, strides=(downsample_strides, downsample_strides),
                          padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))(residual)

        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(out_channels, 3, padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4))(residual)

        # Calculate global means
        residual_abs = Lambda(abs_backend)(residual)
        abs_mean = GlobalAveragePooling2D()(residual_abs)

        # Calculate scaling coefficients
        scales = Dense(out_channels, activation=None, kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(abs_mean)
        # scales = tf.keras.layers.conv2d(scales, filters=out_channels, kernel_size=1)
        scales = BatchNormalization()(scales)
        scales = Activation('relu')(scales)
        # scales = Dense(out_channels, activation='relu', kernel_initializer='he_normal', use_bias=False)(scales)
        scales = Dense(out_channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)
        scales = Lambda(expand_dim_backend)(scales)

        # Calculate thresholds
        thres = tf.keras.layers.multiply([abs_mean, scales])

        # Soft thresholding
        sub = tf.keras.layers.subtract([residual_abs, thres])
        zeros = tf.keras.layers.subtract([sub, sub])
        n_sub = tf.keras.layers.maximum([sub, zeros])
        residual = tf.keras.layers.multiply([Lambda(sign_backend)(residual), n_sub])

        # Downsampling using the pooL-size of (1, 1)
        if downsample_strides > 1:
            identity = AveragePooling2D(pool_size=(1, 1), strides=(2, 2))(identity)

        # Zero_padding to match channels
        if in_channels != out_channels:
            identity = Lambda(pad_backend, arguments={'in_channels': in_channels, 'out_channels': out_channels})(
                identity)

        residual = tf.keras.layers.add([residual, identity])

    return residual


# define and train a model
inputs = Input(shape=input_shape)
net = Conv2D(16, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
net = residual_shrinkage_block(net, 10, 16, downsample=True)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = GlobalAveragePooling2D()(net)
outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(net)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = "Deep-Residual-Shrinkage-Networks_model.{epoch:03d}.h5"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)


# model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(x_test, y_test))

# get results
K.set_learning_phase(0)
DRSN_train_score = model.evaluate(x_train, y_train, batch_size=100, verbose=0)
print('Train loss:', DRSN_train_score[0])
print('Train accuracy:', DRSN_train_score[1])
DRSN_test_score = model.evaluate(x_test, y_test, batch_size=100, verbose=0)
print('Test loss:', DRSN_test_score[0])
print('Test accuracy:', DRSN_test_score[1])
