# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 13:18:23 2020
@author: Administrator
"""

from __future__ import print_function
import keras
# import torch
import random
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, GlobalAveragePooling2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from keras.layers.core import Lambda
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# 加载薄片图像数据：训练数据和验证数据
num_crop = 15
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
                    (b, g, r) = cv2.split(img_data)
                    bH = cv2.equalizeHist(b)
                    gH = cv2.equalizeHist(g)
                    rH = cv2.equalizeHist(r)
                    img_data = cv2.merge((bH, gH, rH))  # 合并通道
                    for i in range(num_crop):  # 随机截取num_crop个图片
                        crop_img = random_crop_img(img_data)
                        # new_image = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)
                        new_image = cv2.resize(crop_img, (224, 224))
                        if new_image.shape[0] > 0:
                            # crop_img.save('d:/temp/crop'+class+)
                            img.append(new_image)
                            train_b = [int(class_name)]
                            labels.append(train_b)  # or other method to convert label

        plt.imshow(new_image)
        plt.show()
        # Shuffle labels.
        combine1 = list(zip(img, labels))  # zip as list for Python 3
        np.random.shuffle(combine1)
        img, labels = zip(*combine1)

        return [np.array(img),
                np.array(labels)]


# Training parameters
batch_size = 6  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = False
num_classes = 10


# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 2

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
# version = 2

# Computed depth from supplied model parameter n
depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, 2)

# Load the data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# ds = ImageLoader('D:\\Python_project\\Deep_learning\\rock_slice\\imagedata',)  # 加载岩石薄片数据
# (X, Y) = ds.load_data()
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

print('x_train:\n', x_train.shape)
print('x_train:\n', y_train.shape)
# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



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


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet(input_shape, depth, num_classes=num_classes):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    global num_filters_out
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)

            # Calculate global means
            residual_abs = Lambda(abs_backend)(y)
            abs_mean = GlobalAveragePooling2D()(residual_abs)

            # Calculate scaling coefficients
            scales = Dense(num_filters_out, activation=None, kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(abs_mean)
            scales = BatchNormalization()(scales)
            scales = Activation('relu')(scales)
            scales = Dense(num_filters_out, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)
            scales = Lambda(expand_dim_backend)(scales)

            # Calculate thresholds
            thres = keras.layers.multiply([abs_mean, scales])

            # Soft thresholding
            sub = keras.layers.subtract([residual_abs, thres])
            zeros = keras.layers.subtract([sub, sub])
            n_sub = keras.layers.maximum([sub, zeros])
            y = keras.layers.multiply([Lambda(sign_backend)(y), n_sub])

            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


model = resnet(input_shape=input_shape, depth=depth)
"""
base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)

for input, output in data:
    # first forward-backward pass
    loss = loss_function(output, model(input))  # use this loss for any training statistics
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # second forward-backward pass
    loss_function(output, model(input)).backward()
    optimizer.second_step(zero_grad=True)
"""

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = "cifar10_%s_model.{epoch:03d}.h5" % model_type
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

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
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

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# 可视化模型
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    predictions = model.predict_classes(x_val)
    truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))

