import numpy as np
import os
import tensorflow as tf
import keras
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes=n_holes
        self.length=length
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h=img.shape[0]
        w=img.shape[1]
        mask=np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y=np.random.randint(h)
            x=np.random.randint(w)
            y1=np.clip(y-self.length//2,0,h)
            y2=np.clip(y+self.length//2,0,h)
            x1=np.clip(x-self.length//2,0,w)
            x2=np.clip(x+self.length//2,0,w)
            mask[y1: y2, x1: x2] = 0.
        mask=np.expand_dims(mask,axis=-1)
        img=img*mask
        return img


class DatasetReader(object):
    def __init__(self, data_path, image_size=None):
        self.data_path = data_path
        self.img_size = image_size
        self.img_size.append(3)
        self.train_path = os.path.join(data_path, "train")
        self.test_path = os.path.join(data_path, "test")
        self.TF_path = os.path.join(data_path, "TFRecordData")
        self.tf_train_path = os.path.join(self.TF_path, "train")
        self.tf_test_path = os.path.join(self.TF_path, "test")
        self.classes = os.listdir(self.train_path)
        self.__Makedirs()
        self.train_batch_initializer = None
        self.test_batch_initializer = None
        self.__CreateTFRecord(self.train_path, self.tf_train_path)
        self.__CreateTFRecord(self.test_path, self.tf_test_path)

    def __CreateTFRecord(self, read_path, save_path):  # 创建TFRecord文件
        path = os.path.join(save_path, "data.TFRecord")
        if os.path.exists(path):
            print("find file " + (os.path.join(save_path, "data.TFRecords")))
            return
        else:
            print("cannot find file %s,ready to recreate" % (os.path.join(save_path, "data.TFRecords")))
        writer = tf.python_io.TFRecordWriter(path=path)
        image_path = []
        image_label = []
        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(read_path, class_name)
            for image_name in os.listdir(class_path):
                image_path.append(os.path.join(class_path, image_name))
                image_label.append(label)
        for i in range(5): image_path, image_label = shuffle(image_path, image_label)
        bar = Progbar(len(image_path))
        for i in range(len(image_path)):
            image, label = Image.open(image_path[i]).convert("RGB"), image_label[i]
            image = image.convert("RGB")
            image = image.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
            }))
            writer.write(example.SerializeToString())
            bar.update(i + 1)
        writer.close()

    def __Makedirs(self):
        if not os.path.exists(self.TF_path):
            os.makedirs(self.TF_path)
        if not os.path.exists(self.tf_train_path):
            os.makedirs(self.tf_train_path)
        if not os.path.exists(self.tf_test_path):
            os.makedirs(self.tf_test_path)

    def __parsed(self, tensor):
        feature = tf.parse_single_example(tensor, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })
        image = tf.decode_raw(feature["image"], tf.uint8)
        image = tf.reshape(image, self.img_size)
        image = tf.random_crop(image, self.img_size)
        image = image / 255

        label = tf.cast(feature["label"], tf.int32)
        label = tf.one_hot(label, len(self.classes))
        return image, label

    def __parsed_distorted(self, tensor):

        feature = tf.parse_single_example(tensor, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })
        image = tf.decode_raw(feature["image"], tf.uint8)
        image = tf.reshape(image, self.img_size)

        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
        image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
        image = image / 255

        random = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
        random_shape = tf.random_uniform(shape=[2], minval=32, maxval=48, dtype=tf.int32)

        croped_image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
        croped_image = tf.image.resize_images(croped_image, random_shape, method=2)

        croped_image = tf.random_crop(croped_image, [32, 32, 3])

        image = tf.cond(random < 0.25, lambda: image, lambda: croped_image)
        image = tf.reshape(image, [32, 32, 3])

        random = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
        cut_image = tf.py_func(Cutout(2, 8), [image], tf.float32)
        image = tf.cond(random < 0.25, lambda: image, lambda: cut_image)
        image = tf.reshape(image, [32, 32, 3])

        image = tf.reshape(image, shape=[32, 32, 3])
        label = tf.cast(feature["label"], tf.int32)
        label = tf.one_hot(label, len(self.classes))
        return image, label

    def __get_dataset(self, path, parsed, batch_size, buffer_size=10000):
        filename = [os.path.join(path, name) for name in os.listdir(path)]
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=19260817)
        dataset = dataset.repeat(count=None)
        dataset = dataset.map(parsed, num_parallel_calls=16)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
        return dataset

    def global_variables_initializer(self):
        initializer = []
        initializer.append(self.train_batch_initializer)
        initializer.append(self.test_batch_initializer)
        initializer.append(tf.global_variables_initializer())
        return initializer

    def test_batch(self, batch_size):
        dataset = self.__get_dataset(self.tf_test_path, self.__parsed, batch_size)
        return dataset

    def train_batch(self, batch_size):
        dataset = self.__get_dataset(self.tf_train_path, self.__parsed_distorted, batch_size)
        return dataset


class SEBlock(object):
    def __init__(self, se_rate=16, l2_rate=0.001):
        self.se_rate = se_rate
        self.l2_rate = l2_rate

    def __call__(self, inputs):
        shape = inputs.shape
        squeeze = GlobalAveragePooling2D()(inputs)
        squeeze = Dense(shape[-1] // self.se_rate, activation="relu")(squeeze)
        extract = Dense(shape[-1], activation=tf.nn.sigmoid)(squeeze)
        extract = tf.expand_dims(extract, axis=1)
        extract = tf.expand_dims(extract, axis=1)
        output = tf.keras.layers.multiply([extract, inputs])
        return output


class Convolution2D(object):
    def __init__(self, filters, ksize, strides=1, padding="valid", activation=None, kernel_regularizer=None):
        self.filters = filters
        self.kernel_regularizer = kernel_regularizer
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.activation = activation

    def __call__(self, x):
        x = Conv2D(self.filters, self.ksize, self.strides, padding=self.padding,
                   kernel_regularizer=self.kernel_regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        return x


class Residual(object):
    def __init__(self, filters, kernel_size, strides=1, padding='valid', activation=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation

    def __call__(self, inputs):
        x = Convolution2D(filters=self.filters[0],
                          ksize=(1, 1),
                          strides=self.strides,
                          padding=self.padding,
                          activation=self.activation)(inputs)
        x = Convolution2D(filters=self.filters[1],
                          ksize=self.kernel_size,
                          strides=self.strides,
                          padding=self.padding,
                          activation=self.activation)(x)
        x = Convolution2D(filters=self.filters[2],
                          ksize=(1, 1),
                          strides=self.strides,
                          padding=self.padding,
                          activation=None)(x)
        x = SEBlock(se_rate=16)(x)
        if x.shape.as_list()[-1] != inputs.shape.as_list()[-1]:
            inputs = Convolution2D(filters=self.filters[2],
                                   ksize=(1, 1),
                                   strides=self.strides,
                                   padding=self.padding,
                                   activation=None)(inputs)
        x = tf.keras.layers.add([inputs, x])
        x = tf.keras.layers.Activation(self.activation)(x)
        return x


class SEResNet50(object):
    def __init__(self, se_rate=16, l2_rate=0, drop_rate=0.25):
        self.se_rate = se_rate
        self.l2_rate = l2_rate
        self.drop_rate = drop_rate

    def __call__(self, inputs):
        x = Convolution2D(32, [3, 3], [1, 1], activation="relu", padding="same")(inputs)

        x = Residual([32, 32, 128], [3, 3], [1, 1], activation="relu", padding="same")(x)
        x = Residual([32, 32, 128], [3, 3], [1, 1], activation="relu", padding="same")(x)
        x = Residual([32, 32, 128], [3, 3], [1, 1], activation="relu", padding="same")(x)
        x = MaxPooling2D([3, 3], [2, 2], padding="same")(x)

        x = Residual([64, 64, 256], [3, 3], [1, 1], activation="relu", padding="same")(x)
        x = Residual([64, 64, 256], [3, 3], [1, 1], activation="relu", padding="same")(x)
        x = Residual([64, 64, 256], [3, 3], [1, 1], activation="relu", padding="same")(x)
        x = Dropout(self.drop_rate)(x)
        x = Residual([64, 64, 256], [3, 3], [1, 1], activation="relu", padding="same")(x)
        x = MaxPooling2D([3, 3], [2, 2], padding="same")(x)

        x = Residual([128, 128, 512], [3, 3], [1, 1], activation="relu", padding="same")(x)
        x = Residual([128, 128, 512], [3, 3], [1, 1], activation="relu", padding="same")(x)
        x = Residual([128, 128, 512], [3, 3], [1, 1], activation="relu", padding="same")(x)
        x = Residual([128, 128, 512], [3, 3], [1, 1], activation="relu", padding="same")(x)
        x = Residual([128, 128, 512], [3, 3], [1, 1], activation="relu", padding="same")(x)
        x = Residual([128, 128, 512], [3, 3], [1, 1], activation="relu", padding="same")(x)
        x = Dropout(self.drop_rate)(x)
        x = MaxPooling2D([3, 3], [2, 2], padding="same")(x)

        x = Residual([256, 256, 1024], [3, 3], [1, 1], activation="relu", padding="same")(x)
        x = Residual([256, 256, 1024], [3, 3], [1, 1], activation="relu", padding="same")(x)
        x = Residual([256, 256, 1024], [3, 3], [1, 1], activation="relu", padding="same")(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.drop_rate)(x)

        x = Dense(10, activation="softmax")(x)
        print(self)
        return x


batch_size=128
data_path="dataseats/cifar10/Image"
mode_path="dataseats/cifar10/Image/TFMode"
dataset=DatasetReader(data_path,image_size=[32,32])
train_dataset=dataset.train_batch(batch_size=batch_size)
test_dataset=dataset.test_batch(batch_size=batch_size)

plateau=ReduceLROnPlateau(monitor="val_acc", #12epoch准确率没有上升，学习率减半
                                verbose=1,
                                mode='max',
                                factor=0.6,
                                patience=12)
early_stopping=EarlyStopping(monitor='val_acc',#70epoch准确率没有上升，停止训练
                                   verbose=1,
                                   mode='max',
                                   patience=70)
checkpoint=ModelCheckpoint(f'SERESNET101_CLASSIFIER_CIFAR10_DA.h5',#保存acc最高的模型
                           monitor='val_acc',
                           verbose=1,
                           mode='max',
                           save_weights_only=False,
                           save_best_only=True)
inputs=Input(shape=[32,32,3]) #定义输入shape
outputs=SEResNet50()(inputs)
model=Model(inputs,outputs)

model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['acc'])
trained_model=model.fit(
    train_dataset,
    steps_per_epoch=50000//batch_size,
    shuffle=True,
    epochs=300,
    validation_data=test_dataset,
    validation_steps=10000//batch_size,
    callbacks=[plateau,early_stopping,checkpoint]
)
