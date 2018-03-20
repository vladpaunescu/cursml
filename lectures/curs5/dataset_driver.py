import numpy as np
import tensorflow as tf
from enum import Enum

import cifar10

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)


class DatasetSplit(Enum):
  TRAIN = 1
  TEST = 2


class Dataset(object):

  def __init__(self):
    self.get_dataset()

    # class labels
    tf.logging.info(self.cls_train.shape)
    tf.logging.info(self.cls_test.shape)

    tf.logging.info("Train count {}".format(self.images_train.shape[0]))
    tf.logging.info("Test count {}".format(self.images_test.shape[0]))

    cls_ids = np.unique(self.cls_train)
    tf.logging.info("Class labels {}.".format(cls_ids))

    n_classes = len(cls_ids)
    tf.logging.info("Num classes {}".format(n_classes))

  def get_dataset(self):

    cifar10.maybe_download_and_extract()

    self.class_names = cifar10.load_class_names()
    self.images_train, self.cls_train, self.labels_train = cifar10.load_training_data()
    self.images_test, self.cls_test, self.labels_test = cifar10.load_test_data()

    tf.logging.info(self.images_train.shape)
    tf.logging.info(self.images_test.shape)

    # one hot encodings
    tf.logging.info(self.labels_train.shape)
    tf.logging.info(self.labels_test.shape)

  def random_batch(self, batch_size=32, split=DatasetSplit.TRAIN):

    img, labels =  self.images_train, self.cls_train
    if split == DatasetSplit.TEST:
      img, labels = self.images_train, self.cls_test

    # Number of images in the training-set.
    num_images = len(img)
    #     print(num_images)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = img[idx, :, :, :]
    y_batch = labels[idx]

    return x_batch, y_batch

  def get_batch(self, step, batch_size=32, split=DatasetSplit.TEST):

    img, labels = self.images_test, self.cls_test
    if split == DatasetSplit.TRAIN:
      img, labels = self.images_train, self.cls_train

    offset = (step * batch_size) % (labels.shape[0] - batch_size)
    #   print(offset)
    batch_imgs = img[offset:(offset + batch_size), :, :, :]
    batch_labels = labels[offset:(offset + batch_size)]

    return batch_imgs, batch_labels

  def get_train_num_examples(self):
    return self.images_train.shape[0]

  def get_test_num_examples(self):
    return self.images_test.shape[0]


def add_placeholders(height, width):

  # image batch input
  image_input = tf.placeholder(
    tf.float32, [None, height, width, 3],
    name='image_input'
  )

  label_input = tf.placeholder(
    tf.int64, [None],
    name='label_input'
  )

  is_training = tf.placeholder(tf.bool, name='is_training')

  learning_rate = tf.placeholder(tf.float32, shape=[])

  return image_input, label_input, is_training, learning_rate


# improve 1 - add data augmentation
def add_preprocessing(image_input, is_training):

  def _process_image(augment_level, image):
    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    if augment_level > 0:
      image = tf.image.random_brightness(image, max_delta=10)
      image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    if augment_level > 1:
      image = tf.image.random_saturation(image, lower=0.5, upper=1.6)
      image = tf.image.random_hue(image, max_delta=0.15)
    image = tf.minimum(image, 255.0)
    image = tf.maximum(image, 0)
    return image

  def _preprocess_train(input_tensor):
    input_tensor = tf.image.random_flip_left_right(input_tensor)
    input_tensor = input_tensor * 255.0
    input_tensor = _process_image(1, input_tensor)
    input_tensor = tf.scalar_mul(1.0 / 255, input_tensor)

    input_tensor = tf.subtract(input_tensor, 0.5)
    input_tensor = tf.multiply(input_tensor, 2.0)
    # input_tensor = tf.Print(input_tensor, [], "there")

    return input_tensor

  def _preprocess_test(input_tensor):
    # input_tensor = tf.scalar_mul(1.0 / 255, input_tensor)
    input_tensor = tf.subtract(input_tensor, 0.5)
    input_tensor = tf.multiply(input_tensor, 2.0)
    # input_tensor = tf.Print(input_tensor, [], "here")

    return input_tensor


  preprocessed_input = tf.map_fn(lambda img:
                                 tf.cond(
                                   tf.equal(
                                        is_training,
                                     tf.constant(True)),
                                    lambda: _preprocess_train(img),
                                    lambda: _preprocess_test(img)), image_input)

  return preprocessed_input