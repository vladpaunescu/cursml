import numpy as np
import tensorflow as tf

import dataset_driver
from nets import baseline_net
from nets import inception_net

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)

tf.logging.set_verbosity(tf.logging.INFO)


# evaluate model
def evaluate(session, input_tensors, output_tensors, dataset, config):

  total_examples = dataset.get_test_num_examples()
  iters = int(total_examples / config.batch_size)
  # tf.logging.info("Total examples {}".format(total_examples))
  # tf.logging.info("Total iters {}".format(iters))

  acc = []
  losses = []

  image_input = input_tensors['image_input']
  label_input = input_tensors['label_input']
  is_training = input_tensors['is_training']

  accuracy = output_tensors['accuracy']
  loss = output_tensors['loss']

  for i in range(iters):
    x, y = dataset.get_batch(i, batch_size=config.batch_size)

    feed_dict = {

      image_input: x,
      label_input: y,
      is_training: False

    }

    testAcc, testLoss = session.run([accuracy, loss], feed_dict=feed_dict)
    acc.append(testAcc)
    losses.append(testLoss)
    # if i % 100 == 0:
    #  tf.logging.info("Test " + str(i) + ": accuracy:" + str(testAcc) + " loss: " + str(testLoss))

  #   tf.logging.info(acc)
  meanAcc = np.mean(np.asarray(acc))
  meanLoss = np.mean(np.asarray(losses))

  #   tf.logging.info("Test Accuracy {:.2f} %".format(meanAcc * 100))

  return meanAcc, meanLoss



class Config(object):
  batch_size = 32
  height = 32
  width = 32
  channels = 3
  num_classes = 10
  initial_learning_rate = 0.001
  experiment="InceptionSlim"


def get_baseline_net(inputs, is_training):
  arg_scope = baseline_net.cifarnet_arg_scope_bnorm(is_training=is_training)
  with slim.arg_scope(arg_scope):
    # logits, end_points = model.cifarnet_bn(image_input, is_training=is_training)
    logits, end_points = baseline_net.cifarnet_bn(inputs, is_training=is_training)

  return logits, end_points

def get_inception_net(inputs, is_training):
  arg_scope = inception_net.inception_resnet_arg_scope()
  with slim.arg_scope(arg_scope):
    # logits, end_points = model.cifarnet_bn(image_input, is_training=is_training)
    logits, end_points = inception_net.inception_resnet_slim(inputs, is_training=is_training)

  return logits, end_points



def train(dataset, config):

  trainingAccuracyList = []
  trainingLossList = []
  testAccuracyList = []
  testLossList = []
  learningRateList = []

  num_steps = 50000
  num_examples = dataset.get_train_num_examples()
  iters = num_examples / config.batch_size
  learning_rate_step = 15000
  learning_rate_decay = 0.1

  tf.reset_default_graph()
  # g = tf.Graph().as_default()

  image_input, label_input, is_training, learning_rate = dataset_driver.add_placeholders(config.height, config.width)
  preprocessed_input = dataset_driver.add_preprocessing(image_input, is_training)

  # logits, end_points = get_baseline_net(preprocessed_input, is_training)
  logits, end_points = get_inception_net(preprocessed_input, is_training)

  loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_input, logits=logits))

  # accuracy of the trained model, between 0 (worst) and 1 (best)
  predictions = end_points['Predictions']

  correct_prediction = tf.equal(tf.argmax(predictions, 1), label_input)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Optimizer.
  # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  # for batch norm training. Note: we should use slim.train_op

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    # Ensures that we execute the update_ops before performing the train_step
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

  running_lr = config.initial_learning_rate

  init = tf.global_variables_initializer()
  sess = tf.Session(graph=tf.get_default_graph())
  # actually initialize our variables
  sess.run(init)

  tf.logging.info("Starting optimization")
  tf.logging.info("Initial LR {}. LR stepdown itnerval {}."
                     " LR deacy factor {}".format(running_lr, learning_rate_step,
                                                                            learning_rate_decay))

  input_tensors = {
    'image_input': image_input,
    'label_input': label_input,
    'is_training': is_training,
    'learning_rate': learning_rate
  }

  eval_out_tensors = {
    'accuracy': accuracy,
    'loss': loss
  }

  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('./logs/train/{}'.format(config.experiment),
                                       sess.graph)

  for i in range(num_steps):

    x, y = dataset.random_batch(batch_size=config.batch_size)

    feed_dict = {

      image_input: x,
      label_input: y,
      is_training: True,
      learning_rate: running_lr

    }

    if i % 250 == 0:
      _, trainAcc, trainLoss = sess.run([optimizer, accuracy, loss], feed_dict=feed_dict)

      testAcc, testLoss = evaluate(session=sess,
                                   input_tensors=input_tensors,
                                   output_tensors=eval_out_tensors,
                                   dataset=dataset,
                                   config=config)

      tf.logging.info("---> Train " + str(i) + ": accuracy:" + str(trainAcc) + " loss: " + str(trainLoss))
      tf.logging.info("Test " + str(i) + ": accuracy:" + str(testAcc) + " loss: " + str(testLoss))

      trainingAccuracyList.append(trainAcc)
      trainingLossList.append(trainLoss)
      testAccuracyList.append(testAcc)
      testLossList.append(testLoss)
      learningRateList.append(running_lr)

    else:
      sess.run([optimizer], feed_dict=feed_dict)

    if i > 0 and i % learning_rate_step == 0:
      tf.logging.info("Learning reate step down. Old {}. New {}".format(running_lr, running_lr * learning_rate_decay))
      running_lr = running_lr * learning_rate_decay

def main():

  dataset = dataset_driver.Dataset()
  config = Config()

  train(dataset, config)


if __name__ == "__main__":
  main()
