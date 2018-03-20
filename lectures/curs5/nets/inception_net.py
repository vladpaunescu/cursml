import tensorflow as tf

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)


# compute number of poolings
# 32 / 4 = 8 -> 2 poolings

def inception_block_1(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'InceptionBlock_1', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 16, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 16, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 16, 3, scope='Conv2d_0b_3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv2d(net, 16, 1, scope='Conv2d_0a_1x1')
      tower_conv2_1 = slim.conv2d(tower_conv2_0, 24, 3, scope='Conv2d_0b_3x3')
      tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_valuee list of `Tensor` arguments that are passed to the op function. to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net


def inception_block_2(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'InceptionBlock_2', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
      tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_valuee list of `Tensor` arguments that are passed to the op function. to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net


def reduction_block_1(net, atrous=False):
  with tf.variable_scope('Reduction_1'):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 64, 3,
                               stride=2,
                               scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 48, 3,
                                  scope='Conv2d_0b_3x3')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 64, 3,
                                  stride=2,
                                  scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
      tower_pool = slim.max_pool2d(net, 3, stride=2,
                                   scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)

  return net


def reduction_block_2(net, atrous=False):
  with tf.variable_scope('Reduction_2'):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 128, 3,
                               stride=2,
                               scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 96, 3,
                                  scope='Conv2d_0b_3x3')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, 3,
                                  stride=2,
                                  scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
      tower_pool = slim.max_pool2d(net, 3, stride=2,
                                   scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)

  return net


def inception_resnet_slim_base(inputs,
                               final_endpoint='reduction3',
                               output_stride=4,
                               align_feature_maps=True,
                               scope=None,
                               activation_fn=tf.nn.relu):
  end_points = {}

  padding = 'SAME' if align_feature_maps else 'VALID'

  def add_and_check_final(name, net):
    end_points[name] = net
    return name == final_endpoint

  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):

      # 16
      net = slim.conv2d(inputs, 32, 3, stride=2, padding=padding, scope='conv1_1')
      net = slim.conv2d(net, 32, 3, padding=padding, scope='conv1_2')
      net = slim.conv2d(net, 32, 3, padding=padding, scope='conv1_3')

           # net_1 = slim.conv2d(net, 32, 3, padding=padding, scope='conv1_2', activation_fn=None)
      # net = net * 0.2 + net_1
      # net = activation_fn(net)
      #
      # net_1 = slim.conv2d(net, 32, 3, padding=padding, scope='conv1_3',  activation_fn=None)
      # net = net * 0.2 + net_1
      # net = activation_fn(net)

      if add_and_check_final('conv1_3', net): return net, end_points

      net = slim.repeat(net, 10, inception_block_1, scale=0.25,
                        activation_fn=activation_fn)

      # # reduction
      # net = slim.conv2d(net, 64, 3, scope='conv1_4')
      #
      # # 8
      # net = slim.max_pool2d(net, 3, stride=2, padding=padding,
      #                       scope='maxpool1')

      # / 8
      net = reduction_block_1(net)

      net = slim.conv2d(net, 80, 1, scope='conv2_1')
      net = slim.conv2d(net, 96, 3, scope='conv2_2')
      net = slim.conv2d(net, 96, 3, scope='conv2_3')

      # net_1 = slim.conv2d(net, 64, 3, scope='conv2_1', activation_fn=None)
      # net = net * 0.2 + net_1
      # net = activation_fn(net)
      #
      # net_1 = slim.conv2d(net, 64, 3, scope='conv2_2', activation_fn=None)
      # net = net * 0.2 + net_1
      # net = activation_fn(net)
      #
      # net_1 = slim.conv2d(net, 64, 3, scope='conv3_2', activation_fn=None)
      # net = net * 0.2 + net_1
      # net = activation_fn(net)

      if add_and_check_final('conv2_3', net): return net, end_points

      net = slim.repeat(net, 10, inception_block_2, scale=0.25,
                        activation_fn=activation_fn)
      if add_and_check_final('block2', net): return net, end_points

      # 4
      net = reduction_block_2(net)

      if add_and_check_final('reduction3', net): return net, end_points


def inception_resnet_slim(images, num_classes=10,
                          is_training=False,
                          dropout_keep_prob=0.8,
                          prediction_fn=tf.nn.softmax,
                          activation_fn=tf.nn.relu,
                          scope='InceptionResnet'):
  with tf.variable_scope(scope, 'InceptionResnet', [images]):
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_resnet_slim_base(images,
                                                   final_endpoint='reduction3',
                                                   output_stride=4,
                                                   align_feature_maps=True,
                                                   scope=scope,
                                                   activation_fn=activation_fn)

      kernel_size = net.get_shape()[1:3]
      net = slim.avg_pool2d(net, kernel_size, padding='VALID', scope="AvgPool")
      net = slim.flatten(net)
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='Dropout1')

      # end_points['Flatten'] = net
      #
      # net = slim.fully_connected(net,
      #                               384,
      #                               activation_fn=tf.nn.relu,
      #                               scope='fc1')
      #
      # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
      #                    scope='Dropout2')
      #

      logits = slim.fully_connected(net,
                                    num_classes,
                                    activation_fn=None,
                                    scope='Logits')
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, name='Predictions')

  return logits, end_points


def inception_resnet_arg_scope(weight_decay=0.00004,
                               batch_norm_decay=0.997,
                               batch_norm_epsilon=0.001,
                               activation_fn=tf.nn.relu):
  """Returns the scope with the default parameters for inception_resnet_v2.

  Args:
    weight_decay: the weight decay for weights variables.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.
    activation_fn: Activation function for conv2d.

  Returns:
    a arg_scope with the parameters needed for inception_resnet_v2.
  """
  # Set weight_decay for weights in conv2d and fully_connected layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):
    batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'fused': None,  # Use fused batch norm if possible.
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=activation_fn,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as scope:
      return scope
