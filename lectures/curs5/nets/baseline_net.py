import tensorflow as tf

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)

def cifarnet_arg_scope_bnorm(weight_decay=0.004, is_training=True):
  """Defines the batch norm cifarnet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
       An `arg_scope` to use for the cifarnet model.
  """

  batch_norm_params = {
    'is_training': is_training,
    'center': True,
    'scale': True,
    'decay': 0.997,
    'epsilon': 0.001,
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_initializer=tf.truncated_normal_initializer(stddev=5e-2),
      activation_fn=tf.nn.relu6,
      normalizer_fn=slim.batch_norm):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope(
          [slim.fully_connected],
          biases_initializer=tf.constant_initializer(0.1),
          weights_initializer=trunc_normal(0.04),
          weights_regularizer=slim.l2_regularizer(weight_decay),

          activation_fn=tf.nn.relu) as sc:
        return sc


trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)


def cifarnet_bn(images, num_classes=10, is_training=False,
                dropout_keep_prob=0.5,
                prediction_fn=slim.softmax,
                scope='CifarNet'):
  end_points = {}

  with tf.variable_scope(scope, 'CifarNet', [images]):
    net = slim.conv2d(images, 64, [5, 5], scope='conv1')
    end_points['conv1'] = net
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    end_points['pool1'] = net
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    end_points['conv2'] = net
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    end_points['pool2'] = net
    net = slim.flatten(net)
    end_points['Flatten'] = net
    net = slim.fully_connected(net, 384, scope='fc3')
    end_points['fc3'] = net
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope='dropout3')
    net = slim.fully_connected(net, 192, scope='fc4')
    end_points['fc4'] = net
    if not num_classes:
      return net, end_points
    logits = slim.fully_connected(net, num_classes,
                                  biases_initializer=tf.zeros_initializer(),
                                  weights_initializer=trunc_normal(1 / 192.0),
                                  weights_regularizer=None,
                                  activation_fn=None,
                                  scope='logits')

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points