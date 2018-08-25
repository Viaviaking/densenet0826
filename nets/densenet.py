"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def densenet(images, num_classes=200, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:

            
    
            #convolution 224x224x3
            end_point = "convolution_7x7"
            logits = slim.conv2d(images, 2*growth, [7,7],stride = 2, padding = "same",scope=end_point)
            end_points[end_point] = logits
            #pooling 112x112x2g
            end_point = "max_pool_3x3"
            logits = slim.max_pool2d(logits,[3,3],stride = 2,padding = "same",scope = end_point)
            end_points[end_point] = logits

            #block1 56x56x2g
            end_point = "block1"
            logits = block(logits, 6, growth,scope=end_point)
            end_points[end_point] = logits

            #transition layer1 56x56x(2g+g*6)=56x56x8g
            end_point = "tran_conv1"
            logits = bn_act_conv_drp(logits, reduce_dim(logits), [1,1],scope=end_point)
            end_points[end_point] = logits
            #56x56x4g
            end_point = "tran_pooling1"
            logits = slim.avg_pool2d(logits,[2,2],stride = 2,scope = end_point)
            end_points[end_point] = logits

            #block2 28x28x4g
            end_point = "block2"
            logits = block(logits, 12, growth,scope=end_point)
            end_points[end_point] = logits

            #transition layer2 28x28x(4g+24*g)=28x28x28g
            end_point = "tran_conv2"
            logits = bn_act_conv_drp(logits, reduce_dim(logits), [1,1],scope=end_point)
            end_points[end_point] = logits
            #28x28x14g
            end_point = "tran_pooling2"
            logits = slim.avg_pool2d(logits,[2,2],stride = 2,scope = end_point)
            end_points[end_point] = logits

            #block3 14x14x14g
            end_point = "block3"
            logits = block(logits, 24, growth,scope=end_point)
            end_points[end_point] = logits

            #transition layer3 14x14x(14g+24*g)=14x14x38g
            end_point = "tran_conv3"
            logits = bn_act_conv_drp(logits, reduce_dim(logits), [1,1],scope=end_point)
            end_points[end_point] = logits
            #14x14x19g
            end_point = "tran_pooling3"
            logits = slim.avg_pool2d(logits,[2,2],stride = 2,scope = end_point)
            end_points[end_point] = logits

            #block4 7x7x19g
            end_point = "block4"
            logits = block(logits, 16, growth,scope=end_point)
            end_points[end_point] = logits

            #classification layer 7x7x(19g+16*g)=7x7x35g
            #global average pooling
            end_point = "class_conv"
            logits = slim.avg_pool2d(logits, logits.shape[1:3],scope=end_point)
            end_points[end_point] = logits
            #1x1x35g=1x1x840
            end_point = "flatten"
            logits = slim.flatten(logits, scope=end_point)
            end_points[end_point] = logits
            end_point = "fully_connected"
            logits = slim.fully_connected(logits, num_classes, activation_fn=None, scope=end_point)
            end_points[end_point] = logits

          
  

    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
