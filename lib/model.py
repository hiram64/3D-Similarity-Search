import collections

import tensorflow as tf


def conv_autoencoder_3d(input_data=None, args=None, is_training=True, reuse=False):
    """3D Convolutional Autoencoder network definition"""
    initialize_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    with tf.variable_scope('autoencoder_3D', reuse=reuse):
        with tf.variable_scope('encoder'):
            with tf.variable_scope('conv1'):
                net = tf.layers.conv3d(input_data, 16, 2, 1, padding='same', kernel_initializer=initialize_kernel,
                                       name='conv')
                net = tf.layers.batch_normalization(net, training=is_training, name='batch_norm')
                net = tf.nn.relu(net, name='relu')
                net = tf.layers.max_pooling3d(net, 2, 2, padding='same', name='maxpooling')

            with tf.variable_scope('conv2'):
                net = tf.layers.conv3d(net, 32, 2, 1, padding='same', kernel_initializer=initialize_kernel,
                                       name='conv')
                net = tf.layers.batch_normalization(net, training=is_training, name='batch_norm')
                net = tf.nn.relu(net, name='relu')
                net = tf.layers.max_pooling3d(net, 2, 2, padding='same', name='maxpooling')

            with tf.variable_scope('conv3'):
                net = tf.layers.conv3d(net, 64, 2, 1, padding='same', kernel_initializer=initialize_kernel,
                                       name='conv')
                net = tf.layers.batch_normalization(net, training=is_training, name='batch_norm')
                net = tf.nn.relu(net, name='relu')
                net = tf.layers.max_pooling3d(net, 2, 2, padding='same', name='maxpooling')

            encoded = net

        with tf.variable_scope('decoder'):
            with tf.variable_scope('deconv1'):
                net = tf.layers.conv3d_transpose(net, 64, 2, 2, padding='same', use_bias=False,
                                                 kernel_initializer=initialize_kernel,
                                                 name='deconv')
                net = tf.nn.bias_add(net, tf.Variable(tf.random_normal([64], mean=0.0, stddev=0.02)))
                net = tf.layers.batch_normalization(net, training=is_training, name='batch_norm')
                net = tf.nn.relu(net, name='relu')

            with tf.variable_scope('deconv2'):
                net = tf.layers.conv3d_transpose(net, 32, 2, 2, padding='same', use_bias=False,
                                                 kernel_initializer=initialize_kernel,
                                                 name='deconv')
                net = tf.nn.bias_add(net, tf.Variable(tf.random_normal([32], mean=0.0, stddev=0.02)))
                net = tf.layers.batch_normalization(net, training=is_training, name='batch_norm')
                net = tf.nn.relu(net, name='relu')

            with tf.variable_scope('deconv3'):
                net = tf.layers.conv3d_transpose(net, 16, 2, 2, padding='same', use_bias=False,
                                                 kernel_initializer=initialize_kernel,
                                                 name='deconv')
                net = tf.nn.bias_add(net, tf.Variable(tf.random_normal([16], mean=0.0, stddev=0.02)))
                net = tf.layers.batch_normalization(net, training=is_training, name='batch_norm')
                net = tf.nn.relu(net, name='relu')

            with tf.variable_scope('deconv4'):
                net = tf.layers.conv3d_transpose(net, 1, 1, 1, padding='same', use_bias=False,
                                                 kernel_initializer=initialize_kernel,
                                                 name='deconv')
                net = tf.nn.bias_add(net, tf.Variable(tf.random_normal([1], mean=0.0, stddev=0.02)))
                decoded = tf.sigmoid(net, name='sigmoid')

        # Loss function and optimizer
        with tf.variable_scope('loss_function'):
            # loss = tf.reduce_mean(tf.square(input_data - decoded))
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=input_data, logits=net))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
                with tf.variable_scope('optimizer'):
                    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

        CAE_3D = collections.namedtuple('CAE_3D', ('decoded', 'loss', 'optimizer', 'encoded'))

        return CAE_3D(
            decoded=decoded,
            loss=loss,
            optimizer=optimizer,
            encoded=encoded
        )
