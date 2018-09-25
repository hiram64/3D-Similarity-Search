import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from lib.model import conv_autoencoder_3d
from lib.utils import load_data


def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D Convolutional AutoEncoder')
    parser.add_argument('--num_epoch', default=500, type=int, help='the number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='mini batch size')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate of optimizer')
    parser.add_argument('--data_path', default='./data/modelnet10.npz', type=str, help='path to dataset to train')
    parser.add_argument('--logdir', default='./log', type=str, help='path to directory to save log')
    parser.add_argument('--checkpoint_dir', default='./checkpoint', type=str, help='path to directory to checkpoint')
    args = parser.parse_args()

    return args


def main():
    # Prepare parameters
    args = parse_args()
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    data_path = args.data_path
    logdir = args.logdir
    checkpoint_dir = args.checkpoint_dir
    rdm = np.random.RandomState(13)

    # Prepare data
    x_train, y_train, x_test, y_test = load_data(data_path)
    x_train, y_train = shuffle(x_train, y_train)

    num_train_data = x_train.shape[0]

    input_data = tf.placeholder(tf.float32, shape=[None, 32, 32, 32], name='input')
    net_input = input_data[..., np.newaxis]

    CAE_3D = conv_autoencoder_3d(net_input, args=args, is_training=True)

    with tf.name_scope('training_summary'):
        tf.summary.scalar('train_loss', CAE_3D.loss)
    sum_op = tf.summary.merge_all()

    # Start Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir, sess.graph)
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epoch):
            print('epoch :', epoch)
            x_train = x_train[rdm.permutation(num_train_data)]

            average_loss = 0
            for i in range(0, num_train_data, batch_size):
                feed_dict = {input_data: x_train[i:i + batch_size]}
                fetch = {'optimizer': CAE_3D.optimizer,
                         'loss': CAE_3D.loss,
                         'summary': sum_op
                         }

                results = sess.run(fetches=fetch, feed_dict=feed_dict)
                average_loss += results['loss']

            print('train loss : ', average_loss / int(num_train_data / batch_size))

            # save summary and checkpoint by epoch
            writer.add_summary(summary=results['summary'], global_step=epoch)
            saver.save(sess, os.path.join(checkpoint_dir, 'model_{0}'.format(epoch)))


if __name__ == '__main__':
    main()
