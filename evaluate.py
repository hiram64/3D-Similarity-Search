import argparse

import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

from lib.model import conv_autoencoder_3d
from lib.utils import load_data
from lib.utils import calculate_average_precision
from lib.visualize import visualize, visualize_3d_iodata, visualize_tsne


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate 3D Convolutional AutoEncoder')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate of optimizer')
    parser.add_argument('--data_path', default='./data/modelnet10.npz', type=str, help='path to dataset to evaluate')
    parser.add_argument('--use_exist_modelout', default=False, type=bool,
                        help='whether to use existing model output in npz. If False, model evaluation will be done again')
    parser.add_argument('--modelout_save', default=False, type=bool, help='whether to save model output')
    parser.add_argument('--checkpoint_dir', default='./checkpoint', type=str, help='path to directory to checkpoint')
    parser.add_argument('--modeleval_out_dir', default='./output.npz', type=str,
                        help='path to directory to save and load model evaluation output')
    parser.add_argument('--num_search_sample', default=1, type=int, help='the number of samples to search')
    parser.add_argument('--num_top_similarity', default=4, type=int, help='search top k similarity data')
    args = parser.parse_args()

    return args


def similarity_search(encoded, k):
    """
    caluculate similarity between encoded data for each row.
    run similar data search based on cosine similarity and return similar data index and similarity
    """
    mat = cosine_similarity(encoded)

    # top k index
    idx = np.argsort(mat)[:, ::-1]

    sims = np.array([mat[i][d] for i, d in enumerate(idx)])

    # return top 2 ~ k + 1 index and similarity since top 1 will be the self-data.
    return idx[:, 1:k + 1], sims[:, 1:k+1]


def main():
    # Prepare parameters
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    data_path = args.data_path
    num_top_similarity = args.num_top_similarity
    num_search_sample = args.num_search_sample
    modelout_save = args.modelout_save
    use_exist_modelout = args.use_exist_modelout
    modeleval_out_dir = args.modeleval_out_dir

    # Prepare Data
    _, _, x_test, y_test = load_data(data_path=data_path)

    input_data = tf.placeholder(tf.float32, shape=[None, 32, 32, 32], name='input')
    net_input = input_data[:, :, :, :, np.newaxis]

    CAE_3D = conv_autoencoder_3d(net_input, args=args, is_training=False)

    if use_exist_modelout:
        data = np.load(modeleval_out_dir)
        idx = data['idx']
        sims = data['sims']
        encoded = data['encoded']
        decoded = data['decoded']
    else:
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            feed_dict = {input_data: x_test}

            # extract encoded features and vectorize them
            encoded = CAE_3D.encoded.eval(session=sess, feed_dict=feed_dict)
            nd, k1, k2, k3, k4 = encoded.shape
            encoded = np.reshape(encoded, (nd, k1 * k2 * k3 * k4))

            decoded = CAE_3D.decoded.eval(session=sess, feed_dict=feed_dict)

            idx, sims = similarity_search(encoded, num_top_similarity)

            if modelout_save:
                np.savez_compressed(modeleval_out_dir, idx=idx, sims=sims, encoded=encoded, decoded=decoded)

    # visualize encoded data with t-SNE
    # visualize_tsne(encoded, y_test)

    # add self-index as the first column
    self_idx = np.arange(encoded.shape[0]).reshape((encoded.shape[0], 1))
    idx = np.concatenate([self_idx, idx], axis=1)

    # select samples to visualize randomly
    sample_idx = np.random.randint(0, x_test.shape[0], num_search_sample)

    # visualize similar search result
    visualize(x_test, y_test, idx[sample_idx])

    # visualize input and its decoded data
    # visualize_3d_iodata(x_test[sample_idx], decoded[sample_idx], y_test[sample_idx])

    # calculate average precision
    ap = calculate_average_precision(y_test, idx[sample_idx], sims[sample_idx], num_search_sample)
    print('Average Precision per sample : ', ap)


if __name__ == '__main__':
    main()
