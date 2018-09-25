import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.manifold import TSNE


def visualize(data, label, index):
    """visualize voxel data with matplotlib. Data and label specified index are selected to visualzie.
    These data are arranged as each row means one query voxel and its similar voexels.
    """
    fig = plt.figure()

    row, col = index.shape

    cnt = 1
    for r in range(row):
        for c in range(col):
            print(cnt)
            ax = fig.add_subplot(row, col, cnt, projection='3d')
            ax.voxels(data[index[r, c]], edgecolor='k')
            plt.title(label[index[r, c]])

            cnt += 1

    # plt.savefig('./sim_search.png')
    plt.show()


def visualize_3d_iodata(input, output, label=None):
    """visualize voxel data. Input means original data and output menas its decoded data.
    Original data is arranged in the first row and decoded data is arranged below.
    """
    fig = plt.figure()

    num_data = input.shape[0]

    for c in range(num_data):
        # draw input data
        ax = fig.add_subplot(2, num_data, c + 1, projection='3d')
        ax.voxels(input[c], edgecolor='k')
        plt.title('Original')

        output = np.where(output >= 0.9, 1, 0)

        # draw output data
        ax = fig.add_subplot(2, num_data, num_data + c + 1, projection='3d')
        ax.voxels(output[c], edgecolor='k')
        plt.title('Decoded')

    plt.show()


def visualize_tsne(data, label):
    """visualize encoded data using t-SNE"""
    category = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

    for i, cat in enumerate(category):
        label[np.where(label == cat)] = i

    result = TSNE(n_components=2, random_state=111, perplexity=30).fit_transform(data)
    plt.scatter(result[:, 0], result[:, 1], c=label)
    plt.show()
