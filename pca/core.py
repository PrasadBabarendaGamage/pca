"""
Finds the principal components of a population of reference geometries.
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json


def find_principal_components(x):
    """Find principal components from an array of data.

    :param x: A two-dimensional data array of flattened three-dimensional node positions.
    :return: A class containing principal components, explained variance etc.
    """

    # Number of principal components to be kept.
    pca = PCA(x.shape[0])

    # Finding average node positions from training set.
    nde_means = np.mean(x, axis=0)

    # Calculating loading scores and variation accounted for by each principal component.
    pca.fit(x - nde_means)

    # Setting average node position as average position in model.
    pca.mean_ = nde_means

    # Vectors for our principal components.
    pca.components_ = (pca.components_.T * np.sqrt(pca.explained_variance_)).T

    return pca


def calculate_weights(pca, input_nds):
    """Calculate weights for a given observation vector.

    :param pca: A class containing principal components, explained variance etc.
    :param input_nds: An index for a stiffness solved on a prone-to-supine bpm pipeline.
    """
    # Calculating weights in number of standard deviations from given PCA input.
    wts = np.dot(input_nds - pca.mean_, pca.components_.T) / pca.explained_variance_
    return wts


def reconstruct_nodes(pca, pc_idx, wts):
    """Reconstruct three-dimensional node positions from PCs and weights.

    :param pca: A class containing principal components, explained variance etc.
    :param wts: A one-dimensional array of component weights in number of standard deviations.
    :param pc_idx: An integer number of components to include.
    :return: A two-dimensional array of three-dimensional node positions.
    """

    wts = wts*-1
    nds = np.dot(wts[0:pc_idx], pca.components_[0:pc_idx, :]) + pca.mean_
    nds = np.reshape(nds, [pca.n_features_ // 3, 3], order='F')

    return nds


def replace_nodes(msh, nds):
    """Replace the nodes of a prone-to-supine process's mesh.

    :param msh: A morphic mesh.
    :param nds: A two-dimensional array of node positions.
    :return: A morphic mesh with nodes replaced.
    """

    ref_nd_ids = msh.get_node_ids()[1]
    for j, nd_id in enumerate(ref_nd_ids):
        msh.nodes[nd_id].set_values = nds[j, :]
    return msh


def plot_variance_explained(pca, num_components):
    """Produces a Scree plot to show how much variance each principal component contributes.

    :param pca: A class containing principal components, explained variance etc.
    :param num_components: Number of  principal components to keep.
    :return: Scree plot.
    """

    # Calculating variance percentage and cumulative variance of principal components.
    variance = pca.explained_variance_ratio_ * 100
    cum_variance = np.cumsum(pca.explained_variance_ratio_) * 100

    # Make a stacked barplot
    N = num_components
    ind = np.arange(1, N)
    barwidth = 0.5
    plt.figure()
    plt.bar(ind, cum_variance[0 : N-1], width=barwidth, color='C0', label='cumulative variance')
    plt.bar(ind, variance[0 : N-1], width=barwidth, color='C1', label='variance')
    #plt.axhline(y=varianceTreshold, linewidth=2, color='k', linestyle='dashed')
    # Create names on the x-axis
    # plt.xticks(y_pos, bars, fontsize=20)
    #plt.xticks([])
    plt.ylim(0.0, 100)
    #plt.title('Cumulative variance accounted for', fontsize=10)
    plt.xlabel('PCA Modes', fontsize=10)
    plt.ylabel('Variance explained [%]', fontsize=10)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    data = {}
    data['PCA modes'] = ind.tolist()
    data['cumulative variance'] = cum_variance[0 : N].tolist()
    data['variance'] = variance[0 : N].tolist()
    with open('variance_explained_plot_data.json', 'w') as outfile:
        json.dump(data, outfile)


def num_components(pca, var_percentage):
    """Generates the number of components to keep, based on the amount of variance explained by each component.

    :param pca: A class containing principal components, explained variance etc.
    :param var_percentage: Amount of variance contributed by each principal component.
    :return: Number of components to keep.
    """

    # Calculating variance contribution by principal components.
    variance = pca.explained_variance_ratio_ * 100
    cum_variance = np.cumsum(variance)

    num_vec_to_keep = 0
    for index, percentage in enumerate(cum_variance):
        if percentage >= var_percentage:
            num_vec_to_keep = index + 1
            break

    return num_vec_to_keep