"""
Finds the principal components of a population of reference geometries
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
    pca = PCA(x.shape[0])
    nde_means = np.mean(x, axis=0)
    pca.fit(x - nde_means)
    pca.mean_ = nde_means
    pca.components_ = (pca.components_.T * np.sqrt(pca.explained_variance_)).T
    return pca


def calculate_weights(pca, input_nds):
    """Calculate weights for a prone-to-supine process and stiffness index.

    :param p: A prone-to-supine process from a bpm pipeline.
    :param pca: A class containing principal components, explained variance etc.
    :param s_idx: An index for a stiffness solved on a prone-to-supine bpm pipeline.
    :return: A one-dimensional array of component weights in number of standard deviations.
    """
    # pth = os.path.join(p.workspace('mechanics').path(), 'reference_parameter_set_{0}.mesh'.format(s_idx))
    # ref_nds = morphic.Mesh(pth).get_nodes()
    ref_nds = input_nds
    wts = np.dot(np.reshape(ref_nds, [1, -1], order='F') - pca.mean_, pca.components_.T) / pca.explained_variance_
    return wts


def reconstruct_nodes(pca, pc_idx, wts):
    """Reconstruct three-dimensional node positions from PCs and weights.

    :param pca: A class containing principal components, explained variance etc.
    :param wts: A one-dimensional array of component weights in number of standard deviations.
    :param pc_idx: An integer number of components to include.
    :return: A two-dimensional array of three-dimensional node positions.
    """
    wts = wts*-1
    nds = np.dot(wts[..., 0:pc_idx], pca.components_[0:pc_idx, :]) + pca.mean_
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
    variance = pca.explained_variance_ratio_ * 100
    cum_variance = np.cumsum(pca.explained_variance_ratio_) * 100

    # Make a stacked barplot
    N = num_components
    ind = np.arange(N)
    barwidth = 0.5
    plt.figure()
    plt.bar(ind, cum_variance[0 : N], width=barwidth, color='C0', label='cumulative variance')
    plt.bar(ind, variance[0 : N], width=barwidth, color='C1', label='variance')
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


# Function to determine the number of components to keep, based on the amount of variance explained by each component
def num_components(pca, var_percentage):
    variance = pca.explained_variance_ratio_ * 100
    cum_variance = np.cumsum(variance)

    num_vec_to_keep = 0
    for index, percentage in enumerate(cum_variance):
        if percentage >= var_percentage:
            num_vec_to_keep = index + 1
            break

    return num_vec_to_keep
