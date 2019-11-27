"""
Finds the principal components of a population of reference geometries and the latent variables between the population
of supine geometries and stiffness values that are all from a prone-to-supine bpm pipeline. Training data consisted of
nodal positions of the torso. The components were found firstly from the full population followed by populations where
each individual was left out. Meshes are then reconstructed from each set of components and exported for each individual
, stiffness in range and number of principal components up to a maximum.

Inputs: A prone-to-supine bpm project and the maximum number of principal components to be used in exporting meshes.
Outputs: To a workspace called "new.data/shape_analysis"
Author: Alexander Catalinac 2017/18
"""


import morphic
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy
import json



def valid_subrmd(sid_rmd, sid_succes):
    if sid_rmd in sid_succes:
        print('Subject %d is valid to remove' % sid_rmd)
    else:
        raise ValueError('Subject %d doesn\'t have a valid mesh!' % sid_rmd)

def leave_mesh_out(data, rm, sid_succes):
    msk = np.zeros(sid_succes.shape[1], dtype=np.int)
    for i in range(sid_succes.shape[1]):
        if sid_succes[0, i] == rm:
            msk[i] = 0
        else:
            msk[i] = sid_succes[0, i]
    data_rmd = data[msk != 0, :]

    return data_rmd

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

def update_mesh(msh, nodes):
    ref_nd_ids = msh.get_node_ids()[1]
    k = list(enumerate(ref_nd_ids))
    k1 = np.array(k)
    refj = np.zeros(len(k))
    nd_id = np.zeros(len(k))
    for i in range(len(k)):         # Preparation to replace nodes
        refj[i] = k1[i, 0]          # Row with the enumerates of nd_id
        nd_id[i] = k1[i, 1]            # Row with node id's
    refj = refj.astype(int)
    nd_id = nd_id.astype(int)
    for i in range(len(k)):         # Actually replace nodes
        msh.nodes[nd_id[i]].set_values(nodes[refj[i], :])
    msh.generate()
    return msh




def export_mesh(msh, out_pth, fle_nme, vsl=False):
    """Export mesh to a path.

    :param msh: A morphic mesh.
    :param out_pth: A string containing the path to export to.
    :param fle_nme: A string containing the filename.
    :param vsl: A boolean for output of nodes and elements.
    :return:
    """
    msh.save(os.path.join(out_pth, '{0}.mesh'.format(fle_nme)))
    if vsl:
        stp = automesh.Params({'results_dir': '{0}'.format(out_pth)})
        mechanics.export_OpenCMISS_mesh(stp, mesh=msh, field_export_name=fle_nme)
    return


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

    a=1

    # ==============

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

def error(new, ref):
    ED_vec = np.zeros(len(new))
    zerosvec = np.zeros(len(new))
    for i in range(len(new)):
        ed = np.sqrt((new[i, 0] - ref[i, 0]) ** 2 + (new[i, 1] - ref[i, 1]) ** 2 + (new[i, 2] - ref[i, 2]) ** 2)
        ED_vec[i] = ed        # Vector with Euclidean distance for each node
    AED_error = np.mean(ED_vec)
    AED_sd = np.std(ED_vec)
    mse = mean_squared_error(zerosvec, ED_vec)
    rmse = np.sqrt(mse)

    return AED_error, AED_sd, rmse

#  --- Calculating the distance between the nodes of the pca mesh and the reference mesh
def pca_error(new, ref):
    dist_vec = np.zeros(len(new))
    for i in range(len(new)):
        d = np.sqrt((new[i, 0] - ref[i, 0]) ** 2 + (new[i, 1] - ref[i, 1]) ** 2 + (new[i, 2] - ref[i, 2]) ** 2)
        dist_vec[i] = d
    # plt.figure()
    # plt.plot(dist_vec)
    # plt.title('Distance between nodes of pca reconstructed mesh and original mesh', fontsize=20)
    return dist_vec

def pca_rmse(new, ref):
    mse = mean_squared_error(ref, new)
    rmse = np.sqrt(mse)
    return rmse

def error_boxplot(error_vec, xlabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.boxplot(error_vec, patch_artist=True)       # Create the boxplot
    for box in bp['boxes']:
        box.set(color='#7570b3', linewidth=2)           # change outline color
        box.set(facecolor='#1b9e77')                    # change fill color
    for whisker in bp['whiskers']:                      ## change color and linewidth of the whiskers
        whisker.set(color='#7570b3', linewidth=2)
    for cap in bp['caps']:                              ## change color and linewidth of the caps
        cap.set(color='#7570b3', linewidth=2)
    for median in bp['medians']:                        ## change color and linewidth of the medians
        median.set(color='#b2df8a', linewidth=2)
    for flier in bp['fliers']:                          ## change the style of fliers and their fill
        flier.set(marker='o', color='r', alpha=0.5)
    ax.set_xticklabels(xlabels, fontsize=20)  ## Custom x-axis labels
    # ax.set_xticklabels(['Reference not included', 'Reference included'], fontsize=20)      ## Custom x-axis labels
    ax.get_xaxis().tick_bottom()                        ## Remove top axes and right axes ticks
    ax.get_yaxis().tick_left()

    return


def generate_points_on_face_pointID(mesh, face, value, element_ids=[], num_points=4, dim=3):
    """
    Generate a grid of points within each element

    Keyword arguments:
    mesh -- mesh to evaluate points in
    face -- face to evaluate points on at the specified xi value
    dim -- the number of xi directions
    """

    if dim == 3:
        if face == "xi1":
            xi1 = [value]
            xi2 = scipy.linspace(0., 1., num_points)
            xi3 = scipy.linspace(0., 1., num_points)
        elif face == "xi2":
            xi1 = scipy.linspace(0., 1., num_points)
            xi2 = [value]
            xi3 = scipy.linspace(0., 1., num_points)
        elif face == "xi3":
            xi1 = scipy.linspace(0., 1., num_points)
            xi2 = scipy.linspace(0., 1., num_points)
            xi3 = [value]
        X, Y, Z = scipy.meshgrid(xi1, xi2, xi3)
        xi = scipy.array([
            X.reshape((X.size)),
            Y.reshape((Y.size)),
            Z.reshape((Z.size))]).T
    elif dim == 2:
        xi1 = scipy.linspace(0., 1., num_points)
        xi2 = scipy.linspace(0., 1., num_points)
        X, Y = scipy.meshgrid(xi1, xi2)
        xi = scipy.array([
            X.reshape((X.size)),
            Y.reshape((Y.size))]).T

    if not element_ids:
        # if elem group is empty evaluate all elements
        element_ids = mesh.elements.ids

    num_Xe = len(element_ids)
    total_num_points = num_Xe * num_points ** 2
    points = scipy.zeros((num_Xe, num_points ** 2, 3))

    pointID = np.zeros(total_num_points)

    for idx, element_id in enumerate(element_ids):
        element = mesh.elements[element_id]
        points[idx, :, :] = element.evaluate(xi)
        ndID = np.array(element.node_ids)
        pointID[idx] = ndID
        print(pointID)
    points = scipy.reshape(points, (total_num_points, 3))

    return points, pointID


if __name__ == "__main__":
    main()