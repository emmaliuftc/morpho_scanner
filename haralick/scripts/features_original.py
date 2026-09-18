import os
import napari
from scipy import stats, signal
# from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib import pyplot as plt, colors, cm
# from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure,
                     morphology, restoration, segmentation, transform,
                     util)
from pathlib import Path
import math
# import pywt
from sklearn import svm,preprocessing,cluster,decomposition,metrics,manifold
import mahotas
# import seaborn as sns
# import umap


def fill(x):
    newfile = np.zeros_like(x)
    for i in range(len(x)):
        roi = x[i,:,:]
        new = morphology.remove_small_holes(roi, area_threshold=4000)
        newfile[i,:,:] = new
    newfile = morphology.remove_small_holes(newfile,area_threshold=100000)
    return newfile


directory_path = Path('/Users/coding/Downloads/_nuclear_morpho_data/Lamin/Preprocessing_3')
dir = [f.path for f in os.scandir(directory_path) if f.is_file()]
all_files = []
for path in dir:
    all_files = np.append(all_files,str(path))

all_files = sorted(all_files)

file = io.imread(all_files[0])

viewer = napari.view_image(file,visible=False)

directory2_path = Path('/Users/coding/Downloads/_nuclear_morpho_data/Lamin/Preprocessing')
dir2 = [f.path for f in os.scandir(directory2_path) if f.is_file()]
all_files2 = []
for path in dir2:
    all_files2 = np.append(all_files2,str(path))

all_files2 = sorted(all_files2)

# print(all_files)

# shape = (number of images,number of features)
data_size = len(all_files)
feature_num = 17
x_train = np.empty(shape=(data_size,feature_num))

# images = np.array((1,1))
volumes = []
surfaces = []
sphericities = []
haralicks = []
zernikes = []
filenames = []

for i in range(len(all_files)):
    fileindex = str(all_files[i][-6]+all_files[i][-5])
    filenames.append(fileindex)
    ogfile = io.imread(all_files[i])

    print(ogfile.shape)
    print(all_files[i])
    hole_fill = fill(ogfile)
    viewer.add_image(hole_fill,name=f"{all_files[i]}",visible=False)
    # images = np.append(images,hole_fill)

    labels = measure.label(hole_fill)
    # properties = measure.regionprops_table(labels,properties=("area","axis_major_length","axis_minor_length"))
    volume = np.count_nonzero(hole_fill)
    # maj = properties["axis_major_length"]
    # mino = properties["axis_minor_length"]
    # print(f"volume: {volume}")
    volumes.append(volume)

    slice = hole_fill[10,:,:]



    haralicks.append(mahotas.features.haralick(hole_fill,return_mean=True))
    # print(f"binary iamge: {features.haralick(hole_fill,return_mean=True)}")
    
    # viewer.add_image(features.haralick(hole_fill),name=f"haralick {all_files[i]}",visible=False)
    # file = io.imread(all_files2[i])
    # print(all_files2[i])
    # haralick = features.haralick(file,return_mean=True)
    # haralicks.append(haralick)
    # print(f"binary iamge: {haralicks[i]}")

    verts,faces,normals,values = measure.marching_cubes(hole_fill,step_size=3)

    surface_area = measure.mesh_surface_area(verts,faces)

    # print(f"surface area: {surface_area}")
    surfaces.append(surface_area)

    sphericity = pow(math.pi, 0.3333)*pow(6*volume,0.666666)/surface_area

    # print(sphericity)
    sphericities.append(sphericity)

    # com = mahotas.center_of_mass(hole_fill)
    # zernike = np.empty(shape=(1,25))
    # zernike[0,:] = mahotas.features.zernike_moments(hole_fill[round(com[0]),:,:],200,8).flatten() # 1D array of length 25 
    # # zernike[1,:] = mahotas.features.zernike_moments(hole_fill[round(com[0]-10),:,:],200,8).flatten() # 1D array of length 25 
    # # zernike[2,:] = mahotas.features.zernike_moments(hole_fill[round(com[0]+10),:,:],200,8).flatten() # 1D array of length 25 
    # zernike = zernike.flatten()
    # print(zernike.shape)
    # zernikes.append(zernike)

'''
    for i in range(10):
        gauss = filters.gaussian(hole_fill,sigma = i)
        viewer.add_image(gauss,name = f"sigma = {i}")
        boundary = segmentation.find_boundaries(gauss)
        viewer.add_image(boundary)
'''
    # wp = pywt.WaveletPacketND(hole_fill,wavelet='db2')
    # # x=np.array([1,2,3,4,5,6,7,8])
    # # wp = pywt.WaveletPacketND(data=x,wavelet='db1')
    # # print(wp)
    # # print(np.count_nonzero(wp))

    # # print(wp.maxlevel)
    # # print(wp['aaa'].data)

    # cA, cD = pywt.dwt(hole_fill,'db2')
    # print(cA)
    # print(cD)
    # viewer.add_image(cA)
    # viewer.add_image(cD)

# real_counts = np.array([2,3,4,2,4,2,1,3,3,3])
real_counts = [2,2,4,4,4,4,2,1,3,4,2,1,4,1,3,4,3,3,3,3,3,3,2,3,2,2,3,4,2,4,3,1,3,2,3,3,2,1,2,3,2,1,3,3,3]

npvolumes = np.array(volumes)
npsurfaces = np.array(surfaces)
npsphericities = np.array(sphericities)
npharalicks = np.array(haralicks)
# npzernikes = np.array(zernikes)

print(npharalicks.shape)
print(npvolumes.shape)

# # x_train[:,0] = np.reshape(volumes,newshape=(data_size,1))[:,0]
# # x_train[:,0] = np.reshape(surfaces,newshape=(data_size,1))[:,0]
# x_train[:,0] = np.reshape(sphericities,newshape=(data_size,1))[:,0]
# x_train[:,1] = np.reshape(real_counts,newshape=(data_size,1))[:,0]
# x_train[:,2:15] = npharalicks

# WITH ZERNIKE

# x_train[:,0] = np.reshape(volumes,newshape=(data_size,1))[:,0]
# x_train[:,1] = np.reshape(surfaces,newshape=(data_size,1))[:,0]
# x_train[:,2] = np.reshape(sphericities,newshape=(data_size,1))[:,0]
# x_train[:,3] = np.reshape(real_counts,newshape=(data_size,1))[:,0]
# x_train[:,4:17] = npharalicks
# x_train[:,18:] = npzernikes

# NO ZERNIKE

x_train[:,0] = np.reshape(volumes,newshape=(data_size,1))[:,0]
x_train[:,1] = np.reshape(surfaces,newshape=(data_size,1))[:,0]
x_train[:,2] = np.reshape(sphericities,newshape=(data_size,1))[:,0]
x_train[:,3] = np.reshape(real_counts,newshape=(data_size,1))[:,0]
x_train[:,4:17] = npharalicks


# print(x_train)

x_train_norm = preprocessing.normalize(x_train)

scaler = preprocessing.StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
# print(x_train)
# print(x_train_scaled)

# need x_test eventually
# scaler.transform(x_test)

# pca = decomposition.PCA(n_components=2)
pca = decomposition.PCA(n_components=2,whiten=True)
x_pca = pca.fit_transform(x_train_scaled)

# print(x_pca)

components = pca.components_
print(components)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(components[0],components[1])
for i in range(feature_num):
    ax.annotate(i, (components[0][i],components[1][i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
s = [ax.quiver(0,0,components[0][i], components[1][i],angles='xy', scale_units='xy',scale=1) for i in range(feature_num)]

explained_variances = pca.explained_variance_ratio_

print(f"explained variance ratio: {explained_variances}")
print(f"cumulative explained variance: {pca.explained_variance_ratio_.sum()}")

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.bar(range(1,len(explained_variances)+1),explained_variances)
ax1.set_ylim(None,1)
ax1.set_xlabel("principal components")
ax1.set_ylabel("% explained variance")




# kmeans = cluster.KMeans(n_clusters=6,random_state=0)
kmeans = cluster.KMeans(n_clusters=6,random_state=0,init="random",n_init=40)
kmeans.fit(x_pca)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

#fig2, (ax7,ax2,ax3,ax4,ax5,ax6) = plt.subplots(1,6)
fig2, (ax7,ax2,ax5,ax6) = plt.subplots(1,4)

ax2.scatter(x_pca[:, 0], x_pca[:, 1], c=labels, cmap='viridis')
ax2.scatter(centers[:, 0], centers[:, 1], color='red', marker='x')
ax2.set_xlabel('pc1')
ax2.set_ylabel('pc2')
ax2.set_title("K-means after PCA with random init n=40")
for i, (x, y_val) in enumerate(x_pca):
    ax2.annotate(filenames[i], (x, y_val), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

ax6.scatter(x_pca[:, 0], x_pca[:, 1],c=real_counts,cmap='viridis')
ax6.set_xlabel("pc1")
ax6.set_ylabel("pc2")
ax6.set_title("manual lobe count")



kmeans1 = cluster.KMeans(n_clusters=6,random_state=0)
kmeans1.fit(x_pca)
labels1 = kmeans1.labels_
centers1 = kmeans1.cluster_centers_
ax7.scatter(x_pca[:, 0], x_pca[:, 1], c=labels1, cmap='viridis')
ax7.scatter(centers1[:, 0], centers1[:, 1], color='red', marker='x')
ax7.set_xlabel('pc1')
ax7.set_ylabel('pc2')
ax7.set_title("K-means after PCA kmeans++ init n=1")
for i, (x, y_val) in enumerate(x_pca):
    ax7.annotate(filenames[i], (x, y_val), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)


pearsons = np.empty(shape=(feature_num,feature_num))
for i in range(feature_num):
    for j in range(feature_num):
        if i>j:
            pearsons[i,j]=0
        else:
            pearsons[i,j]=round(stats.pearsonr(x_train[:,i],x_train[:,j]).statistic,3)

cmap = cm.PiYG
norm = colors.Normalize(vmin=-1,vmax=1)
colors = cmap(norm(pearsons))

cell_text = [[str(item) for item in row] for row in pearsons]

# colors = [['g' if (cell > 0.8) else ('r' if (cell<-0.8) else 'w') for cell in row] for row in pearsons]


col_labels = ['vol',"sa","spher","lbcnt","asm","con","cor","sos:v","idm","sav","sva","sen","en","dva","den","imcI","imcII"]
# row_labels = ['Row A', 'Row B', 'Row C']

fig4 = plt.figure()
ax100 = fig4.add_subplot()

ax100.axis('off')
cells = np.random.randint(-1, 2, (10, 10))
img = plt.imshow(cells,cmap=cmap)
plt.colorbar()
img.set_visible(False)
# table = ax100.table(cellText=cell_text, cellColours=colors,cellLoc='center', loc='center')

table = ax100.table(cellText=cell_text, cellColours=colors,cellLoc='center', loc='center',colLabels=col_labels,rowLabels=col_labels)


table.auto_set_font_size(False)
table.set_fontsize(6)


hdbscan = cluster.HDBSCAN(min_cluster_size=2)
hdbscan.fit(x_pca)

labels2 = hdbscan.labels_
# centers2 = hdbscan.centroids_
print(labels2)

ax5.scatter(x_pca[:, 0], x_pca[:, 1], c=labels2, cmap='viridis')
# ax5.scatter(centers2[:, 0], centers2[:, 1], color='red', marker='x')
ax5.set_xlabel('pc1')
ax5.set_ylabel('pc2')
ax5.set_title('HDBSCAN after PCA')
for i, (x, y_val) in enumerate(x_pca):
    ax5.annotate(filenames[i], (x, y_val), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)



# tsne = manifold.TSNE(2,random_state=0)
# tsne_result = tsne.fit_transform(x_train_scaled)
# print(tsne_result.shape)

# print(f"tsne effective learnign rate: {tsne.learning_rate_}")

# ax3.scatter(tsne_result[:, 0], tsne_result[:, 1])
# ax3.set_xlabel('tsne1')
# ax3.set_ylabel('tsne2')
# ax3.set_title("t-SNE embedding")
# for i, (x, y_val) in enumerate(tsne_result):
#     ax3.annotate(filenames[i], (x, y_val), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

# umap = umap.UMAP(random_state=0)
# umap_result = umap.fit_transform(x_train_scaled)

# print(umap_result.shape)
# ax4.scatter(umap_result[:, 0], umap_result[:, 1])
# ax4.set_xlabel('umap1')
# ax4.set_ylabel('umap2')
# ax4.set_title("UMAP embedding")
# for i, (x, y_val) in enumerate(umap_result):
#     ax4.annotate(filenames[i], (x, y_val), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)



plt.show()

range_n_clusters = [3, 4, 5, 6,7,8,9]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(x_pca) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=0,init="random",n_init=40)
    cluster_labels = clusterer.fit_predict(x_pca)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = metrics.silhouette_score(x_pca, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = metrics.silhouette_samples(x_pca, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = plt.cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        x_pca[:, 0], x_pca[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()