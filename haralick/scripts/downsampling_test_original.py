import os
import time
import napari
from scipy import stats
# from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib import pyplot as plt, colors, cm
# from scipy import ndimage as ndi
from skimage import io,morphology,measure
from skimage import (exposure, feature, filters, io, measure,
                     morphology, restoration, segmentation, transform,
                     util)
from pathlib import Path
import math
# import pywt
from sklearn import preprocessing
from sklearn import svm,preprocessing,cluster,decomposition,metrics,manifold
import mahotas
# import seaborn as sns
# import umap
import warnings



def fill(x):
    newfile = np.zeros_like(x)
    for i in range(len(x)):
        roi = x[i,:,:]
        new = morphology.remove_small_holes(roi, area_threshold=4000)
        newfile[i,:,:] = new
    newfile = morphology.remove_small_holes(newfile,area_threshold=100000)
    return newfile

def mse(true,observed):
    if true.shape != observed.shape:
        raise ValueError("Two array shapes should be the same")
    square_diff = (true-observed)**2
    mse = np.mean(square_diff)
    return mse


def quantize(arr,levels):
    print(f"number of unique gray levels: {len(np.unique(arr))}")
    minimum = np.min(arr)
    maximum = np.max(arr)
    # print(f"minimum gray level: {minimum}")
    # print(f"maximum gray level: {maximum}")
    # for row in arr:
    #     for column in row:
    fig,ax = plt.subplots(ncols=3,figsize=(12,4))
    ax[0].hist(arr.flatten(),bins=2000)
    ax[0].set_xlabel("Original gray level")
    ax[0].set_ylabel("Count")
    ax[0].set_yscale("log")
    
    encoder_kmeans = preprocessing.KBinsDiscretizer(n_bins=levels,encode="ordinal",strategy="kmeans",random_state=0)
    encoder_uniform = preprocessing.KBinsDiscretizer(n_bins=levels, encode="ordinal",strategy="uniform",random_state=0)
    compressed_kmeans = encoder_kmeans.fit_transform(arr.reshape(-1,1)).reshape(arr.shape)
    compressed_uniform = encoder_uniform.fit_transform(arr.reshape(-1,1)).reshape(arr.shape)
    comp_min = np.min(compressed_kmeans)
    comp_max = np.max(compressed_kmeans)
    # print(f"minimum gray level: {comp_min}")
    # print(f"maximum gray level: {comp_max}")
    ax[1].hist(compressed_kmeans.flatten(),bins=50)
    ax[1].set_xlabel("Kmeans compressed gray levels")
    ax[1].set_ylabel("Count")
    ax[1].set_yscale("log")
    ax[2].hist(compressed_uniform.flatten(),bins=50)
    ax[2].set_xlabel("Uniform compressed gray levels")
    ax[2].set_ylabel("Count")
    ax[2].set_yscale("log")
    return compressed_kmeans,compressed_uniform


directory_path = Path('/Users/coding/Downloads/_nuclear_morpho_data/Lamin/Preprocessing_3')
dir = [f.path for f in os.scandir(directory_path) if f.is_file()]
all_files = []
for path in dir:
    all_files = np.append(all_files,str(path))

all_files = sorted(all_files)


directory2_path = Path('/Users/coding/Downloads/_nuclear_morpho_data/Lamin/Preprocessing')
dir2 = [f.path for f in os.scandir(directory2_path) if f.is_file()]
all_files2 = []
for path in dir2:
    all_files2 = np.append(all_files2,str(path))

all_files2 = sorted(all_files2)



file = io.imread(all_files[0])

viewer = napari.view_image(file,visible=False)

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

mse_kmeans = []
mse_uniform = []


if len(all_files2)!=len(all_files):
    warnings.warn(f"all files 2{len(all_files2)} not equal in length ot allfiles{len(all_files)}")

# for i in range(len(all_files2)):
#     ogfile = io.imread(all_files2[i])

#     print(ogfile.shape)

#     viewer.add_image(ogfile,name=f"{all_files2[i]}",visible=False)
#     # reduced = measure.block_reduce(ogfile,block_size=(8,24,24),func=np.median)
    
#     viewer.add_image(ogfile,visible=False)
#     # viewer.add_image(reduced.astype(int),visible=False,name="int reduced")
#     print(f"size of the original file: {ogfile.shape}")
#     # print(reduced)
#     # reduced = np.round(reduced,0)
#     # print(reduced)
#     # print(f"number of unique gray levels: {len(np.unique(reduced))}")
#     compressed_kmeans,compressed_uniform = quantize(ogfile)
#     # viewer.add_image(compressed_kmeans,visible=False,name="kmeans")
#     # viewer.add_image(compressed_uniform,visible=False,name="uniform")
#     plt.close()
#     mse_kmeans.append(mse(ogfile,compressed_kmeans))
#     mse_uniform.append(mse(ogfile,compressed_uniform))
#     # print(f"mean squared error between kmeans and og: {mse_kmeans[i]}, MSE between uniform and og: {mse_uniform[i]}")
    
#     haralick = mahotas.features.haralick(compressed_kmeans.astype(int),return_mean=True)
#     haralicks.append(haralick)

test_levels = [20,50,100,500,1000,2500,5000]
times = []
haralicks = []
plevels = []

ogfile = io.imread(all_files2[0])
print(ogfile.shape)
print(f"size of the original file: {ogfile.shape}")
viewer.add_image(ogfile)

for i in test_levels:
    compressed_kmeans,compressed_uniform = quantize(ogfile,i)
    mse_kmeans.append(mse(ogfile, compressed_kmeans))
    mse_uniform.append(mse(ogfile, compressed_uniform))
    start = time.time()
    haralick = mahotas.features.haralick(compressed_kmeans.astype(int),return_mean=True)
    viewer.add_image(compressed_uniform)
    end = time.time()
    times.append(end-start)
    haralicks.append(haralick)
    ks = stats.ks_2samp(compressed_kmeans,compressed_uniform,"greater")
    plevels.append(ks.pvalue)
    print(plevels)

quantlevel_plot = plt.figure()
ax_q = quantlevel_plot.add_subplot()
ax_q.scatter(test_levels,mse_kmeans,linestyle="-",alpha=0.7,marker="x")
ax_q.scatter(test_levels,mse_uniform,linestyle="-",alpha=0.7,marker="o")

time_plot = plt.figure()
ax_t = time_plot.add_subplot()
ax_t.scatter(test_levels,times)

mse_plot = plt.figure()
ax_mse = mse_plot.add_subplot()
temp = [mse_uniform, mse_kmeans]
# for i in range(len(mse_kmeans)):
#     temp.append(mse_uniform[i]-mse_kmeans[i])
ax_mse.violinplot(temp,[1,2],showmedians=True)
ax_mse.set_xlabel("Quantization partitioning method")
ax_mse.set_ylabel("Mean squared error")

plt.show()