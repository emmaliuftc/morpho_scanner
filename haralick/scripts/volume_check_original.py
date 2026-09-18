import napari
from scipy import stats, signal
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure,
                     morphology, restoration, segmentation, transform,
                     util)
from pathlib import Path

def fill(x):
    newfile = np.zeros_like(x)
    for i in range(len(x)):
        roi = x[i,:,:]
        new = morphology.remove_small_holes(roi, area_threshold=4000)
        newfile[i,:,:] = new
    newfile = morphology.remove_small_holes(newfile,area_threshold=6000)
    return newfile

directory_path = Path('/Users/coding/Downloads/_nuclear_morpho_data/Lamin/Test')
all_files = list(directory_path.glob('*'))

viewer = napari.view_image(io.imread(all_files[0]),visible=False)

volumes = []
testvolumes = []
for path in all_files:
    ogfile = io.imread(path)
    # print(path)
    # viewer.add_image(ogfile)
    hole_fill = fill(ogfile)
    # labels = measure.label(hole_fill)
    # data = measure.regionprops_table(labels,properties=("area"))
    volumes = np.append(volumes,np.count_nonzero(hole_fill)*0.0001931495)
    # testvolumes = np.append(testvolumes,data["area"][0]*0.0001931495)

print(volumes)
plt.hist(volumes)
plt.xlabel("microns")
plt.ylabel("count")

plt.tight_layout()
plt.show()

napari.run()