import os
import warnings
import napari
from scipy import stats, signal, cluster,spatial
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib import pyplot as plt
from skimage import (exposure, feature, filters, io, measure,
                     morphology, restoration, draw, segmentation, transform,
                     util)
import math
from pathlib import Path
from sklearn import metrics
from kneed import KneeLocator

def contains_all_trues(new,prev):
    # could change to nditer if need to be faster?
    anded = np.logical_and(new,prev)
    if np.array_equal(new,anded):
        return True
    else:
        return False


def counting(file,final_count):
    labels = measure.label(file,connectivity=1)
    # viewer.add_image(file)
    # viewer.add_labels(labels)
    lobes = measure.regionprops_table(labels,properties=('area','area_filled',))
    lobe_areas = lobes['area']
    filtered_lobes = lobe_areas[lobe_areas>80000]
    final_count = max(final_count,filtered_lobes.size)
    print(f"number of labels{np.unique(labels).size},number of filtered lobes:{filtered_lobes.size}")
    erosion = morphology.binary_erosion(file,footprint=np.ones(shape=(3,3,3)))
    return erosion,final_count
def fill(x):
    newfile = np.zeros_like(x)
    for i in range(len(x)):
        roi = x[i,:,:]
        new = morphology.remove_small_holes(roi, area_threshold=4000)
        newfile[i,:,:] = new
    newfile = morphology.remove_small_holes(newfile,area_threshold=6000)
    return newfile
def erode(n,file,depth,ids,napariDisplay,prev_comp,prev_ids,total_comps):
    eroded_file = morphology.binary_erosion(file,footprint=np.ones(shape=(2,8,8)))
    total_comps.append(eroded_file)
    labels_eroded,labelcount_eroded = measure.label(eroded_file,connectivity=1,return_num=True)
    labels_og, labelcount_og = measure.label(file,connectivity=1,return_num=True)
    if napariDisplay:
        viewer.add_image(eroded_file,visible=False)
    current_comp=[]
    current_ids=[]
    prev_exists = [False] * len(prev_comp)
    print(f"IDS AT DEPTH: {depth} are: {ids}")
    if depth==n-1:
        print(f"depth: {depth}")
        # no more even if theres nothing left
        # print(f"hit erosion cap of {n}")
        for i in range(len(prev_ids)): # loop through every previous component
            letter = 96
            for j in range(1,labelcount_eroded+1): # loop through current eroded elements
                new = np.where(labels_eroded==j,1,0)
                new = new.astype(bool)
                if contains_all_trues(new,prev_comp[i]): # check if an eroded element is in the previous element
                    current_comp.append(new)
                    letter+=1
                    newid = prev_ids[i]+chr(letter)
                    current_ids.append(newid)
                    prev_exists[i]=True
        # all current ids need to be appended regardless of whether they disappeared or not
        # print(ids)
        # print("SHOULD HAVE APPENDED SOMETHING TO IDS")
        ids.extend(prev_ids) # is this right?????????????
        print(ids)
        # END
    elif labelcount_eroded==0:
        print(f"there's nothing left, depth at {depth}, cap at {n}")
        # print(ids)
        # print("SHOULD HAVE APPENDED SOMETHING TO IDS")
        print(f"prev ids: {prev_ids}")
        ids.extend(prev_ids)
        print(ids)
        # END
    else: # there is Stuff left
        print(f"depth: {depth}")
        for i in range(len(prev_ids)): # loop through every previous component
            letter = 96
            for j in range(1,labelcount_eroded+1): # loop through current eroded elements
                new = np.where(labels_eroded==j,1,0)
                new = new.astype(bool)
                if contains_all_trues(new,prev_comp[i]): # check if an eroded element is in the previous element
                    current_comp.append(new)
                    letter+=1
                    newid = prev_ids[i]+chr(letter)
                    current_ids.append(newid)
                    prev_exists[i]=True
        # if there are prev comps left that "dont exist" that means they eroded into nothing
        for i in range(len(prev_exists)):
            if prev_exists[i]==False: # it disappeared
                # print(ids)
                # print("SHOULD HAVE APPENDED SOMETHING TO IDS")
                ids.append(prev_ids[i])
                print(ids)
        print(f"prev ids:{prev_ids}")
        erode(n,eroded_file,depth+1,ids,napariDisplay,current_comp,current_ids,total_comps)
        # CONTINUE ERODING
    
def close(x):
    newfile = np.zeros_like(x)
    for i in range(len(x)):
        roi = x[i,:,:]
        new = morphology.binary_closing(roi,footprint=np.ones(shape=(7,7))) # optional adjust kernel shape, footprint=np.ones(shape=(5,5,5)) doesn't work
        newfile[i,:,:] = new
    return newfile        
def count_comp(file,depth,id,ids,napariDisplay): # linkage = n-1, 4
    # ID IS ID OF PARENT, TUPLE OF (DEPTH, CODE)
    eroded_file = morphology.binary_erosion(file,footprint=np.ones(shape=(2,8,8))) # change footprint to correspond to resolution?
    labels,labelcount = measure.label(eroded_file,connectivity=1,return_num=True)
    # print(f"depth:{depth}, id: {id}")
    if labelcount>0:
        for i in range(1,labelcount+1):
            new = np.where(labels==i,labels,0)
            newcode = id[1] + chr(i+96)
            count_comp(new,depth+1,(depth+1,newcode),ids,napariDisplay)
            if napariDisplay:
                viewer.add_image(new,name=f"{depth} level blob",visible=False)
    else: # labelcount==0:
        # print("count f adding??????")
        ids.append(id)
        # viewer.add_image(file,name=f"{depth} level blob (final depth)",visible=False)
        # return count + 1


def longest_common_prefix(s1, s2):
    """Finds the longest common prefix between two strings."""
    i = 0
    while i < len(s1) and i < len(s2) and s1[i]  == s2[i]:
        i += 1
    return s1[:i]

def string_prefix_distance(s1, s2, max_len):
    """
    Calculates a 'distance' between two strings based on their longest common prefix.
    A smaller distance means a longer common prefix.
    """
    lcp_len = len(longest_common_prefix(s1, s2))
    # The distance is the difference from the maximum possible length
    # This ensures that longer common prefixes result in smaller distances.
    return max_len - lcp_len

def pdist_string_wrapper(u, v):
        # u and v will be 1-element arrays containing the string
        return string_prefix_distance(u[0], v[0], max_string_length)

accuracies = []


for ii in range(14,15):
    directory_path = Path('/Users/coding/Downloads/_nuclear_morpho_data/Lamin/Preprocessing_3')
    dir = [f.path for f in os.scandir(directory_path) if f.is_file()]
    all_files = []
    for path in dir:
        all_files = np.append(all_files,str(path))

    all_files = sorted(all_files)

    file = io.imread(all_files[0])
    viewer = napari.view_image(file, visible=False)

    lobe_counts = []
    adj_counts = []
    filenames = []
    # real_counts = [2,2,4,4,4,4,2,1,3,4,2,1,4,1,3,4,3,3,3,3,3,3,2,3,2,2,3,4,2,4,3,1,3,2,3,3,2,1,2,3,2,1,3,3,3]
    real_counts = [2,4,4,3]


    if len(real_counts)!=len(all_files):
        warnings.warn(f"real_counts len not equal to all_files, real_counts: {len(real_counts)},all_files: {len(all_files)}")


    max_depth = ii
    print(f"max depth:{ii}")
    for i in range(len(all_files)):
        
        fileindex = str(all_files[i][-6]+all_files[i][-5])
        filenames.append(fileindex)
        ogfile = io.imread(all_files[i])
        print(all_files[i])
        hole_fill = fill(ogfile)

        # REMOVE CROPPING ARTIFACTS
        arr = morphology.remove_small_objects(hole_fill,min_size=100000,connectivity=1) 
        viewer.add_image(arr,visible=False,name=f"{fileindex}")
        # viewer = napari.view_image(arr, name=f"{fileindex}", visible=False)
        final_blobs = np.zeros_like(hole_fill)
        count = -1

        ids_recursive = []
        ids_uniform = []
        view_steps = True
        total_count = count_comp(arr,0,(0,"a"),ids_recursive,False)
        
        # start depth at 1 for uniform erosion
        total_comps = []
        erode(max_depth,arr,1,ids_uniform,view_steps,[arr],["a"],total_comps)
        # for image in reversed(total_comps):
        #     viewer.add_image(image,visible=False)
        
        # viewer.add_image(arr,visible=False,name=f"{fileindex}")
        # print(f"{ids}")
        finalcodes_uniform = ids_uniform
        finalcodes_recursive = [id[1] for id in ids_recursive]
        # print(finalcodes)
        if set(finalcodes_uniform)!=set(finalcodes_recursive):
            warnings.warn("they are NOT THE SAME!!!!!!! \n\n\n\n")
        max_string_length = max(len(s) for s in finalcodes_uniform)

        string_array = np.array(finalcodes_uniform, dtype=object).reshape(-1, 1)

        distance_matrix = spatial.distance.pdist(string_array, metric=pdist_string_wrapper)
        try:
            Z = cluster.hierarchy.linkage(distance_matrix, method='average')

            # print(f"linkage matrix: {Z}")

            fig = plt.figure()
            d = cluster.hierarchy.dendrogram(Z, labels=finalcodes_uniform, color_threshold=0.8*max(Z[:,2]),leaf_rotation=45,leaf_font_size=5) # color_threshold can be adjusted
            # d = cluster.hierarchy.dendrogram(Z, labels=finalcodes,color_threshold='default') # color_threshold can be adjusted        
            colors = d['leaves_color_list']
            # print(colors)
            colors = [str(color) for color in colors]
            plt.xlabel("codes")

            plt.ylabel(f"distance: {max_string_length}")

            if len(np.unique(colors))==1:
                if len(colors) > 10:
                    # print("too many C0")
                    lobe_counts.append(1)
                else:
                    lobe_counts.append(len(colors))
            else:
                count = 0
                if colors.count('C0')>5:
                    # print("too many C0")
                    lobe_counts.append(1)
                else:
                    count += colors.count('C0')
                    count += len(np.unique(list(filter(lambda x: x!='C0',colors))))
                    lobe_counts.append(count)
            
            # print(f"finalcodes: {finalcodes}")
            # lens = [len(code) for code in finalcodes]
            # print(f"lengths: {lens}")
            # total = np.empty(shape=(4,len(finalcodes)),dtype=object)
            # total[0,:] = np.array(lens)
            # total[1,:] = np.array(finalcodes,dtype=object)
            # orderedcolors = np.empty_like(colors)
            # for i in range(len(lens)):
            #     orderedcolors[i] = colors[d['leaves'][i]]
            # print(f"leaves: {d['leaves']}")
            # print(f"ordered colors: {orderedcolors}")
            # total[2,:] = np.array(d['leaves'])
            # total[3,:] = orderedcolors
            # print(f"total: {total}")
            # total = total[:,np.argsort(total[0,:])]
            # print(total)
            # kneedle = KneeLocator(total[0,:].astype(float),range(len(lens)),curve="concave",direction="increasing")
            # print(f"elbow point: {kneedle.knee}")
            # if kneedle.knee is None:
            #     print("no knee")
            #     filteredcolors = total[3,:]
            # else:
            #     filteredcolors = total[3,total[0]>kneedle.knee]
            # print(filteredcolors)
            # if len(np.unique(filteredcolors))==1:
            #     if len(filteredcolors) > 10:
            #         print("too many C0")
            #         lobe_counts.append(1)
            #     else:
            #         lobe_counts.append(len(filteredcolors))
            # else:
            #     count = 0
            #     if np.count_nonzero(filteredcolors == 'C0')>5:
            #         print("too many C0")
            #         lobe_counts.append(1)
            #     else:
            #         count += np.count_nonzero(filteredcolors == 'C0')
            #         count += len(np.unique(list(filter(lambda x: x!='C0',filteredcolors))))
            #         lobe_counts.append(count)
            plt.title(f"real count: {real_counts[i]}, lobe_count: {lobe_counts[-1]}, file: {fileindex}")
        except ValueError:
            # print(f"sa ys that no pdist matrix, real_count is {real_counts[i]}")
            # print("not real ???")
            lobe_counts.append(1)
        # plt.close()

    plt.show()


    # print(lobe_counts)
    # print(real_counts)
    # # # print(adj_counts)
    # print(len(lobe_counts))
    # if len(lobe_counts)==len(real_counts):
    #     print(metrics.accuracy_score(real_counts,lobe_counts))
    print(lobe_counts)
    print(real_counts)
    # # print(adj_counts)
    print(len(lobe_counts))
    if len(lobe_counts)==len(real_counts):
        # print(metrics.accuracy_score(real_counts,lobe_counts))
        accuracies.append(metrics.accuracy_score(real_counts,lobe_counts))
        print(metrics.accuracy_score(real_counts,lobe_counts))

print(accuracies)

plt.bar(range(10,11),accuracies)
plt.show()