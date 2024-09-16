import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from .branch_extraction import get_branch_list


def extract_radius(segmentation, centerlines, voxel_spacing):
    image = segmentation
    skeleton = centerlines
    transf = ndi.distance_transform_edt(image, return_indices=False, sampling=voxel_spacing)
    radius_matrix = transf*skeleton
    return radius_matrix


def extract_skeleton(segmentation):
    image = segmentation
    skeleton = skeletonize(image, method='lee')
    return skeleton.astype(np.uint8, copy=False)


def get_num_neighbors(skeleton):
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    num_neighbors = ndi.convolve(skeleton, kernel, mode='constant', cval=0)
    return num_neighbors


def get_bifurcations_endpoints(skeleton, num_neighbors=None):
    if num_neighbors is None:
        num_neighbors = get_num_neighbors(skeleton)

    bifurcations = num_neighbors > 2
    endpoints = num_neighbors == 1

    bifurcations = bifurcations * skeleton
    endpoints = endpoints * skeleton

    total_bifurcations = bifurcations.sum()
    total_endpoints = endpoints.sum()

    return total_bifurcations, total_endpoints


def analyze_vessels(path_to_nifti):
    out = {}
    # load the nifti file
    nifit_img = nib.load(path_to_nifti)
    voxel_spacing = nifit_img.header.get_zooms()[:3]

    segmentation = nifit_img.get_fdata().astype(bool)
    skeleton = extract_skeleton(segmentation).astype(np.uint8)

    bifurcations, endpoints = get_bifurcations_endpoints(skeleton)
    out['bifurcations'], out['endpoints'] = bifurcations, endpoints

    # get the total volume
    out['total_volume'] = segmentation.sum() * np.prod(voxel_spacing)

    radius = extract_radius(segmentation, skeleton, voxel_spacing)
    radius = radius[skeleton > 0]
    out['mean_radius'] = radius.sum() / skeleton.sum()

    out['max_radius'] = radius.max()
    out['min_radius'] = radius.min()

    out['radius_list'] = radius.flatten()
    branch_list = get_branch_list(skeleton, voxel_spacing)
    branch_list = [branch for branch in branch_list]
    out['number_of_branches'] = len(branch_list)
    out['avg_branch_length'] = np.sum(branch_list) / len(branch_list)
    out['branch_lengths'] = branch_list
    total_vessel_length = 0
    for branch in branch_list:
        total_vessel_length += branch
    out['total_vessel_length'] = total_vessel_length
    out['voxel_volume'] = np.prod(voxel_spacing)

    return out


def get_branches(skeleton, num_neighbors=None):
    if num_neighbors is None:
        num_neighbors = get_num_neighbors(skeleton)

    endpoints = np.transpose(np.where(num_neighbors == 1 or num_neighbors > 2))

    skeleton = skeleton.astype(np.uint8, copy=True)

    branches = []

    for i, j, k in endpoints:
        if skeleton[i, j, k] == 0:
            continue

        branch = []
        current = (i, j, k)

        while True:
            branch.append(current)
            skeleton[current] = 0

            neighbors = np.transpose(np.where(num_neighbors[current] > 1))
            if len(neighbors) == 0:
                break
            current = neighbors[0]

        branches.append(branch)
