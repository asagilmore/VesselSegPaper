import numpy as np
cimport numpy as cnp
from scipy import ndimage as ndi
from cython cimport boundscheck, wraparound, parallel, nogil, gil
cnp.import_array()


def get_branch_list(skeleton, voxel_spacing):

    neighbor_count = get_num_neighbors(skeleton)
    skeleton = skeleton.astype(np.uint8, copy=True)


    bifurcation = neighbor_count > 2
    skeleton[bifurcation] = 0

    struct = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_branches, num_branches = ndi.label(skeleton, structure=struct)
    branch_list = []
    cdef cnp.ndarray[cnp.float32_t, ndim=3] distance_lut = get_distance_lut(voxel_spacing)

    for i in range(1, num_branches + 1):
        length = get_vessel_length(labeled_branches, distance_lut, i)
        if length > 0:
            branch_list.append(length)
    return branch_list


def get_num_neighbors(skeleton):
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    return ndi.convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)

cdef cnp.ndarray[cnp.float32_t, ndim=3] get_distance_lut(tuple voxel_spacing):
    cdef float xspc = voxel_spacing[0]
    cdef float yspc = voxel_spacing[1]
    cdef float zspc = voxel_spacing[2]

    cdef cnp.ndarray[cnp.float32_t, ndim=3] distance_lut = np.zeros((2, 2, 2), dtype=np.float32)
    for x in range(2):
        for y in range(2):
            for z in range(2):
                distance_lut[x, y, z] = np.sqrt((x*xspc) ** 2 + (y*yspc) ** 2 + (z*zspc) ** 2)
    return distance_lut


@boundscheck(False)
@wraparound(False)
cdef float get_vessel_length(cnp.ndarray[int, ndim=3] array, cnp.ndarray[cnp.float32_t, ndim=3] distance_lut, int label):
    cdef:
        int i, j, k, x, y, z, index
        int nx = array.shape[0]
        int ny = array.shape[1]
        int nz = array.shape[2]
        cnp.ndarray[long, ndim=2] indices = np.transpose(np.nonzero(array))
        cnp.ndarray[cnp.uint8_t, ndim=3] used = np.zeros((array.shape[0], array.shape[1], array.shape[2]), dtype=np.uint8)
        float length = 0

    for index in range(indices.shape[0]):
        x, y, z = indices[index, 0], indices[index, 1], indices[index, 2]
        for i in range(x-1, x+2):
            if i < 0 or i >= nx:
                continue
            for j in range(y-1, y+2):
                if j < 0 or j >= ny:
                    continue
                for k in range(z-1, z+2):
                    if k < 0 or k >= nz:
                        continue
                    if used[i, j, k]:
                        continue
                    if array[i, j, k] == label:
                        length += distance_lut[abs(i-x), abs(j-y), abs(k-z)]
                        used[i, j, k] = 1
    return length