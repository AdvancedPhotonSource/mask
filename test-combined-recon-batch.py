import numpy as np
import cupy as cp
import tifffile as tf
import time
from numba import jit


ABSORBTION = np.linspace(0.6, 1, 40, endpoint=False) # a 0.2 means 80% absorbed <- good bad ->
INTENSITY = np.linspace(1, 40, 40)  # i
MASKDIST = 0.5 # d
NUMSCANPTS = 20 # px
NUMGRAINS = 100
BATCHSIZE = 1

# Define 2D mask
maskl = tf.imread('tests/core/mask-256.tiff', out='memmap').copy()
maskl = np.squeeze(maskl)
print ('mask shape', maskl.shape)

# # Invert mask
maskl[maskl==1] = -1
maskl[maskl==0] = 1

# Pad mask
npad = int(NUMSCANPTS/2)
maskl = np.pad(maskl, ((npad, npad), (npad, npad)), mode='constant').copy()
mx, my = maskl.shape
print ('mask shape after padding', maskl.shape)


def get_true_patches():
    truepatch = tf.imread('tests/data/true-patches/all_patches.tiff').copy()
    print ('trupatches shape', truepatch.shape)
    truepatch[truepatch == 1] = -1
    truepatch[truepatch == 0] = 1
    return truepatch


@jit(nopython=True)
def scale_data(data, n):
    # Scale data between -1 and 1
    data = -np.log(data)
    a = -np.log(ABSORBTION[n])
    b = 0
    data = 2 * ((data - b) / (a - b) - 0.5)
    return  data


def get_data():
    # get all_data
    all_data = tf.imread('tests/data/all/all_frames.tiff')
    print ('all data size', all_data.shape)

    # divide the data into batches, and store it in a list of tuples
    # each tuple containing the batch, absorbtion, and intensity
    absorbtion_size = len(ABSORBTION)
    intensity_batch_size = absorbtion_size * NUMGRAINS
    absorbtion_batch_size = NUMGRAINS
    data = []
    for i in range(int(all_data.shape[2]/BATCHSIZE)):
        batch = all_data[:,:,i*BATCHSIZE:(i+1)*BATCHSIZE]
        intensity_index = int(i*BATCHSIZE / intensity_batch_size)
        absorbtion_index = int((i*BATCHSIZE % intensity_batch_size)/absorbtion_batch_size)
        batch = scale_data(batch, absorbtion_index)
        data.append((cp.array(batch), intensity_index, absorbtion_index))

    return data


# this function operstes on gpu, using cupy
# the operations are done on a batch of frames rather than on frame
def register_trans(src_freq, target_image, midpoints):
    if src_freq.shape != target_image.shape:
        raise ValueError("Error: images must be same size for "
                         "register_translation")
    shape = src_freq.shape
    target_freq = cp.fft.fft2(target_image, axes=(0, 1))
    # Whole-pixel shift - Compute cross-correlation by an IFFT
    image_product = src_freq * target_freq.conj()
    cross_correlation = cp.fft.ifft2(image_product, axes=(0, 1))
    # find the indexes of max valuue along the z axis
    indexes = cp.argmax(cp.abs(cross_correlation), axis=(0, 1))
    x = (indexes / cross_correlation.shape[1]).astype(int)
    y = indexes % cross_correlation.shape[1]
    dif = cp.where(x > midpoints[0], shape[0], 0)
    x = x - dif
    dif = cp.where(y > midpoints[1], shape[1], 0)
    y = y - dif
    return x, y


# Reconstruction of patches
overall = np.zeros((len(INTENSITY), len(ABSORBTION)))
true_patches = get_true_patches()
all_data = get_data()
mask_freq = np.fft.fftn(maskl)
midpoints = [np.fix(axis_size / 2) for axis_size in mask_freq.shape]
#tile mask_freq up to the BATCHSIZE
mask_freq_batch = cp.array(np.repeat(mask_freq[:,:,np.newaxis], BATCHSIZE, axis=2))

start = time.time()
grain_no = 0
# for i in range (len(all_data)-1, len(all_data)):
# #for i in range (len(all_data)-2, len(all_data)):
#     batch = all_data[i]
for batch in all_data:
    data = batch[0]
    intensity_index = batch[1]
    absorption_index = batch[2]

    data = cp.pad(data, ((0, mx - NUMSCANPTS), (0, my - NUMSCANPTS), (0, 0)), mode='constant')

# Register with cross-correlation
    batch_shift_x, batch_shift_y = register_trans(mask_freq_batch, data, midpoints)
    batch_shift_x, batch_shift_y = cp.asnumpy(batch_shift_x), cp.asnumpy(batch_shift_y)

    for p in range(BATCHSIZE):
        sx, sy = batch_shift_x[p], batch_shift_y[p]
        patch = maskl[sx:sx + NUMSCANPTS, sy:sy + NUMSCANPTS]
        try:
            indicator = np.mean(np.abs(patch - true_patches[:,:,grain_no]))
            if (indicator > 0.125 * 25):
                overall[intensity_index, absorption_index] += 1
        except:
            print ('m,n,patch, truepatch shapes', intensity_index, absorption_index, patch.shape, true_patches[:, :, grain_no].shape)
        grain_no +=1
        grain_no = grain_no % NUMGRAINS


stop = time.time()
print ('time: ', str(stop-start))
