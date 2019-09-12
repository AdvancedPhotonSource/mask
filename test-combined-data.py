import masks
import numpy as np
import tifffile as tf
import os


ABSORBTION = np.linspace(0.6, 1, 40, endpoint=False) # a 0.2 means 80% absorbed <- good bad ->
INTENSITY = np.linspace(1, 40, 40)  # i
# ABSORBTION = [0.6, 0.7, 0.8, 0.9] # a 0.2 means 80% absorbed <- good bad ->
# INTENSITY = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # i
MASKDIST = 0.5 # d
NUMSCANPTS = 20 # p

# print (ABSORBTION)
# print (INTENSITY)

# Define 2D mask
mask = tf.imread('tests/core/mask-256.tiff')
maskl = tf.imread('tests/core/mask-256.tiff')
maskl = np.squeeze(maskl)
shape = maskl.shape
maskl = np.pad(maskl, ((0,NUMSCANPTS),(0,NUMSCANPTS)), 'constant', constant_values=0.0)

# Define mask's grid coordinates in 3D
mx, my, mz = mask.shape
gridx = np.linspace(-0.5, 0.5, mx+1, dtype='float32')
gridy = np.linspace(-0.5, 0.5, my+1, dtype='float32')
gridz = np.linspace(-0.5 / mx, 0.5 / my, mz+1, dtype='float32')

# Define random orientation vectors
sx = np.load('tests/core/sx.npy')
sy = np.load('tests/core/sy.npy')
dx = np.load('tests/core/dx.npy')
dy = np.load('tests/core/dy.npy')

# Generate measurements
# add pad at the end of mask in x, y dims, so the true_patches are of the same size

true_patch = []
for k in range(sx.size):
	print (k)

	# Define source pixellation
	srcgridx = np.linspace(sx[k], sx[k]+(4 / mx), NUMSCANPTS, dtype='float32')
	srcgridy = np.linspace(sy[k], sy[k]+(4 / mx), NUMSCANPTS, dtype='float32')

	# Define detector pixellation
	detgridx = np.linspace(dx[k], dx[k]+(4 / mx), NUMSCANPTS, dtype='float32') 
	detgridy = np.linspace(dy[k], dy[k]+(4 / mx), NUMSCANPTS, dtype='float32')

	# Calculate projections
	prj = masks.project(mask, gridx, gridy, gridz, detgridx, detgridy, srcgridx, srcgridy, dsrc=MASKDIST, ddet=0)
	#dxchange.write_tiff(prj, fname='tests/data/noisefree-projs/image-'+str(k), overwrite=True)

	# Validation data
	bitsize = mx / 4
	dxi = int(np.ceil((dx[k] + 0.5) * shape[0]))
	dyi = int(np.ceil((dy[k] + 0.5) * shape[1]))
#	print ('dxi,dxi+NUMSCANPTS, dyi, dyi+NUMSCANPTS, patch shape',dxi,dxi+NUMSCANPTS, dyi, dyi+NUMSCANPTS, maskl[dxi:dxi+NUMSCANPTS, dyi:dyi+NUMSCANPTS].shape)
	tf.imsave('tests/data/true-patches/image-'+str(k), maskl[dxi:dxi+NUMSCANPTS, dyi:dyi+NUMSCANPTS])
	true_patch.append(maskl[dxi:dxi+NUMSCANPTS, dyi:dyi+NUMSCANPTS])
	for m in range(len(INTENSITY)):
		for n in range(len(ABSORBTION)):
			data = INTENSITY[m] * np.exp(prj / prj.max() * np.log(ABSORBTION[n]))
			data = (np.random.poisson(data) / INTENSITY[m]).astype('float32')
			normalized_dir = 'tests/data/normalized-projs-i'+str(m)+'-a'+str(n)
			if not os.path.exists(normalized_dir):
				os.makedirs(normalized_dir)
			tf.imsave(normalized_dir+'/image-'+str(k), data)

all_data = []
for m in range(len(INTENSITY)):
	for n in range(len(ABSORBTION)):
		for k in range(sx.size):
			# Read data
			data = tf.imread('tests/data/normalized-projs-i'+str(m)+'-a'+str(n)+'/image-'+str(k)+'.tiff')
			data = np.squeeze(data)
			all_data.append(data)

all_data = np.stack(all_data, axis = 2)
print ('all data shape', all_data.shape)

# r_frames = []
# for kk in range(sx.size):
# 	grain_f = np.take(frames, range(len(INTENSITY) * len(ABSORBTION)), axis=-1)
# 	print (grain_f.shape)
# 	r_frames.append(grain_f)
# frames = r_frames[0]
# for ai in range(1, len(r_frames)):
# 	frames = np.append(frames, r_frames[ai], axis=-1)
# frames = np.stack(r_frames,axis=-1)

tf.imsave('tests/data/all/all_frames.tiff', all_data)

true_patch = np.stack(true_patch, axis=-1)
print('true patches shape', true_patch.shape)
tf.imsave('tests/data/true-patches/all_patches.tiff', true_patch)

