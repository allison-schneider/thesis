import numpy as np

def filenames():
	filenames = ['data/hgt-{:03d}.nc'.format(time) for time in np.arange(0,241,3)]
	return filenames

print(filenames())