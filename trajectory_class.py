import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
import scipy.interpolate
from mpl_toolkits.basemap import Basemap
import sys

class Atmosphere:
	""" Contains atmospheric parameters (u speed, v speed, geopotential height)
	for two time layers at any given time.
	"""
	def __init__(self,
				 time=0): 	# Current time in hours from start
		self.time = time
		self.time_between_files = 3		# Time in hours	between predictions	
		# Generate list of all filenames and index of current file
		self.filenames = ['data/hgt-{:03d}.nc'.format(time) 
							for time in np.arange(0,241,3)]
		self.file_index = np.int(np.floor(self.time / self.time_between_files))					
		# Initialize values for interpolate function
		self.points = self.build_points()
		self.u_values = self.values("u")
		self.v_values = self.values("v")
		self.gh_values = self.values("gh")
		
	def build_points(self):
		""" Returns the first argument for the interpn function,
		a tuple of 3 ndarrays of float, named points,
		of shape ((nlats,), (nlons,), (ntimes)),
		which stays constant throughout the trajectory."""
		file = netcdf.netcdf_file(self.filenames[0], mmap=False)
		vars = file.variables
		file.close()

		latitudes = vars['lat'][:]
		longitudes = vars['lon'][:]
		times = np.array([0, self.time_between_files])
		points = latitudes, longitudes, times
		return points

	def values(self, parameter):
		""" Returns the second argument for the interpn function,
		a 3D ndarray of float, named values, of shape (nlats, nlons, ntimes),
		which is updated when the trajectory passes a new time layer.
		Arguments:
		parameter -- 'u', 'v', or 'gh'; the atmospheric parameter """
		
		# Open first file and retrieve gridded values
		file0 = netcdf.netcdf_file(self.filenames[file_index], mmap=False)
		vars0 = file0.variables
		file0.close()
		t0_values = vars0[parameter][0][0]
		t0_values = t0_values[:, :, np.newaxis]

		# Open second file and retrieve gridded values
		file1 = netcdf.netcdf_file(self.filenames[file_index + 1], mmap=False)
		vars1 = file1.variables
		file1.close()
		t1_values = vars1[parameter][0][0]
		t1_values = t1_values[:, :, np.newaxis]

		# Append values for both times along third dimension
		t_values = np.append(t0_values, t1_values, axis=2)
		return t_values

class Parcel:
	def __init__(self, 
				 latitude=41,	# Latitude in degrees (-90, 90) 
				 longitude=-71, # Longitude in degrees (0, 360]
				 time=0):		# Time in hours

		self.lat = latitude
		self.lon = longitude
		self.time = time

	#def speed_grid(self)


p = Parcel([41, 42, 43], [-71, -72, -73], 0)
atmo = Atmosphere()
#print(p.time)
print(atmo.u_values)