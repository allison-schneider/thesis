"""
Library for calculating an atmospheric trajectory.
Uses the 82 netCDF files starting with "hgt-000.nc" 
The last function, trajectory(), calculates displacement using wind velocity 
times the timestep.

trajectory(lat, lon)
Arguments:
lat -- float, latitude in degrees, from -90 to 90
lon -- float, longitude in degrees, from -180 to 180

Returns:
trajectory_lat -- numpy array of (82,) with latitudes of trajectory
trajectory_lat -- numpy array of (82,) with longitudes of trajectory
"""

# Author: Allison Schneider
# Date: 2017 July 2

# Command line arguments:
# 	1. Latitude
#	2. Longitude
# for example
# python grid_trajectory.py 42.3603088, -71.0893148

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
from mpl_toolkits.basemap import Basemap
import sys

def initial_position(lat,lon):
	""" Choose the initial position. 
	Transforms longitude from [-180,180) to [0,360).

	Arguments:
	lat -- Latitude in degrees, -90 to 90
	lon -- Longitude in degrees, -180 to 180"""
	starting_lat = lat
	starting_lon = lon % 360
	return starting_lat, starting_lon

def round_to_grid(float, spacing=0.5):
	""" Round a latitude or longitude to the closest grid value.

	Arguments:
	float -- A floating point number or numpy array
	spacing -- Distance between grid points"""
	return np.round(float / spacing) * spacing

def spherical_hypotenuse(a, b):
	""" Given the lengths of two sides of a right triangle on a sphere, 
	a and b, find the length of the hypotenuse c. 
	
	Arguments:
	a -- Length of first side of the triangle, in meters
	b -- Length of second side of the triangle, in meters 
	"""
	earth_radius = 6371e3    # meters
	c = earth_radius * np.arccos(np.cos(a / earth_radius) 
		* np.cos(b / earth_radius))
	return c

def destination(lat1, lon1, distance, bearing):
	""" Return the latitude and longitude of a destination point 
	given a starting latitude and longitude, distance, and bearing.
	
	Arguments:
	lat1 -- Starting latitude in degrees, -90 to 90
	lon1 -- Starting longitude in degrees, 0 to 360
	distance -- Distance to travel in meters
	bearing -- Direction between 0 and 360 degrees, clockwise from true North.
	"""
	earth_radius = 6371e3
	angular_distance = distance / earth_radius
	lat2 = np.degrees(np.arcsin(np.sin(np.radians(lat1)) 
		* np.cos(angular_distance) + np.cos(np.radians(lat1)) 
		* np.sin(angular_distance) * np.cos(np.radians(bearing))))
	# Longitude is mod 360 degrees for wrapping around earth
	lon2 = (lon1 + np.degrees(np.arctan2(np.sin(np.radians(bearing)) 
		* np.sin(angular_distance) * np.cos(np.radians(lat1)), 
		np.cos(angular_distance) - np.sin(np.radians(lat1)) 
		* np.sin(np.radians(lat2))))) % 360       
	return lat2, lon2

def compass_bearing(math_bearing):
	""" Transform a vector angle to a compass bearing."""
	bearing = (5 * np.pi / 2 - math_bearing) % (2 * np.pi)
	return bearing

def calc_dx_dy(longitude,latitude):
	''' This definition calculates the distance between grid points that are in
		a latitude/longitude format.
		
		Equations from:
		http://andrew.hedges.name/experiments/haversine/

		dy should be close to 55600 m
		dx at pole should be 0 m
		dx at equator should be close to 55600 m
		
		Accepts, 1D arrays for latitude and longitude
		
		Returns: dx, dy; 2D arrays of distances between grid points 
									in the x and y direction in meters 
	'''
	dlat = np.abs(latitude[1]-latitude[0])*np.pi/180
	dy = 2*(np.arctan2(np.sqrt((np.sin(dlat/2))**2),
			np.sqrt(1-(np.sin(dlat/2))**2)))*6371000
	dy = np.ones((latitude.shape[0],longitude.shape[0]))*dy

	dx = np.empty((latitude.shape))
	dlon = np.abs(longitude[1] - longitude[0])*np.pi/180
	for i in range(latitude.shape[0]):
		a = (np.cos(latitude[i]*np.pi/180)
			*np.cos(latitude[i]*np.pi/180)*np.sin(dlon/2))**2
		c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a) )
		dx[i] = c * 6371000
	dx = np.repeat(dx[:,np.newaxis],longitude.shape,axis=1)
	return dx, dy

def midpoint(lat1, lon1, lat2, lon2):
	""" Midpoint along a great circle path between two points. 
	Equation from http://www.movable-type.co.uk/scripts/latlong.html 

	Latitudes are from -90 to 90 degrees.
	Longitudes are from 0 to 360 degrees. """
	delta_lon = np.absolute(lon2 - lon1)
	bx = np.cos(np.radians(lat2)) * np.cos(np.radians(delta_lon))
	by = np.cos(np.radians(lat2)) * np.sin(np.radians(delta_lon))
	lat_mid = np.degrees(np.arctan2(np.sin(np.radians(lat1)) 
				+ np.sin(np.radians(lat2)),
				np.sqrt((np.cos(np.radians(lat1)) + bx) ** 2 ) + by ** 2))
	lon_mid = lon1 + np.degrees(np.arctan2(by, np.cos(np.radians(lat1)) + bx))
	return lat_mid, lon_mid    

def gh_gradient(grid):
	# Make this take sphere into account
	grad_gh_rows, grad_gh_columns = np.gradient(grid)
	return grad_gh_rows, grad_gh_columns

def speed_force(lat, lon, filename):
	file = netcdf.netcdf_file(filename, mmap=False)
	vars = file.variables
	file.close()

	# Get indices of latitude and longitude in grid
	lat_index = np.where(vars['lat'][:] == lat)[0][0]
	lon_index = np.where(vars['lon'][:] == lon)[0][0]

	# Get geopotential height
	gh = vars['gh'][0][0]

	# Get 1D arrays of longitude and latitude
	latitudes = vars['lat'][:]
	longitudes = vars['lon'][:]

	# Get acceleration in u and v components from geopotential height
	length_grid_v, length_grid_u = calc_dx_dy(longitudes, latitudes)
	v_gradient_grid, u_gradient_grid = gh_gradient(gh)
	v_gradient_grid_meters = v_gradient_grid / length_grid_v
	u_gradient_grid_meters = u_gradient_grid / length_grid_u
	v_gradient = v_gradient_grid_meters[lat_index][lon_index]
	u_gradient = u_gradient_grid_meters[lat_index][lon_index]

	# should g in fact be negative?
	g = -9.806 # ms^-2
	omega = 7.2921150e-5 # radians per second	
	# are u_gradient and v_gradient 
	u_speed = (-g / (2 * omega * np.sin(np.radians(lat)))) * v_gradient
	v_speed = (g / (2 * omega * np.sin(np.radians(lat)))) * u_gradient
	return u_speed, v_speed 

def speed_grid(lat, lon, filename):
	"""  """
	file = netcdf.netcdf_file(filename, mmap=False)
	vars = file.variables
	file.close()   

	# Get indices of latitude and longitude in grid
	lat_index = np.where(vars['lat'][:] == lat)[0][0]
	lon_index = np.where(vars['lon'][:] == lon)[0][0]

	# Get u and v for current latitude
	u_speed = vars['u'][0][0][lat_index][lon_index]    # meters per second
	v_speed = vars['v'][0][0][lat_index][lon_index]    # meters per second
	return u_speed, v_speed

def velocity_components(position_lat, position_lon, filename, scheme):
	""" This gets the u and v components of wind velocity at a point according 
		to the current calculation scheme. 

		Schemes:
		grid -- calculate from wind field
		force -- calculate advection from geostropic wind equation using 
				 geopotential height
	"""
	# Get closest grid position to starting coordinates
	grid_lat = round_to_grid(position_lat)
	grid_lon = round_to_grid(position_lon) % 360

	if scheme == "grid":
		u_speed, v_speed = speed_grid(grid_lat, grid_lon, filename)

	elif scheme == "force":
		u_speed, v_speed = speed_force(grid_lat, grid_lon, filename)

	else:
		print("Invalid calculation scheme.")

	return u_speed, v_speed, grid_lat, grid_lon	

# Open a file and calculate next latitude and longitude from wind speed
def next_position(position_lat, position_lon, filename, scheme):
	""" This gets the next position in a trajectory according to the given 
		calculation scheme, by multiplying velocity with the timestep. 

		Schemes:
		grid -- calculate from wind field
		force -- calculate advection from geostropic wind equation using 
				 geopotential height
	"""

	u_speed, v_speed, grid_lat, grid_lon = velocity_components(position_lat, 
											position_lon, filename, scheme)

	# Get magnitude and direction of wind vector
	wind_speed = spherical_hypotenuse(u_speed, v_speed)
	wind_direction = np.arctan2(v_speed, u_speed)
	wind_vector = np.array([wind_speed, wind_direction])

	# Get displacement using velocity times delta t
	delta_t = 3 * 60 * 60    # in seconds
	displacement = wind_speed * delta_t    # in meters

	# Calculate new latitude and longitude
	# Convert wind bearing to degrees
	wind_bearing = np.degrees(compass_bearing(wind_direction))
	new_lat, new_lon = destination(grid_lat, grid_lon, displacement, 
																wind_bearing)
	return new_lat, new_lon

def filenames():
	"""Generate filenames for test set of GFS netCDF files.
	Assumes there's a folder called "data" in this library's folder. """
	filenames = ['data/hgt-{:03d}.nc'.format(time) for time in np.arange(0,241,3)]
	return filenames

def runge_kutta_trajectory(lat, lon, scheme="grid"):
	num_files = 81
	trajectory = np.zeros((num_files,2))
	# Initial position
	# Green building is 42.3603088, -71.0893148
	initial_lat, initial_lon = np.array(initial_position(lat,lon))
	current_lat, current_lon = initial_lat, initial_lon

	# List all filenames 
	files = filenames()

	# Define timestep
	delta_t = 3 * 60 * 60    # in seconds

	# Implement Runge Kutta integration method
	for i in np.arange(num_files-1):
		guess_lat, guess_lon = next_position(current_lat, current_lon, 
											files[i], scheme)
		initial_u, initial_v, grid_lat, grid_lon = velocity_components(
			lat, lon, files[i], scheme)
		guess_u, guess_v, grid_lat, grid_lon = velocity_components(
			guess_lat, guess_lon, files[i+1], scheme)
		average_u = 0.5 * (initial_u + guess_u)
		average_v = 0.5 * (initial_v + guess_v)

		wind_speed = spherical_hypotenuse(average_u, average_v)
		wind_direction = np.arctan2(average_v, average_u)
		wind_bearing = np.degrees(compass_bearing(wind_direction))
		displacement = wind_speed * delta_t    # in meters

		trajectory[i,:] = destination(current_lat, current_lon, displacement, wind_bearing)
		current_lat = trajectory[i,0]
		current_lon = trajectory[i,1]
	rk_lat, rk_lon = trajectory[:-1,0], trajectory[:-1,1]

	return rk_lat, rk_lon

# Get trajectory from computed wind fields
def trajectory(lat, lon, scheme="grid"):
	num_files = 81
	trajectory = np.zeros((num_files,2))
	# Initial position
	# Green building is 42.3603088, -71.0893148
	initial_lat, initial_lon = np.array(initial_position(lat,lon))
	current_lat, current_lon = initial_lat, initial_lon
	for i, filename in enumerate(filenames()):
		trajectory[i,:] = next_position(current_lat, current_lon, 
										filename, scheme)
		current_lat = trajectory[i,0]
		current_lon = trajectory[i,1]
	trajectory_lat, trajectory_lon = trajectory[:,0], trajectory[:,1]
	trajectory_lat = np.insert(trajectory_lat, 0, initial_lat)
	trajectory_lon = np.insert(trajectory_lon, 0, initial_lon)
	return trajectory_lat, trajectory_lon

def plot_ortho(trajectory_lat, trajectory_lon, lat_center=90, lon_center=-105,
		savefig=False):
	map = Basemap(projection='ortho',lon_0=-105,lat_0=90,resolution='c')
	map.drawcoastlines(linewidth=0.25, color='gray')
	map.drawcountries(linewidth=0)
	map.fillcontinents(color='white',lake_color='white', zorder=1)
	# draw the edge of the map projection region (the projection limb)
	map.drawmapboundary(fill_color='white')
	map.plot(trajectory_lon, trajectory_lat, latlon=True, zorder=2, 
				color='black', marker='.')
	#plt.title("First Order v * dt Integration")
	if savefig == True:
		filename = "trajectory_"+sys.argv[1]+"_"+sys.argv[2]+".eps"
		plt.savefig(filename)
	plt.show()	