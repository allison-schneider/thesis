# Author: Allison Schneider
# Date: 2017 June 28

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
from mpl_toolkits.basemap import Basemap
import sys

def initial_position(lat,lon):
	"""Choose the initial position. 
	Transforms longitude from [-180,180) to [0,360).

	Arguments:
	lat -- Latitude in degrees, -90 to 90
	lon -- Longitude in degrees, -180 to 180"""
	starting_lat = lat
	starting_lon = lon + 180
	return starting_lat, starting_lon

def round_to_grid(float, spacing=0.5):
    """Round a latitude or longitude to the closest grid value.

    Arguments:
    float -- A floating point number or numpy array
    spacing -- Distance between grid points"""
    return np.round(float / spacing) * spacing

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
    dy = 2*(np.arctan2(np.sqrt((np.sin(dlat/2))**2),np.sqrt(1-(np.sin(dlat/2))**2)))*6371000
    dy = np.ones((latitude.shape[0],longitude.shape[0]))*dy

    dx = np.empty((latitude.shape))
    dlon = np.abs(longitude[1] - longitude[0])*np.pi/180
    for i in range(latitude.shape[0]):
        a = (np.cos(latitude[i]*np.pi/180)*np.cos(latitude[i]*np.pi/180)*np.sin(dlon/2))**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a) )
        dx[i] = c * 6371000
    dx = np.repeat(dx[:,np.newaxis],longitude.shape,axis=1)
    return dx, dy

def gh_gradient(grid):
	# Make this take sphere into account
	grad_gh_rows, grad_gh_columns = np.gradient(grid)
	return grad_gh_rows, grad_gh_columns

filename = "hgt-000.nc"
file = netcdf.netcdf_file(filename, mmap=False)
vars = file.variables
file.close()

# Open a file and calculate next latitude and longitude from geostrophic balance
def next_position_force(position_lat, position_lon, filename):
	file = netcdf.netcdf_file(filename, mmap=False)
	vars = file.variables
	file.close()

	# Get closest grid position to starting coordinates
	grid_lat = round_to_grid(position_lat)
	grid_lon = round_to_grid(position_lon)    

	# Get indices of latitude and longitude in grid
	lat_index = np.where(vars['lat'][:] == grid_lat)[0][0]
	lon_index = np.where(vars['lon'][:] == grid_lon)[0][0]

	# Get geopotential height
	gh = vars['gh'][0][0]

	# Get 1D arrays of longitude and latitude
	latitudes = vars['lat'][:]
	longitudes = vars['lon'][:]

	# Get acceleration in u and v components from geopotential height
	# Add coriolis force to this
	length_grid_v, length_grid_u = calc_dx_dy(longitudes, latitudes)
	v_gradient_grid, u_gradient_grid = gh_gradient(gh)
	v_gradient_grid_meters = v_gradient_grid / length_grid_v
	u_gradient_grid_meters = u_gradient_grid / length_grid_u
	v_gradient = v_gradient_grid_meters[lat_index][lon_index]
	u_gradient = u_gradient_grid_meters[lat_index][lon_index]

	g = 9.806 # ms^-2
	omega = 7.2921150e-5 # radians per second	
	u_speed = (-g / (2 * omega * np.sin(np.radians(grid_lat)))) * u_gradient
	v_speed = (g / (2 * omega * np.sin(np.radians(grid_lat)))) * v_gradient

	# print("v accel shape is ", np.shape(v_gradient_grid))
	# print("u accel shape is ", np.shape(u_gradient_grid))
	# print("u gradient is ", u_gradient)
	# print("v gradient is ", v_gradient)
	print("v speed is", v_speed)
	print("u speed is", u_speed)
	# print("phi is ", grid_lat)
	# print("longitude is ", longitudes)
	# print("longitude shape is ", np.shape(longitudes))
	# print("latitude is ", latitudes)
	# print("latitude shape is ", np.shape(latitudes))
	#print("length grid shape is ", np.shape(length_grid))
	return gh

def trajectory_force(lat, lon):
	trajectory = np.zeros((81,2))
	# Initial position
	# Green building is 42.3603088, -71.0893148
	current_lat, current_lon = initial_position(lat, lon)
	for i, time in enumerate(np.arange(0,241,3)):
	    filename = 'hgt-{:03d}.nc'.format(time)
	    trajectory[i,:] = next_position(current_lat, current_lon, filename)
	    current_lat = trajectory[i,0]
	    current_lon = trajectory[i,1]
	trajectory_lat, trajectory_lon = trajectory[:,0], trajectory[:,1]
	return trajectory_lat, trajectory_lon	

def main():
	next_position_force(41,71, "hgt-000.nc")

if __name__ == "__main__":
	main()
