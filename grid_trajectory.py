# Author: Allison Schneider
# Date: 2017 June 14

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

#position_lat, position_lon = initial_position(42.3603088, -71.0893148)

def speed_grid(lat, lon, filename):
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

# Open a file and calculate next latitude and longitude from wind speed
def next_position(position_lat, position_lon, filename):
	# Get closest grid position to starting coordinates
	grid_lat = round_to_grid(position_lat)
	grid_lon = round_to_grid(position_lon) 

	u_speed, v_speed = speed_grid(grid_lat, grid_lon, filename)

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

# Get trajectory from computed wind fields
def trajectory_grid(lat, lon):
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

	## Plot a subset of the trajectory
	# plt.figure()
	# plt.plot(trajectory[0:30,0], trajectory[0:30,1])
	# plt.show()

# Plot trajectory on a globe
def globe_plot():
	trajectory_lat, trajectory_lon = trajectory_grid(lat, lon)
	map = Basemap(projection='ortho', lat_0=90, lon_0=90, resolution='c')
	map.drawcoastlines(linewidth=0.25)
	map.drawcountries(linewidth=0.25)
	map.fillcontinents(color='coral',lake_color='aqua', zorder=1)
	# draw the edge of the map projection region (the projection limb)
	map.drawmapboundary(fill_color='aqua')
	map.plot(trajectory_lon, trajectory_lat, latlon=True, zorder=2)
	plt.show()

def test_plot():
	lat, lon = float(sys.argv[1]), float(sys.argv[2])
	trajectory_lat, trajectory_lon = trajectory_grid(lat, lon)
	# Adjust longitude to -180 to 180 range
	trajectory_lon = trajectory_lon - 180
	map = Basemap(projection='ortho',lon_0=-105,lat_0=90,resolution='c')
	map.drawcoastlines(linewidth=0.25)
	map.drawcountries(linewidth=0)
	map.fillcontinents(color='coral',lake_color='aqua', zorder=1)
	# draw the edge of the map projection region (the projection limb)
	map.drawmapboundary(fill_color='aqua')
	map.plot(trajectory_lon, trajectory_lat, latlon=True, zorder=2)
	#plt.savefig('first_trajectory.svg')
	filename = "trajectory_"+sys.argv[1]+"_"+sys.argv[2]+".png"
	plt.savefig(filename)
	plt.show()	

if __name__ == "__main__":
	test_plot()
