"""
A relatively inaccurate method for calculating the an atmospheric trajectory.
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
def trajectory(lat, lon):
	num_files = 81
	trajectory = np.zeros((num_files,2))
	# Initial position
	# Green building is 42.3603088, -71.0893148
	initial_lat, initial_lon = np.array(initial_position(lat,lon))
	current_lat, current_lon = initial_lat, initial_lon
	trajectory 
	for i, time in enumerate(np.arange(0,241,3)):
	    filename = 'hgt-{:03d}.nc'.format(time)
	    trajectory[i,:] = next_position(current_lat, current_lon, filename)
	    current_lat = trajectory[i,0]
	    current_lon = trajectory[i,1]
	trajectory_lat, trajectory_lon = trajectory[:,0], trajectory[:,1]
	trajectory_lat = np.insert(trajectory_lat, 0, initial_lat)
	trajectory_lon = np.insert(trajectory_lon, 0, initial_lon)
	return trajectory_lat, trajectory_lon