# Plot a wind field at a single point in time

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
from mpl_toolkits.basemap import Basemap

def spherical_hypotenuse(a, b):
    """ Given the lengths of two sides of a right triangle on a sphere, 
    a and b, find the length of the hypotenuse c. 
    
    Arguments:
    a -- Length of first side of the triangle, in meters
    b -- Length of second side of the triangle, in meters 
    """
    earth_radius = 6371e3    # meters
    c = earth_radius * np.arccos(np.cos(a / earth_radius) * np.cos(b / earth_radius))
    return c

def wind_vectors():
	filename = "hgt-000.nc"
	file = netcdf.netcdf_file(filename, mmap=False)
	vars = file.variables
	file.close()

	lat = vars["lat"][:]
	lon = vars["lon"][:]
	u = vars["u"][0][0]
	v = vars["v"][0][0]

	wind_speed = spherical_hypotenuse(u, v)

	x, y = np.meshgrid(lon, lat)

	# Definie a slicing to sample points from the grid
	skip = (slice(None,None,10),slice(None,None,10))

	x_select, y_select, u_select, v_select, wind_speed_select = (
		x[skip], y[skip], u[skip], v[skip], wind_speed[skip])

	return x_select, y_select, u_select, v_select, wind_speed_select

def main():
	x, y, u, v, wind_speed = wind_vectors()

	map = Basemap(projection='ortho', lat_0=42, lon_0=-71, resolution='c')
	map.drawcoastlines(linewidth=0.25)
	map.drawcountries(linewidth=0.25)
	map.fillcontinents(color='coral',lake_color='aqua')
	# draw the edge of the map projection region (the projection limb)
	map.drawmapboundary(fill_color='aqua')
	map.quiver(x, y, u, v, wind_speed, latlon=True, cmap=plt.cm.autumn)
	plt.show()

def mercator():
	x, y, u, v, wind_speed = wind_vectors()

	map = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
	map.drawcoastlines(linewidth=0.25)
	map.drawcountries(linewidth=0)
	map.fillcontinents(color='coral',lake_color='aqua', zorder=0)
	# draw the edge of the map projection region (the projection limb)
	map.drawmapboundary(fill_color='aqua')
	map.quiver(x, y, u, v, wind_speed, latlon=True, cmap=plt.cm.plasma)
	plt.show()	

def lambert_marble():
	filename = "hgt-000.nc"
	file = netcdf.netcdf_file(filename, mmap=False)
	vars = file.variables
	file.close()

	lat = vars["lat"][:]
	lon = vars["lon"][:]
	u = vars["u"][0][0]
	v = vars["v"][0][0]

	wind_speed = spherical_hypotenuse(u, v)

	x, y = np.meshgrid(lon, lat)

	# Definie a slicing to sample points from the grid
	skip = (slice(None,None,10),slice(None,None,10))

	map = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
	map.quiver(x[skip], y[skip], u[skip], v[skip], wind_speed[skip], latlon=True, cmap=plt.cm.RdPu)
	map.bluemarble()
	plt.show()

def rectangular_field():
	filename = "hgt-000.nc"
	file = netcdf.netcdf_file(filename, mmap=False)
	vars = file.variables
	file.close()

	lat = vars["lat"][:]
	lon = vars["lon"][:]
	u = vars["u"][0][0]
	v = vars["v"][0][0]

	x, y = np.meshgrid(lon, lat)

	skip = (slice(None,None,10),slice(None,None,10))

	plt.figure()
	plt.quiver(x[skip], y[skip], u[skip], v[skip])
	plt.show()

if __name__ == '__main__':
	mercator()