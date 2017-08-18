# Plot the ratio of geostrophic wind to observed wind for all points

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
import scipy.interpolate
from mpl_toolkits.basemap import Basemap
import sys

plot_type = "graph speed ratio"

# Global variables
EARTH_RADIUS = 6371e3    # meters
OMEGA = 7.292e-5         # radians per second
STANDARD_GRAVITY = 9.806 # meters per second squared

filename = "data/hgt-000.nc"
file = netcdf.netcdf_file(filename, mmap=False)
vars = file.variables
file.close()

# Get geopotential height
lat = vars["lat"][:][:]
lon = vars["lon"][:][:] % 360
u = vars["u"][0][0]
v = vars["v"][0][0]
gh = vars["gh"][0][0]

# Make a grid of longitude and latitude coordinates
lon_grid, lat_grid = np.meshgrid(lon, lat)

spacing_lat = np.radians(180 / (np.size(lat_grid, axis=0)))  # radians
spacing_lon = np.radians(360 / (np.size(lat_grid, axis=1)))  # radians
dgh_dlat, dgh_dlon = np.gradient(gh, spacing_lat, spacing_lon)

# Calculate geostrophic velocity
f = 2 * OMEGA * np.sin(np.radians(lat_grid))
u_g = ((-STANDARD_GRAVITY / f) * dgh_dlat * (1 / 
    (EARTH_RADIUS + gh)))
v_g = ((STANDARD_GRAVITY / f) * dgh_dlon * (1 / 
    ((EARTH_RADIUS + gh) * np.cos(np.radians(lat_grid)))))

# Calculate magnitude of speed
magnitude = np.sqrt(u ** 2 + v ** 2)
magnitude_g = np.sqrt(u_g ** 2 + v_g ** 2)

# Calculate averages for each latitude
u_mean = np.mean(u, axis=1)
v_mean = np.mean(v, axis=1)
magnitude_mean = np.mean(magnitude, axis=1)
u_g_mean = np.mean(u_g, axis=1)
v_g_mean = np.mean(v_g, axis=1)
magnitude_g_mean = np.mean(magnitude_g, axis=1)

# Speed ratios
u_ratio = u_g_mean / u_mean
v_ratio = v_g_mean / v_mean
magnitude_ratio = magnitude_g_mean / magnitude_mean

# Line at 1
one = np.ones_like(lat)

# Plots
if plot_type == "zonal":
	map = Basemap(projection="ortho", lon_0=-105, lat_0=42, resolution="c")
	map.contourf(lon_grid, lat_grid, u, latlon=True, zorder=2)
	map.drawcoastlines(linewidth=0.25, zorder=3)
	map.fillcontinents(color="white", lake_color="white", zorder=0)
	map.drawmapboundary(fill_color="white", zorder=0)
	plt.title("Zonal Speed")
	plt.show()

elif plot_type == "meridional":
	map = Basemap(projection="ortho", lon_0=-105, lat_0=42, resolution="c")
	map.contourf(lon_grid, lat_grid, v, latlon=True, zorder=2)
	map.drawcoastlines(linewidth=0.25, zorder=3)
	map.fillcontinents(color="white", lake_color="white", zorder=0)
	map.drawmapboundary(fill_color="white", zorder=0)
	plt.title("Meridional Speed")
	plt.show()

elif plot_type == "magnitude":
	map = Basemap(projection="ortho", lon_0=-105, lat_0=42, resolution="c")
	map.contourf(lon_grid, lat_grid, magnitude, latlon=True, zorder=2)
	map.drawcoastlines(linewidth=0.25, zorder=3)
	map.fillcontinents(color="white", lake_color="white", zorder=0)
	map.drawmapboundary(fill_color="white", zorder=0)
	plt.title("Magnitude of Velocity")
	plt.show()

elif plot_type == "zonal geostrophic":
	map = Basemap(projection="ortho", lon_0=-105, lat_0=42, resolution="c")
	map.contourf(lon_grid, lat_grid, u_g, latlon=True, zorder=2)
	map.drawcoastlines(linewidth=0.25, zorder=3)
	map.fillcontinents(color="white", lake_color="white", zorder=0)
	map.drawmapboundary(fill_color="white", zorder=0)
	plt.title("Zonal Geostrophic Speed")
	plt.show()

elif plot_type == "meridional geostrophic":
	map = Basemap(projection="ortho", lon_0=-105, lat_0=42, resolution="c")
	map.contourf(lon_grid, lat_grid, v_g, latlon=True, zorder=2)
	map.drawcoastlines(linewidth=0.25, zorder=3)
	map.fillcontinents(color="white", lake_color="white", zorder=0)
	map.drawmapboundary(fill_color="white", zorder=0)
	plt.title("Meridional Geostrophic Speed")
	plt.show()

elif plot_type == "magnitude geostrophic":
	map = Basemap(projection="ortho", lon_0=-105, lat_0=42, resolution="c")
	map.contourf(lon_grid, lat_grid, v_g, latlon=True, zorder=2)
	map.drawcoastlines(linewidth=0.25, zorder=3)
	map.fillcontinents(color="white", lake_color="white", zorder=0)
	map.drawmapboundary(fill_color="white", zorder=0)
	plt.title("Magnitude of Geostrophic Speed")
	plt.show()

elif plot_type == "graph speed":
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 1, 1)
	ax1.scatter(magnitude_mean, lat)
	plt.show()

elif plot_type == "graph geostrophic speed":
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 1, 1)
	ax1.scatter(u_g_mean, lat)
	ax1.scatter(magnitude_g_mean, lat)
	plt.show()

elif plot_type == "graph speed ratio":
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 1, 1)
	ax1.plot(one, lat, color="0.5", label="geostrophic = observed")
	ax1.plot(magnitude_ratio, lat, color="black", label="measured ratio")
	plt.legend()
	ax1.set_ylim(-90, 90)
	plt.xlabel("Ratio of zonally averaged geostrophic to observed wind speed")
	plt.ylabel("Latitude")
	plt.savefig("plots/speed_ratio.pdf")
	plt.show()

else:
	pass
