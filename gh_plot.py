import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
from mpl_toolkits.basemap import Basemap

filename = "hgt-000.nc"
file = netcdf.netcdf_file(filename, mmap=False)
vars = file.variables
file.close()

# Get geopotential height
lat = vars["lat"][:][:]
lon = vars["lon"][:][:] - 180
gh = vars["gh"][0][0]

lon_grid, lat_grid = np.meshgrid(lon, lat)

print(gh[0])


# Plot one time value
# map = Basemap(projection="ortho", lon_0=-105, lat_0=42, resolution="c")
# #map = Basemap(projection="cyl")
# map.drawcoastlines(linewidth=0.25, zorder=3)
# map.fillcontinents(color="white", lake_color="white", zorder=0)
# map.drawmapboundary(fill_color="white", zorder=0)
# map.contourf(lon_grid, lat_grid, gh, latlon=True, zorder=2)
# plt.title("Geopotential Height at t=0")
# plt.show()

# Make a movie with all time values
for i, time in enumerate(np.arange(0,241,3)):
	filename = 'hgt-{:03d}.nc'.format(time)
	file = netcdf.netcdf_file(filename, mmap=False)
	vars = file.variables
	file.close()

	# Get geopotential height
	lat = vars["lat"][:][:]
	lon = vars["lon"][:][:] - 180
	gh = vars["gh"][0][0]

	lon_grid, lat_grid = np.meshgrid(lon, lat)
	map = Basemap(projection="ortho", lon_0=-105, lat_0=42, resolution="c")
	map.drawcoastlines(linewidth=0.25, zorder=3)
	map.fillcontinents(color="white", lake_color="white", zorder=0)
	map.drawmapboundary(fill_color="white", zorder=0)
	map.contourf(lon_grid, lat_grid, gh, latlon=True, zorder=2)
	title = "Geopotential Height at t={}".format(time)
	plt.title(title)
	plt.show(block=False)
	plt.pause(1.0/20.0)