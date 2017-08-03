import numpy as np

latitudes = np.linspace(41, 42, num=5)
longitudes = np.linspace(-72, -71, num=5)

grid_lon, grid_lat = np.meshgrid(longitudes, latitudes)
lon = np.ndarray.flatten(grid_lon)
lat = np.ndarray.flatten(grid_lat)

for i in np.arange(25):
	print(lat[i], lon[i])