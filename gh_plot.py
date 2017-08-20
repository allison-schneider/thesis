import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
from mpl_toolkits.basemap import Basemap

filename = "data/hgt-000.nc"
file = netcdf.netcdf_file(filename, mmap=False)
vars = file.variables
file.close()

# Get geopotential height
lat = vars["lat"][:][:]
lon = vars["lon"][:][:] % 360
gh = vars["gh"][0][0]

lon_grid, lat_grid = np.meshgrid(lon, lat)

map_type = "cyl"

if map_type == "ortho":
    m = Basemap(projection="ortho", lon_0=-105, lat_0=42, resolution="c")
    m.drawcoastlines(linewidth=0.25, zorder=3)
    m.fillcontinents(color="white", lake_color="white", zorder=0)
    m.drawmapboundary(fill_color="white", zorder=0)
    m.contourf(lon_grid, lat_grid, gh, latlon=True, zorder=2)
    plt.title("Geopotential Height at t=0")
    plt.show()

if map_type == "cyl":
    # Only works with Python 2, not 3
    m = Basemap(projection="cyl",llcrnrlat=-90,urcrnrlat=90,
            llcrnrlon=-180,urcrnrlon=180,resolution='c')
    m.drawcoastlines(linewidth=0.3, zorder=3)
    m.fillcontinents(color="white", lake_color="white", zorder=0)
    m.drawmapboundary(fill_color="white", zorder=0)
    # draw parallels.
    parallels = np.arange(-90,90,30.)
    m.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.25, fontsize=8)
    # draw meridians
    meridians = np.arange(0,360.,30.)
    m.drawmeridians(meridians, labels=[0,0,0,1], linewidth=0.25, fontsize=8)
    gh = m.pcolormesh(lon_grid, lat_grid, gh, latlon=True, 
        zorder=2, cmap="plasma")
    plt.title("Geopotential Height at t=0")
    cbar = m.colorbar(gh, location='bottom', pad="20%")
    plt.show()

#if map_type == "multicyl"

# # Make a movie with all time values
# for i, time in enumerate(np.arange(0,241,3)):
#   filename = 'data/hgt-{:03d}.nc'.format(time)
#   file = netcdf.netcdf_file(filename, mmap=False)
#   vars = file.variables
#   file.close()

#   # Get geopotential height
#   lat = vars["lat"][:][:]
#   lon = vars["lon"][:][:] % 360
#   gh = vars["gh"][0][0]

#   lon_grid, lat_grid = np.meshgrid(lon, lat)
#   m = Basemap(projection="ortho", lon_0=-105, lat_0=42, resolution="c")
#   m.drawcoastlines(linewidth=0.25, zorder=3)
#   m.fillcontinents(color="white", lake_color="white", zorder=0)
#   m.drawmapboundary(fill_color="white", zorder=0)
#   m.contourf(lon_grid, lat_grid, gh, latlon=True, zorder=2)
#   title = "Geopotential Height at t={}".format(time)
#   plt.title(title)
#   plt.show(block=False)
#   plt.pause(1.0/60.0)