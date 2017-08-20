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
    # Only works with Python 2, not 
    m = Basemap(projection="cyl", llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, resolution='c')
    m.drawcoastlines(linewidth=0.3, zorder=3)
    m.fillcontinents(color="white", lake_color="white", zorder=0)
    m.drawmapboundary(fill_color="white", zorder=0)
    # draw parallels
    parallels = np.arange(-90, 90, 30.)
    m.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.25, fontsize=8)
    # draw meridians
    meridians = np.arange(0, 360., 30.)
    m.drawmeridians(meridians, labels=[0,0,0,1], linewidth=0.25, fontsize=8)
    gh = m.pcolormesh(lon_grid, lat_grid, gh, latlon=True, 
        zorder=2, cmap="plasma", vmin=9300, vmax=11000)

    m.colorbar(gh, location='bottom', pad="20%")
    plt.show()

if map_type == "multicyl5":
    # Only works with Python 2, not 3

    fig, axes = plt.subplots(5, 1)
    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', 
        right='off')

    axes[0].set_title("$t=0$")
    m0 = Basemap(projection="cyl", llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, resolution='c', ax=axes[0])
    m0.drawcoastlines(linewidth=0.3, zorder=3)
    m0.fillcontinents(color="white", lake_color="white", zorder=0)
    m0.drawmapboundary(fill_color="white", zorder=0)
    # draw parallels
    parallels = np.arange(-90, 90, 30.)
    m0.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.25, fontsize=8)
    # draw meridians
    meridians = np.arange(0, 360., 30.)
    m0.drawmeridians(meridians, labels=[0,0,0,1], linewidth=0.25, fontsize=8)
    m0.pcolormesh(lon_grid, lat_grid, gh, latlon=True, 
        zorder=2, cmap="plasma")

    axes[1].set_title("$t=0$")
    m2 = Basemap(projection="cyl", llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, resolution='c', ax=axes[1])
    m2.drawcoastlines(linewidth=0.3, zorder=3)
    m2.fillcontinents(color="white", lake_color="white", zorder=0)
    m2.drawmapboundary(fill_color="white", zorder=0)
    m2.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.25, fontsize=8)
    m2.drawmeridians(meridians, labels=[0,0,0,1], linewidth=0.25, fontsize=8)
    m2.pcolormesh(lon_grid, lat_grid, gh, latlon=True, 
        zorder=2, cmap="plasma")

    axes[2].set_title("$t=0$")
    m4 = Basemap(projection="cyl", llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, resolution='c', ax=axes[2])
    m4.drawcoastlines(linewidth=0.3, zorder=3)
    m4.fillcontinents(color="white", lake_color="white", zorder=0)
    m4.drawmapboundary(fill_color="white", zorder=0)
    m4.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.25, fontsize=8)
    m4.drawmeridians(meridians, labels=[0,0,0,1], linewidth=0.25, fontsize=8)
    m4.pcolormesh(lon_grid, lat_grid, gh, latlon=True, 
        zorder=2, cmap="plasma")

    axes[3].set_title("$t=0$")
    m6 = Basemap(projection="cyl", llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, resolution='c', ax=axes[3])
    m6.drawcoastlines(linewidth=0.3, zorder=3)
    m6.fillcontinents(color="white", lake_color="white", zorder=0)
    m6.drawmapboundary(fill_color="white", zorder=0)
    m6.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.25, fontsize=8)
    m6.drawmeridians(meridians, labels=[0,0,0,1], linewidth=0.25, fontsize=8)
    m6.pcolormesh(lon_grid, lat_grid, gh, latlon=True, 
        zorder=2, cmap="plasma")

    axes[4].set_title("$t=0$")
    m8 = Basemap(projection="cyl", llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, resolution='c', ax=axes[4])
    m8.drawcoastlines(linewidth=0.3, zorder=3)
    m8.fillcontinents(color="white", lake_color="white", zorder=0)
    m8.drawmapboundary(fill_color="white", zorder=0)
    m8.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.25, fontsize=8)
    m8.drawmeridians(meridians, labels=[0,0,0,1], linewidth=0.25, fontsize=8)
    m8.pcolormesh(lon_grid, lat_grid, gh, latlon=True, 
        zorder=2, cmap="plasma")

    cbar = fig.colorbar(gh, location='bottom')
    plt.show()

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