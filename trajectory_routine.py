import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
import scipy.interpolate
from mpl_toolkits.basemap import Basemap
import sys

# Global variables
EARTH_RADIUS = 6371e3    # meters

class Atmosphere:
    """ Contains atmospheric parameters (u speed, v speed, geopotential height
    gradient) for two time layers at a given time.

    hour -- The time for the first layer, a multiple of the prediction interval. 
    """
    def __init__(self,
                 hour):      # First layer time in hours

        self.hour = hour
        self.hours_between_files = 3    # Time in hours between samples
        self.total_hours = 240          # Time to run trajectory in hours

        # Test that the argument is a valid hour.
        if self.hour > self.total_hours - self.hours_between_files:
            raise ValueError("Maximum value for hour argument is {}."
                           .format(self.total_hours - self.hours_between_files))

        if self.hour % self.hours_between_files != 0:
            raise ValueError("Hour argument must be a multiple of {}."
                             .format(self.hours_between_files))

        # Generate list of all filenames and index of current file
        self.filenames = ['data/hgt-{:03d}.nc'.format(time) 
                            for time in np.arange(0, 
                                                  self.total_hours + 1, 
                                                  self.hours_between_files)]
        self.file_index = np.int(np.floor(self.hour / self.hours_between_files))                 
        
        # Initialize values for interpolate function
        self.points = self.build_points()
        self.u_values = self.values("u")
        self.v_values = self.values("v")
        self.gh_values = self.values("gh")
        self.gh_dlat, self.gh_dlon = self.gh_gradient()
        
    def build_points(self):
        """ Returns the first argument for the interpn function,
        a tuple of 3 ndarrays of float, named points, 
        of shape ((nlats,), (nlons,), (ntimes)), 
        which stays constant throughout the trajectory.
        """
        file = netcdf.netcdf_file(self.filenames[self.file_index], mmap=False)
        vars = file.variables
        file.close()

        # Flip latitudes so they're increasing, (-90, 90).
        latitudes = np.flip(vars['lat'][:], 0)   
        # Add the longitude 360 to deal with interpolation near 0 degrees.
        longitudes = np.append(vars['lon'][:], 360)
       
        # Array of times has last and next sample times, or current and next.
        times = np.array([self.hour, self.hour + self.hours_between_files])
        
        points = latitudes, longitudes, times
        return points

    def values(self, parameter):
        """ Returns the second argument for the interpn function,
        a 3D ndarray of float, named values, of shape (nlats, nlons, ntimes),
        which is updated when the trajectory passes a new time layer.

        parameter -- 'u', 'v', or 'gh'; the atmospheric parameter. 
        """
        
        # Open first file and retrieve gridded values
        file0 = netcdf.netcdf_file(self.filenames[self.file_index], mmap=False)
        vars0 = file0.variables
        file0.close()
        t0_values = vars0[parameter][0][0]
        t0_values = t0_values[:, :, np.newaxis]

        # Open second file and retrieve gridded values
        file1 = netcdf.netcdf_file(self.filenames[self.file_index + 1], 
                                   mmap=False)
        vars1 = file1.variables
        file1.close()
        t1_values = vars1[parameter][0][0]
        t1_values = t1_values[:, :, np.newaxis]

        # Append values for both times along third dimension
        t_values = np.append(t0_values, t1_values, axis=2)

        # Copy 0th column to end for longitude 360
        first_column = np.expand_dims(t_values[:, 0, :], axis=1)
        t_values = np.concatenate((t_values, first_column), axis=1)

        # Flip along latitude axis, to match points
        t_values = np.flip(t_values, 0)
        return t_values

    def gh_gradient(self):
        """ Returns two 3D arrays of the same shape as gh_values, representing
        gradient in latitude and longitude directions with length in radians.
        """
        spacing_lat = np.radians(180 / (np.size(self.points[0]) - 1))  # radians
        spacing_lon = np.radians(360 / (np.size(self.points[1]) - 1))  # radians
        spacing_time = self.hours_between_files * 60 ** 2              # seconds

        gradient_lat, gradient_lon, gradient_time = np.gradient(self.gh_values, 
            spacing_lat, spacing_lon, spacing_time)
        return gradient_lat, gradient_lon

atmo = Atmosphere(0)
test = np.shape(atmo.gh_dlon)
print(test)
