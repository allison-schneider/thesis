import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
import scipy.interpolate
from mpl_toolkits.basemap import Basemap
import sys

# Global variables
EARTH_RADIUS = 6371e3    # meters
OMEGA = 7.292e-5         # radians per second

class Atmosphere:
    """ Contains atmospheric parameters (u speed, v speed, geopotential height
    gradient) for two time layers at a given time.

    hour -- The first layer time, a multiple of the interval between GFS files. 
    """
    def __init__(self,
                 hour):      # First layer time in hours

        self.layer_time = hour * 60 ** 2       # First layer time in seconds
        self.time_between_files = 3 * 60 ** 2  # Time between samples in seconds
        self.total_time = 240 * 60 ** 2        # Full trajectory time in seconds

        # Test that the argument is a valid hour.
        if self.layer_time > self.total_time - self.time_between_files:
            raise ValueError("Maximum value for hour argument is {}."
                           .format((self.total_time - self.time_between_files)
                                    / 60 ** 2))

        if self.layer_time % self.time_between_files != 0:
            raise ValueError("Hour argument must be a multiple of {}."
                             .format(self.time_between_files))

        # Generate list of all filenames and index of current file
        self.filenames = ['data/hgt-{:03d}.nc'.format(time) 
                for time in np.arange(0, 
                    np.int((self.total_time / (60 ** 2)) + 1), 
                    np.int(self.time_between_files / (60 ** 2)))]
        self.file_index = np.int(self.layer_time / self.time_between_files)                 
        
        # Initialize values for interpolate function
        self.points = self.build_points()
        self.u_values = self.values("u")
        self.v_values = self.values("v")
        self.gh_values = self.values("gh")
        self.dgh_dlat, self.dgh_dlon = self.gh_gradient()
        
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
        times = np.array([self.layer_time, self.layer_time 
                          + self.time_between_files])
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
        spacing_time = self.time_between_files                         # seconds

        gradient_lat, gradient_lon, gradient_time = np.gradient(self.gh_values, 
            spacing_lat, spacing_lon, spacing_time)
        return gradient_lat, gradient_lon

class Parcel:
    """ Represents the position and atmospheric parameters for a parcel of air.
    Latitude and longitude can be scalars, or, to represent multiple parcels,
    lists of the same length.
    """
    def __init__(self,
                 atmosphere,        # Instance of class Atmosphere
                 latitude,          # Latitude in degrees (-90, 90) 
                 longitude,         # Longitude in degrees (0, 360]) 
                 scheme="grid"):    # "grid" or "force"
        
        self.lat = np.radians(np.array(latitude))
        self.lon = np.radians(np.array(longitude))
        self.atmosphere = atmosphere
        self.scheme = scheme

        self.time = 0                   # seconds
        self.timestep = 180             # seconds  

        self.trajectory_lat = np.nan * np.zeros(
                                    (np.int(((self.atmosphere.total_time) 
                                    / self.timestep) + 1), np.size(self.lat)))
        self.trajectory_lon = np.nan * np.zeros(
                                    (np.int(((self.atmosphere.total_time) 
                                    / self.timestep) + 1), np.size(self.lon)))
        self.u = self.interpolate(self.atmosphere.u_values)
        self.v = self.interpolate(self.atmosphere.v_values)
        self.gh = self.interpolate(self.atmosphere.gh_values)

    def interpolate(self, interp_values):
        """ Linear interpolation of u, v, or gh between two time layers of a
        lat-lon grid. The interp_values parameter accepts u_values, g_values,
        or gh_values from the Atmosphere class.
        """
        xi_lat = np.degrees(self.lat)
        xi_lon = np.degrees(self.lon) % 360
        xi_times = np.full_like(self.lat, self.time)
        xi = np.array([xi_lat, xi_lon, xi_times]).T
        
        interp_result = scipy.interpolate.interpn(self.atmosphere.points,
                         interp_values, xi, bounds_error=False, fill_value=np.nan)
        return interp_result

    def calculate_trajectory(self):
        """ Calculate the trajectory of parcels.
        """   
        # Start trajectory at initial position 
        self.trajectory_lat[0,:] = self.lat
        self.trajectory_lon[0,:] = self.lon

        i = 1                   # Index for timestep
        layer_index = 0         # Index for instance of Atmosphere
        next_layer_hour = 0     # Argument for next instance of Atmosphere

        while next_layer_hour < self.atmosphere.total_time / 60 ** 2:
            self.atmosphere = Atmosphere(next_layer_hour)
            for layer_step in np.arange(self.atmosphere.time_between_files 
                                        / self.timestep):
                # Identify starting latitude and longitude
                initial_lat = self.lat 
                initial_lon = self.lon

                # Find u0 and v0 at starting latitude and longitude
                self.u = self.interpolate(self.atmosphere.u_values)
                self.v = self.interpolate(self.atmosphere.v_values)
                self.gh = self.interpolate(self.atmosphere.gh_values)
                initial_u = self.u 
                initial_v = self.v

                # Use u0 and v0 to get guess_lat and guess_lon
                dlat_dt = self.v / (EARTH_RADIUS + self.gh)
                dlon_dt = self.u / ((EARTH_RADIUS + self.gh) * np.cos(self.lat))
                guess_lat = self.lat + dlat_dt * self.timestep
                guess_lon = self.lon + dlon_dt * self.timestep
                self.lat = guess_lat
                self.lon = guess_lon

                # Find guess_u and guess_v at guess position after one timestep
                self.time += self.timestep
                self.u = self.interpolate(self.atmosphere.u_values)
                self.v = self.interpolate(self.atmosphere.v_values)

                # Average initial and guess velocities
                self.u = (initial_u + self.u) / 2
                self.v = (initial_v + self.v) / 2

                # Use the timestep and u and v to get next trajectory position
                dlat_dt = self.v / (EARTH_RADIUS + self.gh)
                dlon_dt = self.u / ((EARTH_RADIUS + self.gh) 
                                    * np.cos(initial_lat))
                self.lat = initial_lat + dlat_dt * self.timestep
                self.lon = initial_lon + dlon_dt * self.timestep

                # Store position in trajectory array
                self.trajectory_lat[i,:] = self.lat
                self.trajectory_lon[i,:] = self.lon

                # Increment timestep index
                i += 1

            # Get new instance of Atmosphere for next time layer
            layer_index += 1
            next_layer_hour = layer_index * (self.atmosphere.time_between_files 
                                             / (60 ** 2)) 
            
        return self.trajectory_lat, self.trajectory_lon

class Trajectory:
    """ Lists of positions for each timestep along the trajectory. Contains 
    functions for multiple trajectory analysis and plotting. """
    def __init__(self,
                 atmosphere,    # Instance of class Atmosphere
                 parcel):       # Instance of class Parcel

        self.parcel = parcel
        self.atmosphere = atmosphere
        self.latitudes, self.longitudes = np.degrees(self.parcel.calculate_trajectory())

        # Remove NaNs from arrays
        self.latitudes = self.latitudes[np.isfinite(
                                               self.latitudes[:,0]),:]
        self.longitudes = self.longitudes[np.isfinite(
                                               self.longitudes[:,0]),:]

    def plot_ortho(self, lat_center=90, lon_center=-105, savefig=False):
        """ Orthographic projection plot."""
        map = Basemap(projection='ortho', lon_0=lon_center, lat_0=lat_center, 
                        resolution='c')
        map.drawcoastlines(linewidth=0.25, color='gray')
        map.drawcountries(linewidth=0)
        map.fillcontinents(color='white',lake_color='white', zorder=1)
        # draw the edge of the map projection region (the projection limb)
        map.drawmapboundary(fill_color='white')
        map.plot(self.longitudes, self.latitudes,
                 latlon=True, zorder=2, color='black')
        if savefig == True:
            filename = "trajectory_"+sys.argv[1]+"_"+sys.argv[2]+".eps"
            plt.savefig(filename)

        plt.show()
        return map

atmo = Atmosphere(0)
p = Parcel(atmo, [41, 42], 
                 [-71, -72])
tra = Trajectory(atmo, p)

tra.plot_ortho()
