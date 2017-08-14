import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
import scipy.interpolate
from mpl_toolkits.basemap import Basemap
import sys

# Global variables
EARTH_RADIUS = 6371e3    # meters
OMEGA = 7.292e-5         # radians per second
STANDARD_GRAVITY = 9.806 # meters per second squared

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
        self.dgh_dlat_values, self.dgh_dlon_values = self.gh_gradient()
        
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
                 scheme,            # "grid" or "force"
                 timestep):         # Timestep in seconds  
        
        self.lat = np.radians(np.array(latitude))
        self.lon = np.radians(np.array(longitude))
        self.atmosphere = atmosphere
        self.scheme = scheme
        self.timestep = timestep        # seconds 

        self.time = 0                   # seconds 

        self.trajectory_lat = np.nan * np.zeros(
                                    (np.int(((self.atmosphere.total_time) 
                                    / self.timestep) + 1), np.size(self.lat)))
        self.trajectory_lon = np.full_like(self.trajectory_lat, np.nan)

        self.trajectory_u = np.full_like(self.trajectory_lat, np.nan)
        self.trajectory_v = np.full_like(self.trajectory_lat, np.nan)
        
        self.gh = self.interpolate(self.atmosphere.gh_values)
        
        if self.scheme == "grid":
            self.u = self.interpolate(self.atmosphere.u_values)
            self.v = self.interpolate(self.atmosphere.v_values)
            
        if self.scheme == "force" or "friction":
            f = 2 * OMEGA * np.sin(self.lat)    # radians per second
            dgh_dlat = self.interpolate(self.atmosphere.dgh_dlat_values)
            dgh_dlon = self.interpolate(self.atmosphere.dgh_dlon_values)
            # Geostrophic u and v in meters per second
            self.u = ((-STANDARD_GRAVITY / f) * dgh_dlat) / EARTH_RADIUS
            self.v = ((STANDARD_GRAVITY / f) * dgh_dlon) / (EARTH_RADIUS 
                                                            * np.cos(self.lat))                 

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

        # Start u and v at initial u and v
        self.trajectory_u[0,:] = self.u
        self.trajectory_v[0,:] = self.v        

        i = 1                   # Index for timestep
        layer_index = 0         # Index for instance of Atmosphere
        next_layer_hour = 0     # Argument for next instance of Atmosphere

        if self.scheme == "grid":
            while next_layer_hour < self.atmosphere.total_time / 60 ** 2:
                self.atmosphere = Atmosphere(next_layer_hour)
                for layer_step in np.arange(self.atmosphere.time_between_files 
                                            / self.timestep):
                    # Identify starting latitude and longitude
                    initial_lat = self.lat 
                    initial_lon = self.lon

                    # Find u0 and v0 at starting latitude and longitude
                    initial_u = self.interpolate(self.atmosphere.u_values)
                    initial_v = self.interpolate(self.atmosphere.v_values)
                    self.gh = self.interpolate(self.atmosphere.gh_values)

                    # Use u0 and v0 to get guess latitude and longitude
                    dlat_dt = initial_v / (EARTH_RADIUS + self.gh)
                    dlon_dt = initial_u / ((EARTH_RADIUS + self.gh) 
                                            * np.cos(self.lat))
                    self.lat = initial_lat + dlat_dt * self.timestep
                    self.lon = initial_lon + dlon_dt * self.timestep

                    # Find guess_u and guess_v at guess position after timestep
                    self.time += self.timestep
                    guess_u = self.interpolate(self.atmosphere.u_values)
                    guess_v = self.interpolate(self.atmosphere.v_values)

                    # Average initial and guess velocities
                    self.u = (initial_u + guess_u) / 2
                    self.v = (initial_v + guess_v) / 2

                    # Use timestep and u and v to get next trajectory position
                    dlat_dt = self.v / (EARTH_RADIUS + self.gh)
                    dlon_dt = self.u / ((EARTH_RADIUS + self.gh) 
                                        * np.cos(initial_lat))
                    self.lat = initial_lat + dlat_dt * self.timestep
                    self.lon = initial_lon + dlon_dt * self.timestep

                    # Store position in trajectory array
                    self.trajectory_lat[i,:] = self.lat
                    self.trajectory_lon[i,:] = self.lon
                    self.trajectory_u[i,:] = self.u
                    self.trajectory_v[i,:] = self.v

                    # Increment timestep index
                    i += 1

                # Get new instance of Atmosphere for next time layer
                layer_index += 1
                next_layer_hour = layer_index * (
                                self.atmosphere.time_between_files / (60 ** 2)) 

        elif self.scheme == "force":
            while next_layer_hour < self.atmosphere.total_time / 60 ** 2:
                self.atmosphere = Atmosphere(next_layer_hour)
                for layer_step in np.arange(self.atmosphere.time_between_files 
                                            / self.timestep):

                    # Get gradient of geopotential height at initial position
                    self.gh = self.interpolate(self.atmosphere.gh_values)
                    dgh_dlat = self.interpolate(self.atmosphere.dgh_dlat_values)
                    dgh_dlon = self.interpolate(self.atmosphere.dgh_dlon_values)

                    # Find u0 and v0 at starting latitude and longitude
                    initial_u = self.u
                    initial_v = self.v

                    # Find guess_u and guess_v at guess position after timestep
                    du_dt = ((2 * OMEGA * np.sin(self.lat) * initial_v)
                            - (1 / ((EARTH_RADIUS + self.gh) 
                                * np.cos(self.lat))) 
                                * STANDARD_GRAVITY * dgh_dlon)
                    dv_dt = ((-2 * OMEGA * np.sin(self.lat) * initial_u)
                            - (1 / (EARTH_RADIUS + self.gh)) 
                            * STANDARD_GRAVITY * dgh_dlat)
                    guess_u = initial_u + du_dt * self.timestep
                    guess_v = initial_v + dv_dt * self.timestep

                    # Average initial and guess velocities
                    self.u = (initial_u + guess_u) / 2
                    self.v = (initial_v + guess_v) / 2

                    # Use timestep and u and v to get next trajectory position
                    dlat_dt = self.v / (EARTH_RADIUS + self.gh)
                    dlon_dt = self.u / ((EARTH_RADIUS + self.gh) 
                                        * np.cos(self.lat))
                    self.lat = self.lat + dlat_dt * self.timestep
                    self.lon = self.lon + dlon_dt * self.timestep

                    # Store position in trajectory array
                    self.trajectory_lat[i,:] = self.lat
                    self.trajectory_lon[i,:] = self.lon
                    self.trajectory_u[i,:] = self.u
                    self.trajectory_v[i,:] = self.v

                    # Increment timestep and timestep index
                    self.time += self.timestep
                    i += 1

                # Get new instance of Atmosphere for next time layer
                layer_index += 1
                next_layer_hour = layer_index * (
                                self.atmosphere.time_between_files / (60 ** 2))

        # Same as "force" scheme, but with a friction term to damp oscillations
        elif self.scheme == "friction":
            while next_layer_hour < self.atmosphere.total_time / 60 ** 2:
                self.atmosphere = Atmosphere(next_layer_hour)
                for layer_step in np.arange(self.atmosphere.time_between_files 
                                            / self.timestep):

                    # Get gradient of geopotential height at initial position
                    self.gh = self.interpolate(self.atmosphere.gh_values)
                    dgh_dlat = self.interpolate(self.atmosphere.dgh_dlat_values)
                    dgh_dlon = self.interpolate(self.atmosphere.dgh_dlon_values)

                    # Find u0 and v0 at starting latitude and longitude
                    initial_u = self.u
                    initial_v = self.v

                    # Get geostrophic u and v
                    f = 2 * OMEGA * np.sin(self.lat)
                    u_g = ((-STANDARD_GRAVITY / f) * dgh_dlat * (1 / 
                        (EARTH_RADIUS + self.gh)))
                    v_g = ((STANDARD_GRAVITY / f) * dgh_dlon * (1 / 
                        ((EARTH_RADIUS + self.gh) * np.cos(self.lat))))

                    # Get friction terms
                    r_f = 10e-6
                    friction_u = - r_f * (initial_u - u_g)
                    friction_v = - r_f * (initial_v - v_g)

                    # Find guess_u and guess_v at guess position after timestep
                    du_dt = ((f * initial_v)
                            - (1 / ((EARTH_RADIUS + self.gh) 
                                * np.cos(self.lat))) 
                                * STANDARD_GRAVITY * dgh_dlon + friction_u)
                    dv_dt = ((-f * initial_u)
                            - (1 / (EARTH_RADIUS + self.gh)) 
                            * STANDARD_GRAVITY * dgh_dlat + friction_v)
                    guess_u = initial_u + du_dt * self.timestep
                    guess_v = initial_v + dv_dt * self.timestep

                    # Average initial and guess velocities
                    self.u = (initial_u + guess_u) / 2
                    self.v = (initial_v + guess_v) / 2

                    # Use timestep and u and v to get next trajectory position
                    dlat_dt = self.v / (EARTH_RADIUS + self.gh)
                    dlon_dt = self.u / ((EARTH_RADIUS + self.gh) 
                                        * np.cos(self.lat))
                    self.lat = self.lat + dlat_dt * self.timestep
                    self.lon = self.lon + dlon_dt * self.timestep

                    # Replace latitudes above pi/2 or below -pi/2
                    np.place(self.lat, self.lat > np.pi / 2, np.pi / 2)
                    np.place(self.lat, self.lat < -np.pi / 2, -np.pi / 2)

                    # Store position in trajectory array
                    self.trajectory_lat[i,:] = self.lat
                    self.trajectory_lon[i,:] = self.lon
                    self.trajectory_u[i,:] = self.u
                    self.trajectory_v[i,:] = self.v

                    # Increment timestep and timestep index
                    self.time += self.timestep
                    i += 1

                # Get new instance of Atmosphere for next time layer
                layer_index += 1
                next_layer_hour = layer_index * (
                                self.atmosphere.time_between_files / (60 ** 2))
        else:
            raise ValueError("Invalid scheme. Try 'grid' or 'force'.")

        return self.trajectory_lat, self.trajectory_lon

class Trajectory:
    """ Lists of positions for each timestep along the trajectory. Contains 
    functions for multiple trajectory analysis and plotting. """
    def __init__(self,
                 atmosphere,    # Instance of class Atmosphere
                 parcel):       # Instance of class Parcel

        self.parcel = parcel
        self.atmosphere = atmosphere
        self.latitudes, self.longitudes = np.degrees(
                                             self.parcel.calculate_trajectory())

        # Remove NaNs from arrays
        self.latitudes = self.latitudes[np.isfinite(
                                               self.latitudes[:,0]),:]
        self.longitudes = self.longitudes[np.isfinite(
                                               self.longitudes[:,0]),:]
        self.trajectory_u = self.parcel.trajectory_u[np.isfinite(
                                               self.latitudes[:,0]),:]
        self.trajectory_v = self.parcel.trajectory_u[np.isfinite(
                                               self.latitudes[:,0]),:]

        self.mean_latitudes, self.mean_longitudes = self.mean_trajectory()

    def haversine(self, latitude1, longitude1, latitude2, longitude2):
        """ Great-circle distance between two points. Latitudes and longitudes
        are 1D NumPy arrays. 
        Returns a 1D array of distances between two trajectories. """

        lat1 = np.radians(latitude1)
        lat2 = np.radians(latitude2)
        lon1 = np.radians(longitude1)
        lon2 = np.radians(longitude2)

        dlat = np.absolute(lat2 - lat1)
        dlon = np.absolute(lon2 - lon1)

        a = (np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) 
            * np.sin(dlon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) 
        d = EARTH_RADIUS * c    # distance in meters
        return d

    def mean_trajectory(self):
        """ Get the centroid of parcels at each timestep. """

        # Convert latitudes and longitudes to Cartesian coordinates
        x = (np.cos(np.radians(self.latitudes)) * 
            np.cos(np.radians(self.longitudes)))
        y = (np.cos(np.radians(self.latitudes)) * 
            np.sin(np.radians(self.longitudes)))
        z = np.sin(np.radians(self.latitudes))

        # Get average x, y, z values
        mean_x = np.mean(x, axis=1)
        mean_y = np.mean(y, axis=1)
        mean_z = np.mean(z, axis=1)

        # Convert average values to trajectory latitudes and longitudes
        mean_longitudes = np.degrees(np.arctan2(mean_y, mean_x))
        hypotenuse = np.sqrt(mean_x ** 2 + mean_y ** 2)
        mean_latitudes = np.degrees(np.arctan2(mean_z, hypotenuse))

        return mean_latitudes, mean_longitudes

    def rms_distance(self):
        """ Calculate the root mean square distance of each trajectory from the
        mean trajectory. """

        # Make mean lat and lon arrays the same shape as trajectory arrays
        mean_lat = np.repeat(tra.mean_latitudes[:, np.newaxis], 
                             np.size(self.latitudes, axis=1), axis=1)
        mean_lon = np.repeat(tra.mean_longitudes[:, np.newaxis], 
                             np.size(self.longitudes, axis=1), axis=1)

        rms = np.sqrt(np.mean(self.haversine(mean_lat, mean_lon, 
                              self.latitudes, self.longitudes) ** 2, axis=1))

        return rms

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
        plt.title("Kinematic Trajectories \n"
                  "3 minute timestep")
        map.plot(self.mean_longitudes, self.mean_latitudes,
                 latlon=True, zorder=2, color='blue')
        if savefig == True:
            filename1 = "plots/test.png"
            filename2 = "plots/test.svg"
            plt.savefig(filename1)
            plt.savefig(filename2)

        plt.show()
        return map

    def graph(self):
        """ Graph of u and v along the trajectory. """
        lat_length = 111.32e3       # Length of a degree of latitude in meters
        time = np.arange(np.size(self.latitudes[:,0])) * (self.parcel.timestep 
                                                           / (60 ** 2 * 24))
        v_threshold = ((0.75 * 0.25 * lat_length) / self.parcel.timestep 
                        * np.ones(np.size(time)))
        v_threshold = v_threshold[:, np.newaxis]
        v_diff = v_threshold - self.parcel.trajectory_v

        u_threshold = (0.75 * 0.25 * lat_length 
            * np.cos(np.radians(self.latitudes))) / self.parcel.timestep
        u_diff = u_threshold - self.parcel.trajectory_u

        zero = np.zeros(np.size(time))
        
        # new style method 1; unpack the axes
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
        u_line = ax1.plot(time, u_diff, color="black")
        u_zero_line = ax1.plot(time, zero, color="black", linestyle="--")
        ax1.set_title("Dynamic Trajectory Zonal Speeds")

        v_line = ax2.plot(time, v_diff, color="black")
        v_zero_line = ax2.plot(time, zero, color="black", linestyle="--")
        ax2.set_title("Dynamic Trajectory Meridional Speeds")

        plt.xlabel("Time in days")
        plt.ylabel("Velocity in m/s")
        plt.savefig("plots/test.png")
        plt.show()
        return ax1

    def save_data(self):
        header_string = (
            "Calculation scheme is {0}.\n"
            "Timestep is {1} seconds.\n"
            "Trajectories calculated for a 5 x 5 grid of parcels between "
            "41, -72 and 42, -71.\n"
            "Number of parcels is {2}.".format(self.parcel.scheme, 
                        self.parcel.timestep, np.size(self.latitudes[0,:])))
        lat_title = ("trajectory_data/"
                    "test_latitudes_{0}_{1}.txt".format(self.parcel.scheme,
                                                self.parcel.timestep))
        lon_title = ("trajectory_data/"
                    "test_longitudes_{0}_{1}.txt".format(self.parcel.scheme,
                                                self.parcel.timestep))
        u_title = ("trajectory_data/"
                    "test_trajectory_u_{0}_{1}.txt".format(self.parcel.scheme,
                                                self.parcel.timestep))
        v_title = ("trajectory_data/"
                    "test_trajectory_v_{0}_{1}.txt".format(self.parcel.scheme,
                                                self.parcel.timestep))
        np.savetxt(lat_title, self.latitudes, header=header_string)
        np.savetxt(lon_title, self.longitudes, header=header_string)
        np.savetxt(u_title, self.trajectory_u, header=header_string)
        np.savetxt(v_title, self.trajectory_v, header=header_string)

# Create grid of latitudes and longitudes to launch parcels from
num_lats = 5  
num_lons = 5
first_lat = 41
last_lat = 42
first_lon = -72
last_lon = -71

latitudes = np.linspace(first_lat, last_lat, num=num_lats)
longitudes = np.linspace(first_lon, last_lon, num=num_lons)
grid_lon, grid_lat = np.meshgrid(longitudes, latitudes)
lon = np.ndarray.flatten(grid_lon)
lat = np.ndarray.flatten(grid_lat)

# Perform the trajectory calculation
scheme_arg = sys.argv[1]
timestep_arg = int(sys.argv[2])

atmo = Atmosphere(0)
p = Parcel(atmo, lat,
                 lon, 
                 scheme_arg,
                 timestep_arg)
tra = Trajectory(atmo, p)

# Save data to text files
#tra.save_data()
