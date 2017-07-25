import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
import scipy.interpolate
from mpl_toolkits.basemap import Basemap
import sys

# Global variables
EARTH_RADIUS = 6371e3    # meters

class Atmosphere:
    """ Contains atmospheric parameters (u speed, v speed, geopotential height)
    for two time layers at a given time.
    """
    def __init__(self,
                 time):     # Current time in hours from start

        self.time = time
        self.time_between_files = 3.0    # Time in hours between samples
        self.timestep = 3.0              # Timestep in hours
        self.total_time = 240.0          # Time to run trajectory in hours
        self.num_files = 81
        
        # Generate list of all filenames and index of current file
        self.filenames = ['data/hgt-{:03d}.nc'.format(time) 
                            for time in np.arange(0,241,3)]
        self.file_index = np.int(np.floor(self.time / self.time_between_files))                 
        
        # Initialize values for interpolate function
        self.points = self.build_points()
        self.u_values = self.values("u")
        self.v_values = self.values("v")
        self.gh_values = self.values("gh")
        
    def build_points(self):
        """ Returns the first argument for the interpn function,
        a tuple of 3 ndarrays of float, named points,
        of shape ((nlats,), (nlons,), (ntimes)),
        which stays constant throughout the trajectory."""
        file = netcdf.netcdf_file(self.filenames[self.file_index], mmap=False)
        vars = file.variables
        file.close()

        # Flip latitudes so they're increasing, (-90, 90).
        latitudes = np.flip(vars['lat'][:], 0)   
        # Add the longitude 360 to deal with interpolation near 0 degrees.
        longitudes = np.append(vars['lon'][:], 360)
       
        # Array of times has last and next sample times, or current and next.
        last_time = self.time
        times = np.array([last_time, last_time + self.time_between_files])
        
        points = latitudes, longitudes, times
        return points

    def values(self, parameter):
        """ Returns the second argument for the interpn function,
        a 3D ndarray of float, named values, of shape (nlats, nlons, ntimes),
        which is updated when the trajectory passes a new time layer.
        Arguments:
        parameter -- 'u', 'v', or 'gh'; the atmospheric parameter 
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
        
        self.lat = np.array(latitude)
        self.lon = np.array(longitude) % 360    # Convert longitude to (0, 360]
        self.atmosphere = atmosphere
        self.scheme = scheme  

        self.trajectory_lat = np.nan * np.zeros(
                    (np.int((self.atmosphere.total_time) / 
                     self.atmosphere.timestep), np.size(self.lat)))
        self.trajectory_lon = np.nan * np.zeros(
                    (np.int((self.atmosphere.total_time) / 
                     self.atmosphere.timestep), np.size(self.lon)))

    def spherical_hypotenuse(self, a, b):
        """ Given the lengths of two sides of a right triangle on a sphere, 
        a and b, find the length of the hypotenuse c. 
        
        Arguments:
        a -- Length of first side of the triangle, in meters
        b -- Length of second side of the triangle, in meters 
        """
        c = EARTH_RADIUS * np.arccos(np.cos(a / EARTH_RADIUS) 
            * np.cos(b / EARTH_RADIUS))
        return c

    def destination(self, distance, bearing):
        """ Return the latitude and longitude of a destination point 
        given a starting latitude and longitude, distance, and bearing.
        
        Arguments:
        lat1 -- Starting latitude in degrees, -90 to 90
        lon1 -- Starting longitude in degrees, 0 to 360
        distance -- Distance to travel in meters
        bearing -- Direction between 0 and 360 degrees, 
                    clockwise from true North.
        """
        angular_distance = distance / EARTH_RADIUS
        lat2 = np.degrees(np.arcsin(np.sin(np.radians(self.lat)) 
            * np.cos(angular_distance) + np.cos(np.radians(self.lat)) 
            * np.sin(angular_distance) * np.cos(np.radians(bearing))))
        # Longitude is mod 360 degrees for wrapping around earth
        lon2 = (self.lon + np.degrees(np.arctan2(np.sin(np.radians(bearing)) 
            * np.sin(angular_distance) * np.cos(np.radians(self.lat)), 
            np.cos(angular_distance) - np.sin(np.radians(self.lat)) 
            * np.sin(np.radians(lat2))))) % 360   
        return lat2, lon2

    def compass_bearing(self, math_bearing):
        """ Transform a vector angle to a compass bearing."""
        bearing = (5 * np.pi / 2 - math_bearing) % (2 * np.pi)
        return bearing

    def interpolate(self, interp_values):
        """ Linear interpolation of u, v, or gh between two time layers of a
        lat-lon grid. The interp_values parameter accepts u_values, g_values,
        or gh_values from the Atmosphere class.
        """

        xi_times = np.full_like(self.lat, self.atmosphere.time)
        xi = np.array([self.lat, self.lon, xi_times]).T
        
        interp_result = scipy.interpolate.interpn(self.atmosphere.points,
                                                  interp_values, xi,
                                                  bounds_error=False,
                                                  fill_value=None)

        return interp_result

    def velocity_components(self):
        """ Gets the u and v components of wind velocity at a point 
        according to the given calculation scheme. 

        Schemes:
        grid -- calculate from wind field
        force -- calculate advection from geostropic wind equation using 
                 geopotential height
        """
        if self.scheme == "grid":
            # Interpolate u and v at given position
            u_speed = self.interpolate(atmo.u_values)
            v_speed = self.interpolate(atmo.v_values)

        ## To do: implement force method    
        #elif scheme == "force":
        #    u_speed, v_speed = speed_force(grid_lat, grid_lon, filename)

        else:
            print("Invalid calculation scheme.")

        return u_speed, v_speed

    def next_position(self):
        """ This gets the next position in a trajectory according to the given 
            calculation scheme ("grid" or "force"), by multiplying velocity with 
            the timestep. 
        """
        u_speed, v_speed = self.velocity_components()

        # Get magnitude and direction of wind vector
        wind_speed = self.spherical_hypotenuse(u_speed, v_speed)
        wind_direction = np.arctan2(v_speed, u_speed)
        wind_vector = np.array([wind_speed, wind_direction])

        # Get displacement using velocity times delta t
        delta_t = self.atmosphere.timestep * 60 ** 2    # in seconds
        displacement = wind_speed * delta_t             # in meters

        # Calculate new latitude and longitude
        # Convert wind bearing to degrees
        wind_bearing = np.degrees(self.compass_bearing(wind_direction))
        new_lat, new_lon = self.destination(displacement, 
                                            wind_bearing)
        return new_lat, new_lon

    def runge_kutta_trajectory(self):
        """ Uses a second order Runge-Kutta method to get the position at the 
        next timestep.
        """        
        i = 1   # Index for timestep
 
        # Start trajectory at initial position 
        self.trajectory_lat[0,:] = self.lat
        self.trajectory_lon[0,:] = self.lon
        
        while self.atmosphere.file_index < self.atmosphere.num_files - 2:

            # Runge-Kutta approximation between two time layers
            for layer_step in np.arange(self.atmosphere.time_between_files 
                                        / self.atmosphere.timestep):
                                
                # First guess position and velocity at starting point
                guess_lat, guess_lon = self.next_position()
                initial_u, initial_v = self.velocity_components()
                
                # Velocity at first guess point at next timestep
                self.atmosphere.time += self.atmosphere.timestep
                guess_u, guess_v = self.velocity_components()
                
                # Average 
                average_u = 0.5 * (initial_u + guess_u)
                average_v = 0.5 * (initial_v + guess_v)

                wind_speed = self.spherical_hypotenuse(average_u, average_v)
                wind_direction = np.arctan2(average_v, average_u)
                wind_bearing = np.degrees(self.compass_bearing(wind_direction))
                displacement = wind_speed * self.atmosphere.timestep * 60 ** 2   

                i = np.int(layer_step + self.atmosphere.file_index 
                                        * (self.atmosphere.time_between_files
                                           / self.atmosphere.timestep))
                self.trajectory_lat[i,:], self.trajectory_lon[i,:] = (
                                self.destination(displacement, wind_bearing))
                self.lat = self.trajectory_lat[i,:]
                self.lon = self.trajectory_lon[i,:] % 360
 
            # Get new instance of Atmosphere for next time layer
            self.atmosphere.time = np.round(((self.atmosphere.file_index + 1) 
                                   * self.atmosphere.time_between_files), 
                                    decimals=4)
            self.atmosphere = Atmosphere(self.atmosphere.time)

        return self.trajectory_lat, self.trajectory_lon

class Trajectory:
    """ Lists of positions for each timestep along the trajectory."""
    def __init__(self,
                 atmosphere,    # Instance of class Atmosphere
                 parcel):       # Instance of class Parcel

        self.parcel = parcel
        self.atmosphere = atmosphere
        self.latitudes, self.longitudes = self.parcel.runge_kutta_trajectory()

        # Remove NaNs from arrays
        self.latitudes = self.latitudes[np.isfinite(
                                               self.latitudes[:,0]),:]
        self.longitudes = self.longitudes[np.isfinite(
                                               self.longitudes[:,0]),:]

        self.mean_latitudes, self.mean_longitudes = self.mean_trajectory()

        print(self.mean_longitudes)

    def haversine(self):
        """ Great-circle distance between two points. """
        lat1 = np.radians(self.latitudes[:,0])
        lat2 = np.radians(self.latitudes[:,1])
        lon1 = np.radians(self.longitudes[:,0])
        lon2 = np.radians(self.longitudes[:,1])
        dlat = np.absolute(lat2 - lat1)
        dlon = np.absolute(lon2 - lon1)

        a = (np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) 
            * np.sin(dlon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) 
        d = EARTH_RADIUS * c    # distance in meters
        return d

    def mean_trajectory(self):
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
        map.plot(self.mean_longitudes, self.mean_latitudes,
                 latlon=True, zorder=2, color='blue')
        if savefig == True:
            filename = "trajectory_"+sys.argv[1]+"_"+sys.argv[2]+".eps"
            plt.savefig(filename)

        plt.show()
        return map
        
    def plot_cyl(self):
        """ Equidistant cylindrical plot. """
        
        # Convert longitudes to (-180, 180)
        self.lon_180 = self.longitudes - (360 * (self.longitudes >= 180))

        map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,
            llcrnrlon=-180,urcrnrlon=180,resolution='c')
        map.drawcoastlines(linewidth=0.25, color='gray')
        map.drawcountries(linewidth=0)
        map.fillcontinents(color='white',lake_color='white', zorder=1)
        map.drawparallels(np.arange(-90.,91.,30.))
        map.drawmeridians(np.arange(-180.,181.,60.))
        map.drawmapboundary(fill_color='white')

        # Convert latitudes and longitudes to map projection coordinates
        xpt, ypt = map(self.lon_180, self.latitudes)

        map.plot(xpt, ypt,
                 latlon=False, zorder=2, color='black')
        return map

atmo = Atmosphere(0)
p = Parcel(atmo, [41, 41, 51, 51], 
                 [-41, -51, -41, -51])
tra = Trajectory(atmo, p)

tra.plot_ortho()
