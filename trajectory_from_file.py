import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Global variables
EARTH_RADIUS = 6371e3    # meters

class Trajectory:
    """ Version of trajectory class for analyzing trajectory data from text files."""
    
    def __init__(self):       # Instance of class Parcel

        self.latitudes = np.loadtxt("trajectory_data/latitudes.txt")
        self.longitudes = np.loadtxt("trajectory_data/longitudes.txt")
        self.trajectory_u = np.loadtxt("trajectory_data/trajectory_u.txt")
        self.trajectory_v = np.loadtxt("trajectory_data/trajectory_v.txt")

        self.timestep = 180     # seconds
        # List of times for plotting
        self.times = np.arange(np.size(self.latitudes[:,0])) * (self.timestep 
                                                          / 60 ** 2)

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
        map.plot(self.mean_longitudes, self.mean_latitudes,
                 latlon=True, zorder=2, color='blue')
        if savefig == True:
            filename = "plots/inertial.png"
            plt.savefig(filename)

        plt.show()
        return map

    def graph_speed(self):
        """ Graph of u and v along the trajectory. """

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        u_line, = ax1.plot(self.times, self.trajectory_u[:,0], label="u")
        v_line, = ax1.plot(self.times, self.trajectory_v[:,0], label="v")
        ax1.legend()
        ax1.set_title("Force Trajectory Speeds")
        ax1.set_xlabel("Time (hours)")
        ax1.set_ylabel("Velocity (m/s)")
        #plt.savefig("plots/force_trajectory_speeds.png")
        plt.show()
        return ax1

    def graph_rms(self):
        """ Graph of rms distance from mean trajectory. """
        rms = self.rms_distance()

        fig = plt.figure()
        ax2 = fig.add_subplot(1, 1, 1)

        rms_line, = ax2.plot(self.times, rms)
        ax2.set_title("RMS Distance from Mean Trajectory")
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("RMS distance (m)")
        plt.show()
        return ax2

tra = Trajectory()
tra.graph_rms()