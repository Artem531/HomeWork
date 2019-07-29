import numpy as np


class Configuration:
    """
    Store fields with boundaries of 'zone of interest'
    """

    def __init__(self, config):
        """
        :param config: a Python dictionary of hyperparameter name-value pairs
        """
        #self.min_x = config["geometry"][2]
        #self.max_x = config["geometry"][3]
        #self.min_y = config["geometry"][0]
        #self.max_y = config["geometry"][1]
        #self.min_z = config["geometry"][4]
        #self.max_z = config["geometry"][5]

        self.min_x = config["geometry"]["length_min"]
        self.max_x = config["geometry"]["length_max"]
        self.min_y = config["geometry"]["width_min"]
        self.max_y = config["geometry"]["width_max"]
        self.min_z = config["geometry"]["height_min"]
        self.max_z = config["geometry"]["height_max"]


    def check_point(self, point):
        """ checks position a point is in the zone of interest
        :param point: tuple or list with coordinates (x, y, z) or (x, y)
        :return: True if point in zone, else: False
        """
        return self.min_x <= point[0] <= self.max_x \
               and self.min_y <= point[1] <= self.max_y

    def check_objinfo(self, label):
        """
        checks position label is in the zone of interest
        :param label: ObjectInfo object
        :return: True if label  in zone, else: False
        """
        bbox = label.bbox
        return np.amax(bbox[:, 0]) <= self.max_x \
               and np.amin(bbox[:, 0]) >= self.min_x \
               and np.amax(bbox[:, 1]) <= self.max_y \
               and np.amin(bbox[:, 1]) >= self.min_y
