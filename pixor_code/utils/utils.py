import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import os.path
from shapely import geometry


def iou_bboxes(obj1, obj2):
    def preproc_point(obj):
        points_list = obj.bbox[obj.bbox[:, 2] == np.min(obj.bbox[:, 2])]
        points_list[[2, 3]] = points_list[[3, 2]]
        points_list1 = [tuple(idx) for idx in points_list]
        return points_list1


    points_list1 = preproc_point(obj1)
    points_list2 = preproc_point(obj2)
    polygon_pair = (geometry.Polygon(points_list1),
                    geometry.Polygon(points_list2))
    if False not in list(map(lambda x: x.is_valid, polygon_pair)):
        union = polygon_pair[0].union(polygon_pair[1]).area
        res = polygon_pair[0].intersection(polygon_pair[1]).area / union
        return res
    else:
        raise ValueError("invalid Polygon")


def trasform_label2metric(label, ratio=None, voxel_size=None, base_height=100):
    '''
    :param label: numpy array of shape [..., 2] of coordinates in label map space
    :return: numpy array of shape [..., 2] of the same coordinates in metric space
    '''

    metric = np.copy(label)
    metric[..., 1] -= base_height
    metric = metric * voxel_size * ratio

    return metric


def transform_metric2label(metric, ratio=None, voxel_size=None, base_height=100):
    '''
    :param label: numpy array of shape [..., 2] of coordinates in metric space
    :return: numpy array of shape [..., 2] of the same coordinates in label_map space
    '''

    label = (metric / ratio ) / voxel_size
    label[..., 1] += base_height
    return label


def get_bev_image(velo_array, voxel_size, label_list = None, gt_label_list = None, map_height = None):
    '''
    Plot a Birds Eye View Lidar and Bounding boxes (Using OpenCV!)
    The heading of the vehicle is marked as a red line
        (which connects front right and front left corner)

    :param velo_array: a 2d velodyne points
    :param label_list: a list of numpy arrays of shape [4, 2], which corresponds to the 4 corners' (x, y)
    The corners should be in the following sequence:
    rear left, rear right, front right and front left
    :param map_height: height of the map
    :param window_name: name of the open_cv2 window
    :return: None
    '''
    intensity = np.zeros((velo_array.shape[0], velo_array.shape[1], 3), dtype = np.uint8)
    #print("intensity shape at plotting", intensity.shape)
    # val = 1 - velo_array[::-1, :, -1]
    val = ((1 - velo_array[::-1, :, :-1].max(axis=2))*255).astype(dtype = np.uint8)
    val[val == 255] = 128
    intensity[:, :, 0] = val
    intensity[:, :, 1] = val
    intensity[:, :, 2] = val
    # FLip in the x direction

    if gt_label_list is not None:
        for corners in gt_label_list:
            plot_corners = corners/voxel_size
            plot_corners[:, 1] += int(map_height//2)
            plot_corners[:, 1] = map_height - plot_corners[:, 1]
            if  not isinstance(plot_corners, np.ndarray):
                plot_corners = plot_corners.cpu().numpy()
            plot_corners = plot_corners.astype(int).reshape((-1, 1, 2))
            cv2.polylines(intensity, [plot_corners], True, (0, 255, 0), 3)
            #cv2.line(intensity, tuple(plot_corners[2, 0]), tuple(plot_corners[3, 0]), (0, 0, 255), 3)

    if label_list is not None:
        for corners in label_list:
            plot_corners = corners/voxel_size
            plot_corners[:, 1] += int(map_height//2)
            plot_corners[:, 1] = map_height - plot_corners[:, 1]
            if  not isinstance(plot_corners, np.ndarray):
                plot_corners = plot_corners.cpu().numpy()
            plot_corners = plot_corners.astype(int).reshape((-1, 1, 2))
            cv2.polylines(intensity, [plot_corners], True, (0, 128, 256), 2)
            #cv2.line(intensity, tuple(plot_corners[2, 0]), tuple(plot_corners[3, 0]), (0, 0, 255), 3)

    return intensity


def plot_label_map(label_map):
    # print("label map shape", label_map.shape)
    plt.figure()
    plt.imshow(label_map[::-1, :])
    plt.show()


def get_points_in_a_rotated_box(corners, labels_shape):
    def minY(x0, y0, x1, y1, x):
        if x0 == x1:
            # vertical line, y0 is lowest
            return int(math.floor(y0))

        m = (y1 - y0) / (x1 - x0)

        if m >= 0.0:
            # lowest point is at left edge of pixel column
            return int(math.floor(y0 + m * (x - x0)))
        else:
            # lowest point is at right edge of pixel column
            return int(math.floor(y0 + m * ((x + 1.0) - x0)))


    def maxY(x0, y0, x1, y1, x):
        if x0 == x1:
            # vertical line, y1 is highest
            return int(math.ceil(y1))

        m = (y1 - y0) / (x1 - x0)

        if m >= 0.0:
            # highest point is at right edge of pixel column
            return int(math.ceil(y0 + m * ((x + 1.0) - x0)))
        else:
            # highest point is at left edge of pixel column
            return int(math.ceil(y0 + m * (x - x0)))


    # view_bl, view_tl, view_tr, view_br are the corners of the rectangle
    view = [(corners[i, 0], corners[i, 1]) for i in range(4)]

    pixels = []

    # find l,r,t,b,m1,m2
    l, m1, m2, r = sorted(view, key=lambda p: (p[0], p[1]))
    b, t = sorted([m1, m2], key=lambda p: (p[1], p[0]))

    lx, ly = l
    rx, ry = r
    bx, by = b
    tx, ty = t
    m1x, m1y = m1
    m2x, m2y = m2

    xmin = 0
    ymin = 0
    xmax = labels_shape[1]
    ymax = labels_shape[0]

    # inward-rounded integer bounds
    # note that we're clamping the area of interest to (xmin,ymin)-(xmax,ymax)
    lxi = max(int(math.ceil(lx)), xmin)
    rxi = min(int(math.floor(rx)), xmax)
    byi = max(int(math.ceil(by)), ymin)
    tyi = min(int(math.floor(ty)), ymax)

    x1 = lxi
    x2 = rxi

    for x in range(x1, x2):
        xf = float(x)

        if xf < m1x:
            # Phase I: left to top and bottom
            y1 = minY(lx, ly, bx, by, xf)
            y2 = maxY(lx, ly, tx, ty, xf)

        elif xf < m2x:
            if m1y < m2y:
                # Phase IIa: left/bottom --> top/right
                y1 = minY(bx, by, rx, ry, xf)
                y2 = maxY(lx, ly, tx, ty, xf)

            else:
                # Phase IIb: left/top --> bottom/right
                y1 = minY(lx, ly, bx, by, xf)
                y2 = maxY(tx, ty, rx, ry, xf)

        else:
            # Phase III: bottom/top --> right
            y1 = minY(bx, by, rx, ry, xf)
            y2 = maxY(tx, ty, rx, ry, xf)

        y1 = max(y1, byi)
        y2 = min(y2, tyi)

        for y in range(y1, y2):
            pixels.append((x, y))

    return pixels


def load_config(path):
    """ Loads the configuration file

     Args:
         path: A string indicating the path to the configuration file
     Returns:
         config: A Python dictionary of hyperparameter name-value pairs
     """
    with open(path) as file:
        config = json.load(file)
    x_range = config['geometry'][3] - config['geometry'][2]
    y_range = config['geometry'][1] - config['geometry'][0]
    # print('here')
    config["input_shape"] = [int(y_range / config['voxel_size']), int(x_range / config["voxel_size"]), 36]
    config["labels_shape"] = [int(y_range / config['grid_size']), int(x_range / config["grid_size"]), 7]
    # print(config['input_shape'], config["labels_shape"])
    return config


def get_model_name(name):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        name: Name of ckpt
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    # path = "model_"
    # path += "epoch{}_".format(config["max_epochs"])
    # path += "bs{}_".format(config["batch_size"])
    # path += "lr{}".format(config["learning_rate"])

    path = os.path.join("weights", name)
    return path


def get_box_angle(cos_t, sin_t):
        """
        Calculate angle (0..2pi)
        :param cos_t: cos of angle
        :param sin_t: sin of angle

        :return: angle in radians
        """
        a_cos = math.acos(cos_t)
        angle = a_cos if sin_t >= 0 else -a_cos + 2 * math.pi
        return angle

def normalize_angle(angle):
    """
    Convert an angle to an angle belonging to the range -pi..pi
    :param angle: angle value in radians
    :return: angle value in radians
    """
    angle %= (2 * math.pi)
    angle = angle - 2 * math.pi if angle > math.pi else angle
    return angle