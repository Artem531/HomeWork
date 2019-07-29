import os
import math
import numpy as np

#from augmentation.scripts import valeo_implement
from augmentation.structures.object_info import ObjectInfo
from augmentation.utils import paths_manager as pm
from augmentation.utils.utils import crop_cloud
from augmentation.utils.vis_3d import visualize
from augmentation.utils import object_transformation


def get_kitti_labels(filename, config):
    object_list = {'Car': 1, 'Truck': 0, 'DontCare': 0, 'Van': 1, 'Tram': 0}
    label_list = []
    with open(filename, 'r') as f:
        lines = f.readlines()  # get rid of \n symbol
        for line in lines:
            bbox = []
            entry = line.split(' ')
            name = entry[0]
            if name in list(object_list.keys()):
                bbox.append(object_list[name])
                bbox.extend([float(e) for e in entry[1:]])
                if name in ['Car', 'Van']:
                    corners, reg_target = get_corners(bbox)
                    if object_in_roi(corners, config):
                        angle = object_transformation.get_angle(reg_target[0], reg_target[1]) - math.pi / 2
                        angle = object_transformation.normalize_angle(angle)
                        cos_t = math.cos(angle)
                        sin_t = math.sin(angle)
                        bbox_center = reg_target[2:4]
                        bbox_size   = reg_target[4:6]
                        label = ObjectInfo([*bbox_center, *bbox_size, cos_t, sin_t])
                        label_list.append(label)
    if not label_list:
        return None
    return label_list


def get_labels_range(label_list, delta=1.0):
    min_x = np.amin(label_list[0].bbox[:, 0])
    max_x = np.amax(label_list[0].bbox[:, 0])

    min_y = np.amin(label_list[0].bbox[:, 1])
    max_y = np.amax(label_list[0].bbox[:, 1])

    for i in range(1, len(label_list)):
        min_x = min(min_x, np.amin(label_list[i].bbox[:, 0]))
        min_y = min(min_y, np.amin(label_list[i].bbox[:, 1]))
        max_x = max(max_x, np.amax(label_list[i].bbox[:, 0]))
        max_y = max(max_y, np.amax(label_list[i].bbox[:, 1]))

    return min_y - delta, max_y + delta, min_x - delta, max_x + delta


def get_corners(bbox):
    w, h, l, y, z, x, yaw = bbox[8:15]
    y = -y
    # manually take a negative s. t. it's a right-hand system, with
    # x facing in the front windshield of the car
    # z facing up
    # y facing to the left of driver
    yaw = -(yaw + np.pi / 2)
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    # rear left
    bev_corners[0, 0] = x - l / 2 * np.cos(yaw) - w / 2 * np.sin(yaw)
    bev_corners[0, 1] = y - l / 2 * np.sin(yaw) + w / 2 * np.cos(yaw)

    # rear right
    bev_corners[1, 0] = x - l / 2 * np.cos(yaw) + w / 2 * np.sin(yaw)
    bev_corners[1, 1] = y - l / 2 * np.sin(yaw) - w / 2 * np.cos(yaw)

    # front right
    bev_corners[2, 0] = x + l / 2 * np.cos(yaw) + w / 2 * np.sin(yaw)
    bev_corners[2, 1] = y + l / 2 * np.sin(yaw) - w / 2 * np.cos(yaw)

    # front left
    bev_corners[3, 0] = x + l / 2 * np.cos(yaw) - w / 2 * np.sin(yaw)
    bev_corners[3, 1] = y + l / 2 * np.sin(yaw) + w / 2 * np.cos(yaw)

    reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]

    return bev_corners, reg_target


def object_in_roi(bbox, config):
    if np.amax(bbox[:, 0]) > config['geometry'][3] \
            or np.amin(bbox[:, 0]) < config['geometry'][2] \
            or np.amax(bbox[:, 1]) > config['geometry'][1] \
            or np.amin(bbox[:, 1]) < config['geometry'][0]:
        return False
    return True


def get_source_cloud(cloud_name, paths_dict, visualize_flag=False):
    """
    Read cloud file and file with labels
    :param cloud_name: filename of cloud
    :param valeo_flag: if True - valeo dataset else - kitti
    :param paths_dict: dict with paths (clouds and labels path, model path, configuration file, test file, valeo_falg)
    :param visualize_flag: flag for visualization
    :return: cloud , labels list
    """
    config, model_path, filelist, clouds_folder, labels_folder, valeo_flag = pm.unpack_paths_dict(paths_dict)
    extention = '.json' if valeo_flag else '.txt'
    gt_labels_filename = os.path.join(labels_folder, cloud_name.split('.')[-2] + extention)
    cloud_path = os.path.join(clouds_folder, cloud_name)
    if os.path.isfile(cloud_path):
        if labels_folder is not None and os.path.isfile(gt_labels_filename):
            if valeo_flag:
                #label_list = valeo_implement.get_valeo_labels(gt_labels_filename, config)
                pass
            else:
                label_list = get_kitti_labels(gt_labels_filename, config)
            if label_list is None:
                return [], []
            cloud = np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 4)

            if visualize_flag:
                visualize([cloud], [label_list])
            return cloud, label_list
    return None, None


def get_processed_cloud(cloud_name,
                        paths_dict,
                        visualize_flag=False,
                        remove_ground=None):
    """

    :param cloud_name: filename of cloud
    :param valeo_flag: if True - valeo dataset else - kitti
    :param paths_dict: dict with paths (clouds and labels path, model path, configuration file, test file, valeo_falg)
    :param visualize_flag: flag for visualization
    :param remove_ground: optional; remove groundplane flag
    :return:
    """
    config, model_path, filelist, clouds_folder, labels_folder, valeo_flag = pm.unpack_paths_dict(paths_dict)
    extention = '.json' if valeo_flag else '.txt'
    gt_labels_filename = os.path.join(labels_folder, cloud_name.split('.')[-2] + extention)
    cloud_path = os.path.join(clouds_folder, cloud_name)
    if os.path.isfile(cloud_path):
        if labels_folder is not None and os.path.isfile(gt_labels_filename):
            if valeo_flag:
                #label_list = valeo_implement.get_valeo_labels(gt_labels_filename, config)
                if remove_ground is None:
                    remove_ground = False
            else:
                label_list = get_kitti_labels(gt_labels_filename, config)
                if remove_ground is None:
                    remove_ground = True
            if not label_list:
                return None, None
            labels_range = get_labels_range(label_list)
            cloud = crop_cloud(np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 4),
                               config, labels_range, remove_ground=remove_ground)

            if visualize_flag:
                visualize([cloud], [label_list])
            return cloud, label_list
    return None, None
