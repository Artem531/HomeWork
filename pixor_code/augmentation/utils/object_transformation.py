import numpy as np
import math

from augmentation.structures.object_info import ObjectInfo
from augmentation.structures import configuration_info


def normalize_angle(angle):
    """
    Convert an angle to an angle belonging to the range 0..2pi
    :param angle: angle value in radians
    :return: angle value in radians
    """
    angle %= (2 * math.pi)
    return angle


def get_angle(cos_t, sin_t):
    """
    Calculate angle (0..2pi)
    :param cos_t: cos of angle
    :param sin_t: sin of angle
    :return: angle in radians
    """
    a_cos = math.acos(cos_t)
    angle = a_cos if sin_t >= 0 else -a_cos + 2 * math.pi
    return angle


def rotate_box(src_label: ObjectInfo, cos_angle, sin_angle):
    """
    Create rotated ObjectInfo.
    :param src_label: label of object before transformation
    :param cos_angle: cos of object angle
    :param sin_angle: sin of object angle
    :return: ObjectInfo object
    """
    bbox_size   = src_label.bbox_size[:2]
    bbox_center = src_label.bbox_center[:2]
    min_h = src_label.bbox_center[2] - src_label.bbox_size[2] / 2
    max_h = min_h + src_label.bbox_size[2]
    obj = ObjectInfo([*bbox_center, *bbox_size, cos_angle, sin_angle], min_h=min_h, max_h=max_h)
    return obj


def rotate_cloud(points, rotation_angle, pc_center_point=[0, 0, 0]):
    """
    Rotation points of the cloud relative to the point (0, 0, 0)
    :param points: point cloud
    :param rotation_angle: addition angle in radians [0..2*pi)
    :param pc_center_point: cloud center point in format [x, y, z];  default_value = [0, 0, 0]
    :return: rotated points
    """
    transform_matrix_z = np.array([[math.cos(rotation_angle), math.sin(rotation_angle), 0, 0],
                                   [-math.sin(rotation_angle), math.cos(rotation_angle), 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])
    addition_cshift = np.array([pc_center_point[0], pc_center_point[1], pc_center_point[2], 0])
    shifted_points = points - addition_cshift
    rotated_points = np.dot(shifted_points, transform_matrix_z)
    rotated_points += addition_cshift
    return rotated_points


def shift_box(src_label: ObjectInfo, shift_list):
    """
    Create ObjectInfo object with shift
    :param src_label: ObjectInfo object label of object before transformation
    :param shift_list: list or tuple with sift values (shift_x, shift_y, shift_z)
    :return: ObjectInfo object
    """
    shift_x, shift_y, shift_z = shift_list
    bbox_size   = src_label.bbox_size[:2]
    bbox_center = np.array(src_label.bbox_center[:2]) + np.array([shift_x, shift_y])
    min_h = (src_label.bbox_center[2] - src_label.bbox_size[2] / 2) + shift_z
    max_h = (src_label.bbox_center[2] + src_label.bbox_size[2] / 2) + shift_z

    obj = ObjectInfo([*bbox_center.tolist(), *bbox_size, src_label.cos_t, src_label.sin_t], min_h=min_h, max_h=max_h)
    return obj


def rotate_object(points, label, angle):
    """
    Rotate point cloud and label of object
    :param points: points of object
    :param label: ObjectInfo for points of object
    :param angle: if angle !=0, rotate the object at an angle, else angle = aug_obj.rotation_2d
    :return: new point cloud and new label
    """
    addition_val = math.radians(angle)
    if addition_val == 0:
        return points, label

    rotation_angle = addition_val
    rotation_angle = normalize_angle(rotation_angle)
    rotated_points = rotate_cloud(points, rotation_angle, pc_center_point=label.bbox_center)

    rotation_angle += get_angle(label.cos_t, label.sin_t)
    rotation_angle = normalize_angle(rotation_angle)
    rotated_label = rotate_box(label, math.cos(rotation_angle), math.sin(rotation_angle))
    return rotated_points, rotated_label


def flip(pcloud, label: ObjectInfo):
    """
    Flip object
    :param pcloud: points of object
    :param label: ObjectInfo for points of object
    :return: new point cloud and new label
    """
    pcloud[:, 1] *= -1
    shifts = [0, -label.bbox_center[1] * 2, 0]
    new_label = shift_box(label, shifts)
    angle_s = - get_angle(new_label.cos_t, new_label.sin_t)
    angle_s = normalize_angle(angle_s)
    new_label = rotate_box(new_label, math.cos(angle_s), math.sin(angle_s))
    return pcloud, new_label


def shift_obj(aug_obj, pcloud, label: ObjectInfo):
    """
    Shift object
    :param aug_obj: AugmentParameters object
    :param pcloud: points of object
    :param label: ObjectInfo for points of object
    :return: new point cloud and new label
    """
    aug_obj.generate_random_transform_params()
    pobject = get_obj_points(pcloud, label)
    pobject[:, 0] += aug_obj.shift_x
    pobject[:, 1] += aug_obj.shift_y
    pobject[:, 2] += aug_obj.shift_z
    new_label = shift_box(label, [aug_obj.shift_x, aug_obj.shift_y, aug_obj.shift_z])
    return pobject, new_label


def change_density(aug_obj, pcloud, density_val=None):
    """
    Delete points from point cloud
    :param aug_obj: AugmentParameters object
    :param pcloud: point cloud
    :param density_val: optional.
    :return: new point cloud
    """
    density = aug_obj.density if density_val is None else density_val
    new_size = int(pcloud.shape[0] * density)
    dif_shape = pcloud.shape[0] - new_size
    delete_num_rows = sorted(np.random.random_integers(0, pcloud.shape[0], dif_shape), reverse=True)
    new_pcloud = np.delete(pcloud, delete_num_rows, 0)
    return new_pcloud


def get_obj_points(pcloud, label: ObjectInfo):
    """
    Extract object's points from point cloud
    :param pcloud: scene cloud
    :param label: box of object
    :return: point cloud of object
    """
    bbox = label.bbox
    obj_points = pcloud[(pcloud[:, 0] < np.amax(bbox[:, 0])) &
                        (pcloud[:, 0] > np.amin(bbox[:, 0])) &
                        (pcloud[:, 1] < np.amax(bbox[:, 1])) &
                        (pcloud[:, 1] > np.amin(bbox[:, 1]))]
    angle = get_angle(label.cos_t, label.sin_t)
    angle = normalize_angle(- angle)
    rotated_points = rotate_cloud(obj_points, angle, label.bbox_center)
    rotated_box = rotate_box(label, math.cos(0), math.sin(0))
    rotated_bbox = rotated_box.bbox
    obj_points = rotated_points[(rotated_points[:, 0] < np.amax(rotated_bbox[:, 0])) &
                                (rotated_points[:, 0] > np.amin(rotated_bbox[:, 0])) &
                                (rotated_points[:, 1] < np.amax(rotated_bbox[:, 1])) &
                                (rotated_points[:, 1] > np.amin(rotated_bbox[:, 1]))]
    angle = normalize_angle(- angle)
    rotated_points = rotate_cloud(obj_points, angle, label.bbox_center)
    return rotated_points


def remove_object(pcloud, label: ObjectInfo):
    """
    Remove object from point cloud
    :param pcloud: point cloud
    :param label: ObjectInfo object
    :return: point cloud without points of object
    """
    bbox = label.bbox
    #print(bbox)

    obj_points = pcloud
    obj_points = obj_points[(obj_points[:, 0] >= np.amax(bbox[:, 0])) |
                            (obj_points[:, 0] <= np.amin(bbox[:, 0])) |
                            (obj_points[:, 1] >= np.amax(bbox[:, 1])) |
                            (obj_points[:, 1] <= np.amin(bbox[:, 1]))]

    #from utils.visualize_utils import visualize
    #visualize([obj_points], [[]])
    return obj_points


def remove_objects_from_cloud(pcloud, labels):
    """
    Remove list of objects from point cloud
    :param pcloud: point cloud
    :param labels: list of ObjectInfo objects
    :return: point cloud without objects
    """
    env_points = pcloud
    for label in labels:
        env_points = remove_object(env_points, label)
    return env_points


def distance_shift(aug_obj, pcloud, label: ObjectInfo, config):
    """
    Object shift along y axis
    :param aug_obj: AugmentParameters object
    :param pcloud: points of object
    :param label: ObjectInfo for points of object
    :return: new point cloud and new label
    """

    def calculate_max_shift_x():
        center = label.bbox_center
        angle = math.atan2(center[1], center[0])
        angle = normalize_angle(angle)
        nonlocal delta_y
        shift_x = delta_y * math.tan(angle)
        return shift_x

    aug_obj.generate_random_transform_params()
    delta_y = aug_obj.distance

    distance_x = calculate_max_shift_x()
    delta_x = np.random.uniform(0, distance_x)
    pcloud[:, 0] += delta_x
    pcloud[:, 1] += delta_y
    label = shift_box(label, [delta_x, delta_y, 0])

    config = configuration_info.Configuration(config)
    length = (config.max_y - config.min_y)
    density = 1 - (abs(label.bbox_center[1]) / length)

    pcloud = change_density(aug_obj, pcloud, density_val=density)
    return pcloud, label
