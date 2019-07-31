from config.config_dict import get_default_config
from data_processing.prepare_dataset import get_data
from data_processing.kitti_datagen_experimental import KittiDataset


from utils.visualize_utils import visualize

from augmentation.data_augmentation import AugmentParameters
import numpy as np
from structures.object_info import ObjectInfo
from augmentation.data_augmentation import augment_cloud
from augmentation.utils.save_annot import save_cloud
from augmentation.utils.save_annot import save_labels

import math
import os
import json


def inverter(pcloud):
    """
    this function invert axis for proper save
    :param anno: annotations
    :return:
    """
    # swap x <-> y because of Config.network.input_shape used lwh format
    points = pcloud.copy()
    points[:, 0] = -points[:, 0]
    points[:, [0, 1]] = points[:, [1, 0]]
    return points


def normalize_angle(angle):
    """
    Convert an angle to an angle belonging to the range -pi..pi
    :param angle: angle value in radians
    :return: angle value in radians
    """
    angle %= (2 * math.pi)
    angle = angle - 2 * math.pi if angle > math.pi else angle
    return angle

def get_box_angle(cos_t, sin_t):
    """
    Calculate angle (-pi..pi)
    :param cos_t: cos of angle
    :param sin_t: sin of angle

    :return: angle in radians
    """
    a_cos = math.acos(cos_t)
    angle = a_cos if sin_t >= 0 else -a_cos + 2 * math.pi
    return normalize_angle(angle)

def save_labels(out_path, name, labels: list, valeo_flag):
    """
    Save labels as txt file
    :param name: name of file
    :param labels: list of ObjectInfo objects
    :param valeo_flag: if True - valeo dataset else - kitti
    """
    label_path = out_path
    path_and_name = os.path.join(label_path, name)
    if valeo_flag:
        out_dict = {"objects": [label.annotation_to_dict() for label in labels]}
        with open(path_and_name + ".json", 'w') as outfile:
            json.dump(out_dict, outfile, indent=2, sort_keys=True)
    else:
        with open(path_and_name + ".txt", 'w') as outfile:
            list_info_labels = []
            for label in labels:
                list_info = ['Car']
                # list_info.extend([0] * 7)
                # list_info.append(label.bbox_size[0])
                # list_info.append(label.bbox_size[2])
                # list_info.append(label.bbox_size[1])
                # list_info.append(-label.bbox_center[1])
                # list_info.append(label.bbox_center[2])
                # list_info.append(label.bbox_center[0])
                # list_info.append(-get_box_angle(label.cos_t, label.sin_t))
                # list_info = map(str, list_info)
                list_info.extend([0] * 7)
                list_info.append(label.bbox_size[2])
                list_info.append(label.bbox_size[1])
                list_info.append(label.bbox_size[0])
                list_info.append(label.bbox_center[0])
                list_info.append(label.bbox_center[2])
                list_info.append(label.bbox_center[1])
                list_info.append(-get_box_angle(label.cos_t, label.sin_t))
                list_info = map(str, list_info)

                s1 = ' '.join(list_info)
                list_info_labels.append(s1)
            result_str = '\n'.join(list_info_labels)
            outfile.write(result_str)


def save(cloud, cloud_path, cloud_name, labels, label_path):
    """
    main function to save cloud and label
    :param cloud:
    :param cloud_path:
    :param cloud_name:
    :param labels:
    :param label_path:
    :return:
    """
    pcloud = inverter(cloud)
    save_labels(label_path, cloud_name, labels, False)
    save_cloud(cloud_path, cloud_name + '.bin', pcloud)



def augment():
    """
    test augment functions
    :return:
    """
    Config = get_default_config()
    path_to_velodyne = 'training/velodyne'
    dataset = get_data(Config, path_to_velodyne)

    Data = KittiDataset(dataset)
    index = 16
    tmp_label_data = Data.get_annotations(index)

    CONFIG = Config

    labels = []
    # for anno in tmp_label_data:
    #    bbox = [anno.bbox3d.velodyne2d.shifts[1],
    #            -anno.bbox3d.velodyne2d.shifts[0],
    #            anno.bbox3d.length, anno.bbox3d.width,
    #            np.cos(anno.bbox3d.yaw), np.sin(anno.bbox3d.yaw)]
    #    labels.append(ObjectInfo(bbox))

    for anno in tmp_label_data:
        angle = anno.bbox3d.velodyne2d.yaw
        angle = normalize_angle(angle)
        print(angle, anno.bbox3d.yaw)
        bbox = [anno.bbox3d.velodyne2d.shifts[0],
                anno.bbox3d.velodyne2d.shifts[1],
                anno.bbox3d.length, anno.bbox3d.width,
                np.cos(angle), np.sin(angle)]
        # bbox = [anno.bbox3d.velodyne2d.x,
        #         anno.bbox3d.velodyne2d.y,
        #         anno.bbox3d.length, anno.bbox3d.width,
        #         np.cos(angle), np.sin(angle)]

        labels.append(ObjectInfo(bbox))

    points = Data.get_velodyne(index)
    print(np.shape(points))
    #points = KittiDataset.inverter(Data, points)

    visualize([points], [labels])

    aug_obj = AugmentParameters()
    aug_obj.generate_random_transform_params()

    new_pcloud, new_labels = augment_cloud(aug_obj, points, labels, CONFIG)

    #points = KittiDataset.inverter(Data, new_pcloud)

    #labels = []
    # for anno in new_labels:
    #     bbox = [anno.bbox_center[1],
    #             -anno.bbox_center[0],
    #             anno.bbox_size[1], anno.bbox_size[0],
    #             anno.sin_t, anno.cos_t]
    #     labels.append(ObjectInfo(bbox))

    #visualize([points], [labels])
    visualize([new_pcloud], [new_labels])

    label_path = '/home/artem/MAIN/test_dataset/training/augment/training/label_2'
    cloud_path = '/home/artem/MAIN/test_dataset/training/augment/training/velodyne'
    cloud_name = '000000'
    cloud_name1 = '000001'
    save(points, cloud_path, cloud_name, labels, label_path)
    save(new_pcloud, cloud_path, cloud_name1, new_labels, label_path)

def open_and_visualize(path_to_velodyne, name, index):
    """
    open_and_visualize  cloud and label for checking
    :param path_to_velodyne:
    :param name:
    :param index:
    :return:
    """
    Config = get_default_config()
    dataset = [name]

    Data = KittiDataset(dataset, path_to_velodyne)
    tmp_label_data = Data.get_annotations(int(index))

    #print(tmp_label_data)
    labels = []
    for anno in tmp_label_data:
        angle = anno.bbox3d.velodyne2d.yaw
        angle = normalize_angle(angle)
        print(angle, anno.bbox3d.yaw)
        bbox = [anno.bbox3d.velodyne2d.shifts[0],
                anno.bbox3d.velodyne2d.shifts[1],
                anno.bbox3d.length, anno.bbox3d.width,
                np.cos(angle), np.sin(angle)]
        # bbox = [anno.bbox3d.velodyne2d.x,
        #         anno.bbox3d.velodyne2d.y,
        #         anno.bbox3d.length, anno.bbox3d.width,
        #         np.cos(angle), np.sin(angle)]

        labels.append(ObjectInfo(bbox))

    points = Data.get_velodyne(int(index))
    #points = KittiDataset.inverter(Data, points)

    visualize([points], [labels])

if __name__ == '__main__':
    Config = get_default_config()
    #augment()
    #path_to_velodyne = '/home/artem/MAIN/test_dataset/training/augment/'
    #open_and_visualize(path_to_velodyne, '000000', 0)

    Config = get_default_config()
    path_to_velodyne = 'training/velodyne'
    dataset = get_data(Config, path_to_velodyne)

    Data = KittiDataset(dataset)

    list_clouds, list_grid, list_output_class, list_output_reg, list_annos = Data.get_learning_data(16, number_of_aug=6, return_clouds = True)
    print('########list_clouds#########')
    print(list_clouds)
    print('########list_output_class#########')
    print(list_output_class)
    print('########list_output_reg#########')
    print(list_output_reg)
    print('########list_anno#########')
    print(list_annos)

    list_of_labels = []

    # visualise results
    for idx, annos in enumerate(list_annos):
        labels = []
        for anno in annos:
            angle = anno.bbox3d.velodyne2d.yaw
            angle = normalize_angle(angle)
            # print(angle, anno.bbox3d.yaw)
            bbox = [anno.bbox3d.velodyne2d.shifts[0],
                    anno.bbox3d.velodyne2d.shifts[1],
                    anno.bbox3d.length, anno.bbox3d.width,
                    np.cos(angle), np.sin(angle)]
            labels.append(ObjectInfo(bbox))
        #print(type(list_clouds[idx]))
        print('_______________', idx)
        print(labels)
        visualize([list_clouds[idx]], [labels])

