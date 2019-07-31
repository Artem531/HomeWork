import numpy as np
import os
from tqdm import tqdm
from argparse import ArgumentParser
from utils.utils import load_config, iou_bboxes
from structures.object_info import ObjectInfo
from augmentation.scripts import test_inference
from augmentation.utils import load_annotation
from augmentation.structures import configuration_info
from augmentation.utils import object_transformation

parser = ArgumentParser()
parser.add_argument('-cfg',
                    '--config_file',
                    type=str,
                    default=".\\configs\\config.json",
                    help="Path to config file")
parser.add_argument('-out',
                    '--out_path',
                    type=str,
                    default=r"D:\New folder\valeo_dynamic_obj_detection_test",
                    help="Path to out files")


def create_folders(out_path):
    """
    Create folder for output clouds and labels
    :param out_path: out folder
    """
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    cloud_path = os.path.join(out_path, "aug_clouds")
    if not os.path.exists(cloud_path):
        os.mkdir(cloud_path)
    label_path = os.path.join(out_path, "aug_labels")
    if not os.path.exists(label_path):
        os.mkdir(label_path)


class AugmentParameters:
    """  Set boundaties of values for parameters and value for augmentation point cload and label.
    """
    def __init__(self):
        self.min_augment_shift_x = - 1
        self.max_augment_shift_x = 1
        self.min_augment_shift_y = -1
        self.max_augment_shift_y = 5
        self.min_augment_shift_z = 0
        self.max_augment_shift_z = 0
        self.max_rotation = 7
        self.max_density = 1
        self.min_density = 0.8
        self.max_distance = 10

        self.shift_x = 0
        self.shift_y = 0
        self.shift_z = 0
        self.rotation_2d = 0
        self.density = 0
        self.distance = 0

    def generate_random_transform_params(self):
        """
        Generate values for thansformation of objects and point cloud.
        """
        self.shift_x = np.random.uniform(self.min_augment_shift_x, self.max_augment_shift_x)
        self.shift_y = np.random.uniform(self.min_augment_shift_y, self.max_augment_shift_y)
        self.shift_z = np.random.uniform(self.min_augment_shift_z, self.max_augment_shift_z)

        self.rotation_2d = np.random.uniform(-self.max_rotation, self.max_rotation)
        self.density = np.random.uniform(self.min_density, self.max_density)

        self.distance = np.random.uniform(0, self.max_distance)

    def __str__(self):
        main_str = ""
        for att in self.__dict__:
            main_str += f"{att}: { self.__dict__[att]}\n "
        return main_str


def check_intersections(labels_list, label: ObjectInfo):
    """
    Checks if the label intersections with labels from a list
    :param labels_list: list of ObjectInfo objects
    :param label: ObjectInfo object
    :return: True if the label intersects at least one label from the list, else False
    """
    zero_threshold = 10e-100
    for label_in_list in labels_list:
        iou = iou_bboxes(label_in_list, label)
        if iou >= zero_threshold:
            return True
    return False


def check_valid_place(env, label: ObjectInfo):
    """
    Checks that space on the point cloud is free to add a new object
    :param env: point cloud where a new object will be added
    :param label: ObjectInfo object
    :return: True if space is free
    """
    const_perc_threshold = 0.02
    threshold_num_points = label.bbox_size[0] * label.bbox_size[1] * const_perc_threshold
    get_env_block = object_transformation.get_obj_points(env, label)
    return get_env_block.shape[0] <= threshold_num_points


def augment_object(aug_obj,
                   pobject,
                   label: ObjectInfo, CONFIG,
                   flip_flag=False,
                   rotate_flag=False,
                   distance_flag=False):
    """
    Apply augmentation to object
    :param aug_obj: AugmentParameters object
    :param pobject: points of object
    :param label: ObjectInfo for points of object
    :param flip_flag: optional; flip objects
    :param rotate_flag: optional; rotate object
    :param distance_flag: optional; shift object
    :return: new point cloud and new label
    """
    pobject, label = object_transformation.shift_obj(aug_obj, pobject, label)
    if flip_flag:
        pobject, label = object_transformation.flip(pobject, label)
    if rotate_flag:
        pobject, label = object_transformation.rotate_object(pobject, label, aug_obj.rotation_2d)
    if distance_flag:
        pobject, label = object_transformation.distance_shift(aug_obj, pobject, label, CONFIG)
    return pobject, label


def augment_env(pcloud):
    """
    Augment pcloud
    :param pcloud: point cloud
    :return: new point cloud
    """
    random_shifting_matrix = np.random.rand(*pcloud.shape) * np.random.rand(*pcloud.shape)
    return pcloud + random_shifting_matrix / 2


def add_object_to_cloud(aug_obj, pcloud, labels, pobject, label):
    """
    Add object to point cloud
    :param aug_obj:  AugmentParameters object
    :param pcloud: source point cloud
    :param labels: source list of ObjectInfo objects
    :param pobject: new object
    :param label: new ObjectInfo object
    :return: new point cloud, new list of ObjectInfo objects
    """
    if pobject.shape[0] < 4:
        #print(labels)
        return pcloud, labels
    new_pcloud = pcloud
    config = configuration_info.Configuration(CONFIG)
    counter = 1
    max_iterasion = 5000
    while True:
        if not config.check_objinfo(label) \
                or check_intersections(labels, label) \
                or not check_valid_place(new_pcloud, label):
            if counter > max_iterasion:
                return pcloud, labels
            aug_obj.generate_random_transform_params()
            pobject, label = object_transformation.shift_obj(aug_obj, pobject, label)
            counter += 1
        else:
            new_pcloud = np.concatenate((pcloud, pobject), axis=0)
            labels.append(label)
            #print(labels)
            return new_pcloud, labels


def add_object_to_cloud(aug_obj, pcloud, labels, pobject, label, CONFIG):
    """
    Add object to point cloud
    :param aug_obj:  AugmentParameters object
    :param pcloud: source point cloud
    :param labels: source list of ObjectInfo objects
    :param pobject: new object
    :param label: new ObjectInfo object
    :return: new point cloud, new list of ObjectInfo objects
    """
    if pobject.shape[0] < 4:
        return pcloud, labels
    new_pcloud = pcloud
    config = configuration_info.Configuration(CONFIG)
    counter = 1
    max_iterasion = 5000
    while True:
        if not config.check_objinfo(label) \
                or check_intersections(labels, label) \
                or not check_valid_place(new_pcloud, label):
            if counter > max_iterasion:
                return pcloud, labels
            aug_obj.generate_random_transform_params()
            pobject, label = object_transformation.shift_obj(aug_obj, pobject, label)
            counter += 1
        else:
            new_pcloud = np.concatenate((pcloud, pobject), axis=0)
            labels.append(label)
            return new_pcloud, labels
        if counter == 10:
            new_pcloud = np.concatenate((pcloud, pobject), axis=0)
            labels.append(label)
            return new_pcloud, labels

def augment_cloud(aug_obj, pcloud, labels: list, CONFIG,  flag_transp_env=False):
    """
    Apply augmentation to cloud
    :param aug_obj: AugmentParameters object
    :param pcloud: source point cloud
    :param labels:  source list of ObjectInfo objects
    :param flag_transp_env: optional; to do augmentation of environment point cloud
    :return: new point cloud, new list of ObjectInfo objects
    """

    env = object_transformation.remove_objects_from_cloud(pcloud, labels)

    if flag_transp_env:
        env = augment_env(env)
    new_labels = []
    new_pcloud = env
    for label in labels:
        aug_obj.generate_random_transform_params()
        pobject = object_transformation.get_obj_points(pcloud, label)
        #from utils.visualize_utils import visualize
        #visualize([pobject], [[label]])
        flip_flag     = random_flag()
        rotate_flag   = random_flag()
        distance_flag = random_flag()
        pobject, label = augment_object(aug_obj, pobject, label, CONFIG,
                                        flip_flag    =flip_flag,
                                        rotate_flag  =rotate_flag,
                                        distance_flag=distance_flag)
        #print(label)
        new_pcloud, new_labels = add_object_to_cloud(aug_obj, new_pcloud, new_labels, pobject, label, CONFIG)

        #from utils.visualize_utils import visualize
        #visualize([new_pcloud], [[new_labels]])
    return new_pcloud, new_labels


def aug_process_clouds(cloud_folder, label_folder, valeo_flag, visualize_flag=False, file_range=None):
    """
    Apply augmentation to files
    :param cloud_folder: path to folder with clouds files
    :param label_folder: path to folder with labels files
    :param valeo_flag: if True - valeo dataset else - kitti
    :param visualize_flag: optional; show the result of augmentation
    :param file_range: optional; num of file for processing
    """
    create_folders(args.out_path)

    show_me = test_inference.visualize
    files_list = os.listdir(cloud_folder) if file_range is None else os.listdir(cloud_folder)[:file_range]
    paths_dict = {"clouds_folder": cloud_folder,
                  "labels_folder": label_folder,
                  "valeo_flag": valeo_flag}

    for pc_filename in tqdm(files_list[7:]):

        if not os.path.isfile(os.path.join(cloud_folder, pc_filename)):
            continue

        if not valeo_flag:
            cloud, label_list = load_annotation.get_processed_cloud(pc_filename, paths_dict)
        else:
            cloud, label_list = load_annotation.get_source_cloud(pc_filename, paths_dict)

        if label_list:
            augment = AugmentParameters()
            new_pcloud, new_labels = augment_cloud(augment, cloud, label_list,
                                                    flag_transp_env=random_flag())

            if visualize_flag:
                show_me([cloud], [label_list])
                show_me([new_pcloud], [new_labels])
            if new_labels and new_pcloud.size > 0:
                name_cloud = "aug_" + pc_filename
                name_label = "aug_" + pc_filename.split(".")[0]
                # save_cloud(name_cloud, new_pcloud)
                # save_labels(name_label, new_labels, valeo_flag)


def random_flag():
    """
    Generate random bool flag
    :return: bool flag
    """
    return bool(np.random.random_integers(0, 1, 1))


if __name__ == '__main__':
    cloud_folder = os.path.normpath(r"D:\New folder\valeo_dynamic_obj_detection_test\clouds")
    label_folder = os.path.normpath(r"D:\New folder\valeo_dynamic_obj_detection_test\labels")
    # cloud_folder = r"D:\kitti_pixor\lidar_pixor\data_object_velodyne\training\velodyne"
    # label_folder = r"D:\kitti_pixor\lidar_pixor\data_object_label_2\training\label_2"
    # cloud_folder = r'D:\New folder\pixor\out_files\kitti_x2\aug_clouds'
    # label_folder = r'D:\New folder\pixor\out_files\kitti_x2\aug_labels'
    # cloud_folder = r"E:\Docs\Datasets\kitti\lidar_pixor\data_object_velodyne\training\velodyne"
    # label_folder = r"E:\Docs\Datasets\kitti\lidar_pixor\data_object_label_2\training\label_2"

    args = parser.parse_args()
    CONFIG = load_config(args.config_file)
    valeo_flag = True
    aug_process_clouds(cloud_folder, label_folder, valeo_flag, visualize_flag=True, file_range=None)
