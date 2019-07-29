'''
Load pointcloud/labels from the KITTI dataset folder
'''
import os.path
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from addict import Dict
import torch
import torch.utils.data

from config.config_dict import get_default_config
from utils.utils import get_points_in_a_rotated_box, trasform_label2metric,get_box_angle

Config = get_default_config()


class KITTI(Dataset):
    def __init__(self, numbers, path=Config.dataset.path_str, geometry=Config.geometry, train=True):
        self._path = Path(path).expanduser().absolute()
        if train:
            self._path /= 'training'
        else:
            self._path /= 'testing'
        self.frame_range = Config.frame_range
        self.velo = numbers
        self.use_npy = Config.use_npy
        self.path_to_trainval_txt = self.get_path_to_train_val_txt()
        self.path_to_velo         = self.get_path_to_velodyne()
        self.path_to_labels       = self.get_path_to_annotations()
        self._numbers = numbers

        self.target_mean = Config.dataset.reg.mean
        self.target_std_dev = Config.dataset.reg.std
        self.transform = transforms.Normalize(self.target_mean, self.target_std_dev)
        self.y_min = geometry.width_min
        self.y_max = geometry.width_max
        self.x_min = geometry.length_min
        self.x_max = geometry.length_max
        self.h_min = geometry.height_min
        self.h_max = geometry.height_max
        self.voxel_size = Config.geometry.discretization

        self.grid_size   = Config.geometry.grid_size
        self.input_shape = Config.network.input_shape
        self.label_shape = np.array(Config.network.output_reg_shape) + np.array([0, 0, Config.network.output_class_shape[-1]])

        # self.image_sets = self.load_imageset(train)  # names
        # self.image_sets = self.filter_empty_annotations(self.image_sets)

    def __len__(self):
        return len(self._numbers)

    def __getitem__(self, item):
        return self.get_learning_data(item)

    def create_anno(self, label_list):
        annos = []
        for label in label_list:
            anno = Dict()
            anno.bbox3d.width  = torch.tensor(label.reg_target[-2])
            anno.bbox3d.length = torch.tensor(label.reg_target[-1])
            anno.bbox3d.shifts = torch.tensor([label.reg_target[3], 0, label.reg_target[2]])
            anno.bbox3d.yaw =    torch.tensor(get_box_angle(*label.reg_target[0:2]))
            annos.append(anno)
        return annos

    def get_learning_data(self, item):
        input_discrete, points = self.load_velo_scan(item)
        input_discrete = input_discrete.astype(np.float32)
        label_map, label_list = self.get_label(item)
        if label_map is not None:
            output_class, output_reg = label_map[..., 0], label_map[..., 1:]
            # output_class = np.squeeze(output_class, axis=-1)
            self.reg_target_transform(label_map)
            annotations_filter = self.create_anno(label_list)
        else:
            annotations_filter = []
            output_class, output_reg = np.squeeze(np.zeros(Config.network.output_class_shape), axis=-1),\
                                                  np.zeros(Config.network.output_reg_shape)
        output_class, output_reg = output_class.astype(np.float32), output_reg.astype(np.float32)

        return input_discrete, output_class, output_reg, annotations_filter

    # %>--------------------------------------------------<%#
    # %>--------------------- Loader ---------------------<%#
    # %>--------------------------------------------------<%#

    def load_dataset(self, batch_size):
        dataloader_kwargs = Dict()
        dataloader_kwargs.shuffle = False
        dataloader_kwargs.num_workers = 0
        dataloader_kwargs.drop_last = False
        dataloader_kwargs.pin_memory = False
        dataloader = torch.utils.data.DataLoader(self, batch_size=batch_size, **dataloader_kwargs)
        return dataloader

    def get_path_to_velodyne(self):
        return str(self._path / 'velodyne')

    def get_path_to_train_val_txt(self):
        return str(r'D:\Docs\Tasks\575_main_version_pixor\train_val_info')

    def get_path_to_annotations(self):
        return str(self._path / 'label_2')

    def reg_target_transform(self, label_map):
        '''
        Inputs are numpy arrays (not tensors!)
        :param label_map: [200 * 175 * 7] label tensor
        :return: normalized regression map for all non_zero classification locations
        '''
        cls_map = label_map[..., 0]
        reg_map = label_map[..., 1:]

        index = np.nonzero(cls_map)
        reg_map[index] = (reg_map[index] - self.target_mean) / self.target_std_dev

    def filter_empty_annotations(self, filenames):
        filtered_annotations = []
        for idx in range(len(filenames)):
            if self.annotated_objects_in_range(idx):
                filtered_annotations.append(filenames[idx])
        return filtered_annotations

    def annotated_objects_in_range(self, index):
        label_map, label_list = self.get_label(index)
        if label_list is None:
            return False
        return True

    def load_imageset(self, train):
        # path = KITTI_PATH
        if train:
            path = os.path.join(self.path_to_trainval_txt, "train.txt")
        else:
            path = os.path.join(self.path_to_trainval_txt, "val.txt")

        with open(path, 'r') as f:
            lines = f.readlines()  # get rid of \n symbol
            names = []
            if self.frame_range is not None:
                for line in lines[:-1]:
                    if int(line[:-1]) < self.frame_range:
                        names.append(line[:-1])
            else:
                for line in lines[:-1]:
                    names.append(line[:-1])
            # Last line does not have a \n symbol
            names.append(lines[-1][:6])
            print("There are {} images in txt file".format(len(names)))

            return names

    def get_corners(self, bbox):
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

    def update_label_map(self, map, bev_corners, reg_target):
        bev_corners[:, 0] -= self.x_min
        label_corners = bev_corners / self.grid_size
        label_corners[:, 1] += self.label_shape[0] / 2
        points = get_points_in_a_rotated_box(label_corners, self.input_shape)

        for p in points:
            label_x = p[0]
            label_y = p[1]
            metric_x, metric_y = trasform_label2metric(np.array(p),
                                                       ratio=(self.grid_size / self.voxel_size),
                                                       voxel_size=self.voxel_size)
            actual_reg_target = np.copy(reg_target)
            actual_reg_target[2] = reg_target[2] - metric_x
            actual_reg_target[3] = reg_target[3] - metric_y
            actual_reg_target[4] = np.log(reg_target[4])
            actual_reg_target[5] = np.log(reg_target[5])

            map[label_y, label_x, 0] = 1.0
            map[label_y, label_x, 1:7] = actual_reg_target
        return map

    def get_label(self, index):
        '''
        :param i: the ith velodyne scan in the train/val set
        :return: label map: <--- This is the learning target
                a tensor of shape 800 * 700 * 7 representing the expected output


                label_list: <--- Intended for evaluation metrics & visualization
                a list of length n; n =  number of cars + (truck+van+tram+dontcare) in the frame
                each entry is another list, where the first element of this list indicates if the object
                is a car or one of the 'dontcare' (truck,van,etc) object

        '''
        index = self._numbers[index]
        f_name = (6 - len(index)) * '0' + index + '.txt'
        label_path = os.path.join(self.path_to_labels, f_name)
        object_list = {}
        valid_object_list = ['Car', 'Truck', 'Van']

        label_map = np.zeros(self.label_shape, dtype=np.float32)
        label_list = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                bbox = []
                entry = line.split(' ')
                name = entry[0]
                object_list[name] = object_list[name] + 1 if name in object_list else 1
                bbox.append(object_list[name])
                bbox.extend([float(e) for e in entry[1:]])
                if name not in valid_object_list:
                    continue
                corners, reg_target = self.get_corners(bbox)
                if self.object_in_roi(corners):
                    label_map = self.update_label_map(label_map, corners, reg_target)

                    label = Dict()
                    label.corners = corners
                    label.reg_target = reg_target
                    label_list.append(label)
        if not label_list:
            return None, None
        return label_map, label_list

    def object_in_roi(self, bbox):
        if np.amax(bbox[:, 0]) > self.x_max \
                or np.amin(bbox[:, 0]) < self.x_min \
                or np.amax(bbox[:, 1]) > self.y_max \
                or np.amin(bbox[:, 1]) < self.y_min:
            return False
        return True

    def get_rand_velo(self):
        import random
        rand_v = random.choice(self.velo)
        print("A Velodyne Scan has shape ", rand_v.shape)
        return random.choice(self.velo)

    def load_velo_scan(self, item):
        """Helper method to parse velodyne binary files into a list of scans."""
        filename = os.path.join(self.get_path_to_velodyne(), self.velo[item])

        if self.use_npy:
            points = np.load(filename[:-4] + '.npy')
        else:
            filename += '.bin'
            points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
            scan = self.lidar_preprocess(points)
        return scan, points

    def load_velo(self):
        """Load velodyne [x,y,z,reflectance] scan data from binary files."""
        # Find all the Velodyne files

        velo_files = []
        for file in self.image_sets:
            file = '{}.bin'.format(file)
            velo_files.append(os.path.join(self.path_to_velo, file))

        print('Found ' + str(len(velo_files)) + ' Velodyne scans...')
        # Read the Velodyne scans. Each point is [x,y,z,reflectance]
        self.velo = velo_files

        print('done.')

    def point_in_roi(self, point):
        if (point[0] - self.x_min) < 0.01 or (self.x_max - point[0]) < 0.01:
            return False
        if (point[1] - self.y_min) < 0.01 or (self.y_max - point[1]) < 0.01:
            return False
        if (point[2] - self.h_min) < 0.01 or (self.h_max - point[2]) < 0.01:
            return False
        return True

    def lidar_preprocess(self, scan):
        velo = scan
        velo_processed = np.zeros(self.input_shape, dtype=np.float32)
        intensity_map_count = np.zeros((velo_processed.shape[0], velo_processed.shape[1]))
        for i in range(velo.shape[0]):
            if self.point_in_roi(velo[i, :]):
                x = int((velo[i, 1] - self.y_min) / self.voxel_size)
                y = int((velo[i, 0] - self.x_min) / self.voxel_size)
                z = int((velo[i, 2] - self.h_min) / 0.1)
                # print("velo_processed shape = ", velo_processed.shape, x, y, z)
                velo_processed[x, y, z] = 1
                velo_processed[x, y, -1] += velo[i, 3]
                intensity_map_count[x, y] += 1
        velo_processed[:, :, -1] = np.divide(velo_processed[:, :, -1], intensity_map_count,
                                             where=intensity_map_count != 0)

        return velo_processed
