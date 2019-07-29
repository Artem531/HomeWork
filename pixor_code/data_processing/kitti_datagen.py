from config.config_dict import get_default_config

import warnings
warnings.filterwarnings('ignore')  # suppress Anaconda warnings
warnings.simplefilter('ignore', DeprecationWarning)


from pathlib import Path
import collections
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import vispy
import vispy.scene
from addict import Dict

from IPython.display import Image

import torch
import torch.utils.data
import torch.nn
import torch.optim
import torch.autograd

from structures.object_info import create_from_src_anno
import augmentation

Config = get_default_config()

from augmentation import data_augmentation as augmentation


class KittiDatasetIterator(collections.Iterator):
    def __init__(self, kitti_dataset):
        self._kitti_dataset = kitti_dataset
        self._current_index = 0
        self._limit = len(self._kitti_dataset)

    def __next__(self):
        if self._current_index >= self._limit:
            raise StopIteration

        self._current_index += 1
        return self._kitti_dataset[self._current_index - 1]

    def __len__(self):
        return self._limit


class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, numbers, path=Config.dataset.path_str, geometry=Config.dataset, is_training=True):
        self._path = Path(path).expanduser().absolute()
        if is_training:
            self._path /= 'training'
        else:
            self._path /= 'testing'

        self._numbers = numbers

        self._geometry = Config.geometry.copy()
        if geometry:
            self._geometry.update(geometry)

    def __getitem__(self, index):
        return self.get_learning_data(index)

    def __len__(self):
        return len(self._numbers)

    def __iter__(self):
        return KittiDatasetIterator(self)

    # %>--------------------------------------------------<%#
    # %>---------------------- Path ----------------------<%#
    # %>--------------------------------------------------<%#

    def get_path_to_velodyne(self, index):
        return str(self._path / 'velodyne' / f'{self._numbers[index]}.bin')

    def get_path_to_calib(self, index):
        return str(self._path / 'calib' / f'{self._numbers[index]}.txt')

    def get_path_to_image_2(self, index):
        return str(self._path / 'image_2' / f'{self._numbers[index]}.png')

    def get_path_to_annotations(self, index):
        return str(self._path / 'label_2' / f'{self._numbers[index]}.txt')

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

    # %>--------------------------------------------------<%#
    # %>-------------------- Velodyne --------------------<%#
    # %>--------------------------------------------------<%#

    def get_velodyne(self, index, num_point_features=4):
        path = self.get_path_to_velodyne(index)
        points = np.fromfile(path, dtype=np.float32).reshape(-1, num_point_features)  # [[x, y, z, reflectance], ...]

        # swap x <-> y because of Config.network.input_shape used lwh format
        points[:, [0, 1]] = points[:, [1, 0]]
        points[:, 0] = -points[:, 0]

        return points

    @staticmethod
    def filter_raw_velodyne(points, geometry=Config.geometry):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        mask = (geometry.width_min <= x) & (x < geometry.width_max) & \
               (geometry.length_min <= y) & (y < geometry.length_max) & \
               (geometry.height_min <= z) & (z < geometry.height_max)

        return points[mask]

    @staticmethod
    def preprocess_raw_velodyne(points, geometry=Config.geometry, return_filtered=False):
        points_filtered = KittiDataset.filter_raw_velodyne(points, geometry)
        points = points_filtered.copy()

        points[:, :3] -= [geometry.width_min, geometry.length_min, geometry.height_min]

        #         assert np.all(points.min(axis=0) >= 0), points.min(axis=0)

        points[:, :3] /= geometry.discretization
        points[:, :3] = points[:, :3].astype(np.uint64)

        #         assert np.sum(np.isnan(points)) == 0, np.sum(np.isnan(points))
        #         assert np.all(points[:, :3].max(axis=0) < Config.network.input_shape), points[:, :3].max(axis=0)

        result = np.zeros(Config.network.input_shape, dtype=np.float32)
        result[points[:, 0].astype(np.int64), points[:, 1].astype(np.int64), points[:, 2].astype(np.int64)] = 1

        #         assert np.sum(np.isnan(points)) == 0, np.sum(np.isnan(points))
        #         assert np.sum(result[:, :,  -1]) == 0, np.sum(result[:, :,  -1])

        points_x_y = points[:, [0, 1]].astype(np.int64)
        reflectance = points[:, -1].T

        # Black magic
        uniq, indices, counts = np.unique(points_x_y, return_index=True, return_counts=True, axis=0)
        split = np.split(reflectance, np.cumsum(counts))[:-1]
        mean_reflectance = np.array(list(map(lambda ar: ar.mean(), split)))  # TODO: vectorize

        x_ind, y_ind = points_x_y[indices].T
        result[x_ind, y_ind, -1] = mean_reflectance

        #         assert np.sum(result[:, :, -1]) != 0
        #         assert np.sum(np.isnan(points)) == 0, np.sum(np.isnan(points))

        if return_filtered:
            return result, points_filtered

        return result

    def get_velodyne_preproc(self, index, **kwargs):
        return KittiDataset.preprocess_raw_velodyne(self.get_velodyne(index), self._geometry, **kwargs)

    @staticmethod
    def show_raw_velodyne(points):
        # http://vispy.org/scene.html#module-vispy.scene.canvas
        # http://vispy.org/visuals.html#vispy.visuals.MarkersVisual.set_data
        canvas = vispy.scene.canvas.SceneCanvas(keys='interactive', show=False, always_on_top=True)
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'

        axis = vispy.scene.visuals.XYZAxis(parent=view.scene)
        scatter = vispy.scene.visuals.Markers(parent=view.scene)
        scatter.set_data(
            pos=points[:, :3],
            size=points[:, 3] * 3,
            edge_width=0.0,
            symbol='disc',
            edge_color='white',
            face_color='white',
        )

        return canvas.show()

    def show_velodyne(self, index):
        path = self.get_path_to_velodyne(index)
        return KittiDataset.show_raw_velodyne(self.get_velodyne(index))

    @staticmethod
    def show_raw_velodyne_bev(points, annotations=None, scale=1, geometry=Config.geometry):
        fig, ax = plt.subplots()

        plt.xlim(Config.geometry.width_min * scale, Config.geometry.width_max * scale)
        plt.ylim(Config.geometry.length_min * scale, Config.geometry.length_max * scale)

        ax.scatter(points[:, 0], points[:, 1], s=points[:, -1] * 2.5)

        # axes
        ax.arrow(0, 0, 2, 0, color='r', head_width=0.4, head_length=0.3)  # OX
        ax.arrow(0, 0, 0, 2, color='g', head_width=0.4, head_length=0.3)  # OY

        for anno in annotations or []:
            ax.add_patch(mpl.patches.Polygon(
                list(zip(anno.bbox3d.velodyne2d.x_coord, anno.bbox3d.velodyne2d.y_coord)), facecolor=anno.color,
                edgecolor=anno.color, alpha=0.5)
            )

        plt.show()
        return fig, ax

    def show_velodyne_bev(self, index, **kwargs):
        path = self.get_path_to_velodyne(index)
        return KittiDataset.show_raw_velodyne_bev(self.get_velodyne(index), annotations=self.get_annotations(index),
                                                  **kwargs)

    # %>-----------------------------------------------<%#
    # %>-------------------- Calib --------------------<%#
    # %>-----------------------------------------------<%#

    #     @functools.lru_cache(maxsize=Config.settings.lru_cache_size)
    def get_calib(self, index):
        path = self.get_path_to_calib(index)
        calib = Dict()
        with open(path, encoding='utf-8', mode='r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                key, value = line.split(':', 1)
                calib[key] = np.array([float(x) for x in value.strip().split()])

                if key == 'R0_rect':
                    R0_rect = np.zeros((4, 4))
                    R0_rect[:3, :3] = calib.R0_rect.reshape(3, 3)
                    R0_rect[-1][-1] = 1.0
                    calib.R0_rect = R0_rect

                elif key.startswith('Tr_'):
                    Tr_xxx_to_xxx = np.concatenate([calib[key].reshape(-1, 4), np.zeros((1, 4))], axis=0)
                    Tr_xxx_to_xxx[-1][-1] = 1.0
                    calib[key] = Tr_xxx_to_xxx

                else:
                    calib[key] = calib[key].reshape(-1, 4)

        return calib

    # %>-------------------------------------------------<%#
    # %>-------------------- Image_2 --------------------<%#
    # %>-------------------------------------------------<%#

    def show_image_2(self, index):
        return Image(filename=self.get_path_to_image_2(index))

        # %>-----------------------------------------------------<%#

    # %>-------------------- Annotations --------------------<%#
    # %>-----------------------------------------------------<%#

    #     @functools.lru_cache(maxsize=Config.settings.lru_cache_size)


    @staticmethod
    def save_labels(self, anno, pcloud, cloud_path, cloud_name, label_path, label_name, valeo_flag):
        """
        This function save labels and cloud like Kitti
        :param anno:
        :param pcloud:
        :return:
        """
        pcloud = self.inverter(pcloud)
        augmentation.save_cloud(cloud_path, cloud_name, pcloud)
        anno.bbox3d.velodyne2d.y_coord = anno.bbox3d.corners[1, :4]
        anno.bbox3d.velodyne2d.y = anno.bbox3d.y

        Obj_info = create_from_src_anno(anno)
        augmentation.save_labels(label_path, label_name, Obj_info, valeo_flag)

    def get_annotations(self, index):
        path = self.get_path_to_annotations(index)
        calib = self.get_calib(index)

        annotations = []
        with open(path, encoding='utf-8', mode='r') as f:
            for label in f:
                label = label.strip()
                if not label:
                    continue

                anno = Dict()
                anno.index = index
                anno.filenumber = self._numbers[index]

                label = label.split(' ')
                anno.type = label[0]  # 'Car', 'Cyclist', 'Pedestrian', ...
                anno.cls = Config.dataset.type2cls[anno.type]

                if anno.cls == Config.dataset.remove_cls:
                    continue

                anno.color = Config.dataset.cls2color[anno.cls]

                label = list(map(float, label[1:]))

                anno.truncated = label[0]  # truncated pixel ratio ([0..1])
                anno.occluded = int(label[1])  # 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
                anno.alpha = label[2]  # object observation angle ([-pi..pi])

                #                 # 2D bounding box of object in the image (0-based index): contains left, top, right, bottom pixel coordinates
                anno.bbox2d.x1 = label[3]  # left
                anno.bbox2d.y1 = label[4]  # top
                anno.bbox2d.y2 = label[5]  # right
                anno.bbox2d.x2 = label[6]  # bottom

                # 3D object dimensions: height, width, length (in meters)
                w = anno.bbox3d.width  = label[7] + Config.dataset.add_width
                h = anno.bbox3d.height = label[8] + Config.dataset.add_height
                l = anno.bbox3d.length = label[9] + Config.dataset.add_length
                # 3D object location x,y,z in camera coordinates (in meters)
                x = anno.bbox3d.x = label[10]
                y = anno.bbox3d.y = label[11]
                z = anno.bbox3d.z = label[12]
                # Rotation ry around Y-axis in camera (!) coordinates [-pi..pi]
                anno.bbox3d.yaw = label[13]

                c = np.cos(anno.bbox3d.yaw)
                s = np.sin(anno.bbox3d.yaw)
                Ry = anno.bbox3d.Ry = np.array([
                    [c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c]
                ])

                x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
                y_corners = [0, 0, 0, 0, h, h, h, h]
                z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

                # x,y,z + distance between velo to cam
                anno.bbox3d.shifts = np.array([x, y, z]) - (calib.R0_rect @ calib.Tr_velo_to_cam)[:-1, -1]

                corners = anno.bbox3d.corners = Ry.dot(
                    np.array([x_corners, y_corners, z_corners])) + anno.bbox3d.shifts.reshape((3, 1))

                # TODO: convert via `calib.R0_rect @ calib.Tr_velo_to_cam` matrix
                # Convert camera coordinates to point cloud
                # + swap x <-> y because of Config.network.input_shape used lwh format
                anno.bbox3d.velodyne2d.x_coord = anno.bbox3d.corners[0, :4]
                anno.bbox3d.velodyne2d.y_coord = anno.bbox3d.corners[2, :4]
                anno.bbox3d.velodyne2d.x = x  # z
                anno.bbox3d.velodyne2d.y = z  # -x
                anno.bbox3d.velodyne2d.shifts = [anno.bbox3d.shifts[0], anno.bbox3d.shifts[2]]
                anno.bbox3d.velodyne2d.yaw = anno.bbox3d.yaw  # [-pi..pi]
                anno.bbox3d.velodyne2d.Rz = np.array([
                    [np.cos(anno.bbox3d.velodyne2d.yaw), np.sin(anno.bbox3d.velodyne2d.yaw)],
                    [-np.sin(anno.bbox3d.velodyne2d.yaw), np.cos(anno.bbox3d.velodyne2d.yaw)],
                ])
                anno.bbox3d.velodyne2d.Rz_inv = np.array([
                    [np.cos(-anno.bbox3d.velodyne2d.yaw), np.sin(-anno.bbox3d.velodyne2d.yaw)],
                    [-np.sin(-anno.bbox3d.velodyne2d.yaw), np.cos(-anno.bbox3d.velodyne2d.yaw)],
                ])

                #                 # Convert camera coordinates to point cloud
                #                 # + swap x <-> y because of Config.network.input_shape used lwh format
                #                 anno.bbox3d.velodyne3d.x_coord = anno.bbox3d.corners[0, :]
                #                 anno.bbox3d.velodyne3d.y_coord = anno.bbox3d.corners[2, :]
                #                 anno.bbox3d.velodyne3d.z_coord = -anno.bbox3d.corners[1, :]
                #                 anno.bbox3d.velodyne3d.x = x   # z
                #                 anno.bbox3d.velodyne3d.y = z   # -x
                #                 anno.bbox3d.velodyne3d.z = -y
                #                 anno.bbox3d.velodyne3d.yaw = -anno.bbox3d.yaw  # [-pi..pi]

                #                 anno.network.regression = [np.cos(anno.bbox3d.velodyne2d.yaw), np.cos(anno.bbox3d.velodyne2d.yaw),
                #                                            anno.bbox3d.velodyne2d.x, anno.bbox3d.velodyne2d.y,
                #                                            anno.bbox3d.width, anno.bbox3d.length]

                annotations.append(anno)
        return annotations

    # %>-------------------------------------------------------<%#
    # %>----------------------- General -----------------------<%#
    # %>-------------------------------------------------------<%#

    def show(self, index, bev_scale=1):
        self.show_velodyne_bev(index, scale=bev_scale)
        # self.show_velodyne(index)
        return self.show_image_2(index)

    def show_random(self, bev_scale=1):
        index = np.random.randint(0, len(self), 1)[0]
        return self.show(index, bev_scale=bev_scale)

    # %>-------------------------------------------------------<%#
    # %>-------------------- Learning pair --------------------<%#
    # %>-------------------------------------------------------<%#

    #     @functools.lru_cache(maxsize=Config.settings.lru_cache_size)
    def get_learning_data(self, index):
        # _________This is augmentation____________________

        #__________________________________________________


        input_discrete, points = self.get_velodyne_preproc(index, return_filtered=True)
        output_class = np.zeros(Config.network.output_class_shape)
        output_reg   = np.zeros(Config.network.output_reg_shape)

        #         calib = self.get_calib(index)
        annotations = self.get_annotations(index)
        annotations_filter = []
        for anno in annotations:
            points_in_center_bbox = anno.bbox3d.velodyne2d.Rz_inv.dot(
                (points[:, [0, 1]] - anno.bbox3d.velodyne2d.shifts).T).T
            mask = (-anno.bbox3d.width / 2 <= points_in_center_bbox[:, 1]) & (
                        points_in_center_bbox[:, 1] <= anno.bbox3d.width / 2) & \
                   (-anno.bbox3d.length / 2 <= points_in_center_bbox[:, 0]) & (
                               points_in_center_bbox[:, 0] <= anno.bbox3d.length / 2)
            points_in_box = points[mask][:, [0, 1]]

            if points_in_box.shape[0] == 0:
                continue

            points_dx_dy = np.array([anno.bbox3d.velodyne2d.x, anno.bbox3d.velodyne2d.y]) - points_in_box
            points_x_y = (points_in_box - np.array([self._geometry.width_min, self._geometry.length_min])) / \
                                                    self._geometry.discretization / \
                                                    Config.network.in_out_ratio
            points_x_y = points_x_y.astype(np.int64)

            c = np.cos(anno.bbox3d.velodyne2d.yaw)
            s = np.sin(anno.bbox3d.velodyne2d.yaw)
            w = anno.bbox3d.width
            l = anno.bbox3d.length

            for uniq_x_y in np.unique(points_x_y, axis=0):
                dx, dy = points_dx_dy[(points_x_y == uniq_x_y).all(axis=1)].mean(axis=0)
                with open('log_file.txt', 'a')as f:
                    f.write(str(dx) + ' ' + str(dy) + '\n')

                x, y = uniq_x_y

                output_reg[x, y, :] = (np.array(
                    [c, s, dx, dy, np.log(w), np.log(l)]) - Config.dataset.reg.mean) / Config.dataset.reg.std
                output_class[x, y, 0] = anno.cls

            #             assert np.sum(np.isnan(output_reg)) == 0, np.sum(np.isnan(output_reg))

            # Save annotations with valid GT
            annotations_filter.append(anno)
            # print([x, y], anno.cls, [c, s, dx, dy, np.log(w), np.log(l)])

        input_discrete = input_discrete.astype(np.float32)
        output_class   = np.squeeze(output_class, axis=-1).astype(np.float32)
        output_reg     = output_reg.astype(np.float32)

        return input_discrete, output_class, output_reg, annotations_filter, points

