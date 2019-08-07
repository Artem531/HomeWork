from config.config_dict import get_default_config
from train import train_model
from infer import inference
from data_processing.prepare_dataset import prepare_data

from data_processing.prepare_dataset import get_data
from data_processing.kitti_datagen_experimental import KittiDataset


from utils.visualize_utils import visualize


import numpy as np
from structures.object_info import ObjectInfo


if __name__ == '__main__':
    Config = get_default_config()
    #augment()
    #path_to_velodyne = '/home/artem/MAIN/test_dataset/training/augment/'
    #open_and_visualize(path_to_velodyne, '000000', 0)

    # Config = get_default_config()
    # path_to_velodyne = 'training/velodyne'
    # dataset = get_data(Config, path_to_velodyne)
    #
    # Data = KittiDataset(dataset)
    #
    # list_clouds, list_grid, list_output_class, list_output_reg, list_annos = Data.get_learning_data(16, return_clouds = True)
    # print('########list_clouds#########')
    # print(list_clouds)
    # print('########list_output_class#########')
    # print(list_output_class)
    # print('########list_output_reg#########')
    # print(list_output_reg)
    # print('########list_anno#########')
    # print(list_annos)
    #
    # list_of_labels = []
    #
    # # visualise results
    # for idx, annos in enumerate(list_annos):
    #     labels = []
    #     for anno in annos:
    #         angle = anno.bbox3d.velodyne2d.yaw
    #         angle = normalize_angle(angle)
    #         # print(angle, anno.bbox3d.yaw)
    #         bbox = [anno.bbox3d.velodyne2d.shifts[0],
    #                 anno.bbox3d.velodyne2d.shifts[1],
    #                 anno.bbox3d.length, anno.bbox3d.width,
    #                 np.cos(angle), np.sin(angle)]
    #         labels.append(ObjectInfo(bbox))
    #     #print(type(list_clouds[idx]))
    #     print('_______________', idx)
    #     print(labels)
    #     visualize([list_clouds[idx]], [labels])

    ### Train model
    path_to_filelist = '/home/artem/luxoft_test_git/HomeWork/pixor_code/train_val_info'
    path_to_velodyne = 'training/velodyne'
    _, data_loaders = prepare_data(Config, path_to_filelist, path_to_velodyne)
    #train_model.train_net(Config, data_loaders)
    inference.create_kpi_json(data_loaders.test,
                             threshold=0.5,
                            date_time='2019-08-08')

