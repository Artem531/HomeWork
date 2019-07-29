from data_processing.kitti_datagen import KittiDataset
from data_processing.kitti_datagen_src import KITTI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from addict import Dict

from utils import generate_file_list as gen_fl
import json


def read_list_of_files(filename):
    """
    Read list of filename from file
    :param filename: name of file with list of files
    :return: list of names
    """
    name = filename.split('.')[-1]
    names = []
    if name == "txt":
        with open(filename, 'r') as f:
            lines = f.readlines()  # get rid of \n symbol
            for line in lines[:-1]:
                names.append(line[:-1])# + '.bin')
    elif name == "json":
        with open(filename, 'r') as f:
            data = json.load(f)
        for line in data['files']:
            names.append(line) #+ '.bin')
    return names


def set_default_settings():
    # ## Set config parametrs and other
    np.set_printoptions(precision=4, suppress=True)

    sns.set(style="ticks", context="talk")
    # plt.style.use('seaborn')
    # plt.rcParams['figure.figsize'] = (20.0, 15.0)
    # plt.rcParams['font.size'] = 10
    torch.multiprocessing.set_sharing_strategy('file_system')


def get_data(Config, path_to_velodyne):
    path_clouds_config = Config.dataset.path.glob(path_to_velodyne + '/*.bin')
    X_train_test_val = list(map(lambda p: p.stem, path_clouds_config))
    X_train_test_val= X_train_test_val if Config.dataset.frame_range is None \
                                       else X_train_test_val[:Config.dataset.frame_range]
    X_train_test_val.sort()
    return X_train_test_val


def prepare_data(Config,
                 path_to_filelist,
                 path_to_velodyne,
                 create_new_dataset=True,
                 validation_data=True):

    set_default_settings()
    # ## Inputs
    # The benchmarks consist of 7481 training images (and point clouds)
    # and 7518 test images (and point clouds) for each task.

    ### Prepare Dataset
    config_param = Dict()
    config_param.random_state  = Config.random_state
    config_param.test_val_size = Config.dataset.test_val_size

    datasets_dict = Dict({'full':  [],
                          'train': [],
                          'val':   [],
                          'test':  []})
    if create_new_dataset:
        X_train_test_val = get_data(Config, path_to_velodyne)
        X = gen_fl.create_txt_tain_val_lists(path_to_filelist,
                                              X_train_test_val,
                                              config_param,
                                              validation=validation_data)
        datasets_dict.full  = X_train_test_val
        datasets_dict.train = X[0] if create_new_dataset else []
        datasets_dict.val   = [] if not validation_data or not create_new_dataset else X[1]
        datasets_dict.test  = X[-1]

    else:
        filenames = ['train.txt', 'val.txt', 'test.txt']
        filedict  = {name.split('.')[0]: os.path.join(path_to_filelist, name)
                     for name in filenames if os.path.exists(os.path.join(path_to_filelist, name))}
        if not filedict:
            raise ValueError('Files with data not found.')
        for data_name in filedict:
            if data_name in datasets_dict:
                datasets_dict[data_name] = read_list_of_files(filedict[data_name])
                datasets_dict.full.extend(datasets_dict[data_name])

    print(f"Dataset size:", end=' ')
    for dataset_name in datasets_dict:
        print(f"{dataset_name}: {len(datasets_dict[dataset_name])}", end=' ')

    KittiStructure = KittiDataset if not Config.dataset.chinese else KITTI
    datasets_kitti = {dataset_name: KittiStructure(datasets_dict[dataset_name])
                      for dataset_name in datasets_dict if datasets_dict[dataset_name] != []}

    print("Dataset was prepared")
    #
    ### Neural Network
    print("Start to load dataset...")
    batch_size = Config.network.batch_size #* torch.cuda.device_count()

    data_loaders = Dict()
    for dataset_name in datasets_kitti:
        if dataset_name == 'full': pass
        data_loaders[dataset_name] = datasets_kitti[dataset_name].load_dataset(batch_size)
    return datasets_dict.full, data_loaders
