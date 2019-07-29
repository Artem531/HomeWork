from utils.utils import load_config
from collections import namedtuple
import json
import os


def set_default_paths():
    """
    Return main default paths
    :return: config, model path, filelist, clouds folder, labels folder, valeo flag
    """
    config        = ".\\configs\\config.json"
    model_path    = "weights\\model_0040.pth"
    filelist      = ".\\val.json"
    clouds_folder = "clouds"
    labels_folder = "labels"
    valeo_flag    = True
    return config, model_path, filelist, clouds_folder, labels_folder, valeo_flag


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
                names.append(line[:-1] + '.bin')
    elif name == "json":
        with open(filename, 'r') as f:
            data = json.load(f)
        for line in data['files']:
            names.append(line + '.bin')
    return names


def unpack_paths_dict(pdict, print_warning_flag=False):
    """
    Unpack paths dictionary
    :param pdict: dictionary with paths
    :param print_warning_flag:optional; print warnings flag
    :return: loaded config object, model path, filelist for processing, clouds folder, labels folder, valeo flag
    """
    config, model_path, filelist, clouds_folder, labels_folder, valeo_flag = get_main_paths()
    list_names = ['config', 'model_path', 'filelist', 'clouds_folder', 'labels_folder', 'valeo_flag']
    list_vars  = [config,    model_path,   filelist,   clouds_folder,   labels_folder,   valeo_flag]
    for idx in range(len(list_names)):
        try:
            list_vars[idx] = pdict[list_names[idx]]
        except KeyError:
            if print_warning_flag:
                print(f"Warning: {list_names[idx]} the parameter isn't set and will be set by default")

    config     = load_config(list_vars[0])
    model_path = list_vars[1]
    if list_vars[2] is not None:
        filelist = read_list_of_files(list_vars[2])
    else:
        filelist = os.listdir(list_vars[3])
    clouds_folder = list_vars[3]
    labels_folder = list_vars[4]
    valeo_flag    = list_vars[5]
    return config, model_path, filelist, clouds_folder, labels_folder, valeo_flag


def get_main_paths(args=None):
    """
    Create namedtuple with path from arguments of set default values
    :param args: optional; arguments from parser
    :return: namedtuple
    """
    if args is not None:
        config        = args.config_file
        model_path    = args.model
        filelist      = args.test_file
        clouds_folder = args.clouds_folder
        labels_folder = args.labels_folder
        valeo_flag    = args.valeo_flag
    else:
        config, model_path, filelist, clouds_folder, labels_folder, valeo_flag = set_default_paths()
    Paths = namedtuple('Paths', ['config', 'model_path', 'filelist', 'clouds_folder', 'labels_folder', 'valeo_flag'])
    main_paths = Paths(config, model_path, filelist, clouds_folder, labels_folder, valeo_flag)
    return main_paths




