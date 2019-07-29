"""
    Generate train.txt or .json file which contains a list of training indices.
"""
import json
import argparse
import os
from  sklearn.model_selection import train_test_split
from addict import Dict


def split_dataset(filelist, ratio, validation=False, valid_ratio=0.5, random_state=42):
    filelist = filelist[18:]
    X_train, X_test_val = train_test_split(filelist,
                                           test_size=ratio,
                                           shuffle=True,
                                           random_state=random_state)
    if validation:
        X_test, X_val = train_test_split(X_test_val,
                                         test_size=valid_ratio,
                                         shuffle=True,
                                         random_state=random_state)
        return X_train, X_val, X_test
    return X_train, X_test_val


def create_txt_tain_val_lists(out_path, files, config_param, validation=False):
    """
    Create .txt file with list of filename for train and test model
    :param out_path: path to out file
    :param files: list of labels filename
    :param config_param: Dict() object with fields random_state, test_val_size
    :param validation: optional; create validation dataset
    """
    files = list(map(lambda name: name.split('.')[0], files))
    output_file_train = os.path.join(out_path, 'train.txt')
    output_file_val   = os.path.join(out_path, 'test.txt')
    output_file_list = [output_file_train, output_file_val]
    if validation:
        output_file_test = os.path.join(out_path, 'val.txt')
        output_file_list.append(output_file_test)

    datasets = split_dataset(files, ratio=config_param.test_val_size,
                                    validation=validation,
                                    valid_ratio=0.5,
                                    random_state=config_param.random_state)

    for idx in range(len(output_file_list)):
        with open(output_file_list[idx], 'w') as out_file:
            res_lines = "\n".join(datasets[idx])
            out_file.write(res_lines)
    return datasets


def create_json_tain_val_lists(out_path, files,  config_param, validation=False):
    """
    Create .json file with list of filename for train and test model
    :param out_path: path to out file
    :param files: list of labels filename
    :param config_param: Dict() object with fields random_state, test_val_size
    :param validation: optional; create validation dataset
    """
    files = list(map(lambda name: name.split('.')[0], files))
    output_file_train = os.path.join(out_path, 'train.json')
    output_file_val   = os.path.join(out_path, 'test.json')
    output_file_list = [output_file_train, output_file_val]
    if validation:
        output_file_test = os.path.join(out_path, 'val.json')
        output_file_list.append(output_file_test)

    datasets = split_dataset(files, ratio=config_param.test_val_size,
                                    validation=validation,
                                    valid_ratio=0.5,
                                    random_state=config_param.random_state)
    json_dicts = [{"files": filelist, "number_of_files": len(filelist)} for filelist in datasets]

    for idx in range(len(output_file_list)):
        with open(output_file_list[idx], 'w') as out_file:
            json.dump(json_dicts[idx], out_file, indent=4, sort_keys=True)
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Generate training idx.')
    #TODO: ratio from config
    parser.add_argument('--train_ratio',         type=float, default=0.70, help='Training ratio.')
    parser.add_argument('--path_to_annotations', type=str,   default=r"E:\Docs\Datasets\pixor\data_object_label_2\training\label_2")
    parser.add_argument('--path_to_train_val',   type=str,   default='..\\train_val_info')
    parser.add_argument('--validation_dataset',  type=bool,  default=True)
    parser.add_argument('--file_format',         type=str,   default='txt',
                        choices=['json', 'txt'], help="Dataset formats")

    args = parser.parse_args()

    if not os.path.exists(args.path_to_train_val):
        os.makedirs(args.path_to_train_val)

    files = [f for f in os.listdir(args.path_to_annotations) if os.path.isfile(os.path.join(args.path_to_annotations, f))]
    train_ratio, path_to_train_val = args.train_ratio, args.path_to_train_val
    validation = args.validation_dataset
    config_param = Dict()
    config_param.random_state = 42
    config_param.test_val_size = 1 - train_ratio

    if args.file_format == "txt":
        create_txt_tain_val_lists(path_to_train_val, files,  config_param,  validation=validation)
    elif args.file_format == "json":
        create_json_tain_val_lists(path_to_train_val, files, config_param, validation=validation)
    else:
        raise ValueError("Invalid format")
