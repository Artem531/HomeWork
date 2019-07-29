import os
import numpy as np
import json
from augmentation.utils.object_transformation import get_angle


def save_cloud(out_path, cloud_name, cloud):
    """
    Save point clouud as bin file
    :param cloud_name: name of file
    :param cloud: point cloud
    """
    cloud_path = os.path.join(out_path, "aug_clouds")
    cloud.astype(dtype=np.float32).tofile(os.path.join(cloud_path, cloud_name))


def save_labels(out_path, name, labels: list, valeo_flag):
    """
    Save labels as json file
    :param name: name of file
    :param labels: list of ObjectInfo objects
    :param valeo_flag: if True - valeo dataset else - kitti
    """
    label_path = os.path.join(out_path, "aug_labels")
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
                list_info.extend([0] * 7)
                list_info.append(label.bbox_size[0])
                list_info.append(label.bbox_size[2])
                list_info.append(label.bbox_size[1])
                list_info.append(-label.bbox_center[1])
                list_info.append(label.bbox_center[2])
                list_info.append(label.bbox_center[0])
                list_info.append(-get_angle(label.cos_t, label.sin_t))
                list_info = map(str, list_info)
                s1 = ' '.join(list_info)
                list_info_labels.append(s1)
            result_str = '\n'.join(list_info_labels)
            outfile.write(result_str)
