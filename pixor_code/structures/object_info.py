import numpy as np
import math
from addict import Dict
import json

class ObjectInfo(object):
    """ Main object structure to be updated all through code """

    def __init__(self, bbox, min_h=0, max_h=2):
        """ Param info
        :param bbox: [center_x, center_y, size_x, size_y, cos, sin]
        """
        self.classname = "not_set"
        self.points = None
        self.bbox_center = [bbox[0], bbox[1], (max_h + min_h)/2]
        self.bbox_size   = [bbox[2], bbox[3],  max_h - min_h]
        self.cos_t = bbox[4]
        self.sin_t = bbox[5]
        self.bbox = self.get_bbox_properties(min_h, max_h)

    def __str__(self):
        return f"\nObjectInfo: box_center: {self.bbox_center}  box_size: {self.bbox_size}, "\
               f"cos_t: {self.cos_t}, sin_t: {self.sin_t}, angle: {math.degrees(self.get_box_angle())}\n"

    def __repr__(self):
        return self.__str__()

    def get_box_angle(self):
        """
        Calculate angle (0..2pi)
        :return: angle in radians
        """
        a_cos = math.acos(self.cos_t)
        angle = a_cos if self.sin_t >= 0 else -a_cos + 2 * math.pi
        return angle

    def annotation_to_dict(self):
        """
        Helper for dumping to json
        Returns:
             serializes part of the structure to dict
        """
        out_dict = {'bbox_center': self.bbox_center,
                    'bbox_size': np.array(self.bbox_size, dtype="float64").tolist(),
                    'rotation': self.get_box_angle(),
                    'label': self.classname}
        return out_dict

    def get_bbox_properties(self, min_z=-2, max_z=1):
        """
        """
        rear_left_x = self.bbox_center[0] - self.bbox_size[0] / 2 * self.cos_t - self.bbox_size[1] / 2 * self.sin_t
        rear_left_y = self.bbox_center[1] - self.bbox_size[0] / 2 * self.sin_t + self.bbox_size[1] / 2 * self.cos_t
        rear_right_x = self.bbox_center[0] - self.bbox_size[0] / 2 * self.cos_t + self.bbox_size[1] / 2 * self.sin_t
        rear_right_y = self.bbox_center[1] - self.bbox_size[0] / 2 * self.sin_t - self.bbox_size[1] / 2 * self.cos_t
        front_right_x = self.bbox_center[0] + self.bbox_size[0] / 2 * self.cos_t + self.bbox_size[1] / 2 * self.sin_t
        front_right_y = self.bbox_center[1] + self.bbox_size[0] / 2 * self.sin_t - self.bbox_size[1] / 2 * self.cos_t
        front_left_x = self.bbox_center[0] + self.bbox_size[0] / 2 * self.cos_t - self.bbox_size[1] / 2 * self.sin_t
        front_left_y = self.bbox_center[1] + self.bbox_size[0] / 2 * self.sin_t + self.bbox_size[1] / 2 * self.cos_t

        bbox = np.asarray([[rear_left_x, rear_left_y, min_z],
                           [rear_left_x, rear_left_y, max_z],
                           [rear_right_x, rear_right_y, max_z],
                           [rear_right_x, rear_right_y, min_z],
                           [front_left_x, front_left_y, min_z],
                           [front_left_x, front_left_y, max_z],
                           [front_right_x, front_right_y, max_z],
                           [front_right_x, front_right_y, min_z]])
        return bbox


def create_hada_json():
    pass

def save_json_annottions(annotations, scores, anno_name):
    """

    :param annotations: ObjectInfo list
    :return:
    """
    dict_pattern = {"objects": []}
    object_pattern = {"bbox": [],
                      "classname": '',
                      "contours": [],
                      "mask": None,
                      "metaname": 'dont_care',
                      "score": 0}
    for idx in range(len(annotations)):
        obj = annotations[idx]
        score = scores[idx]
        c = np.array(obj.bbox_center[:2])
        s = np.array(obj.bbox_size[:2])/2
        object_pattern['bbox'] = [*(c - s).tolist(), *(c + s).tolist()]
        object_pattern['classname'] = obj.classname
        object_pattern['score'] = score
        p1 = c - s
        p2 = np.array([c[0] + s[0], c[1] - s[1]])
        p3 = c + s
        p4 = np.array([c[0] - s[0], c[1] + s[1]])
        object_pattern["contours"] = list(map(lambda x: x.tolist(), [p1, p2, p3, p4]))

        dict_pattern["objects"].append(object_pattern)
        print(object_pattern)
    with open(anno_name.json, 'w') as out_file:
        json.dump(dict_pattern, out_file)


def create_from_src_anno(anno):
    anno = Dict(anno)
    centers = [anno.bbox3d.shifts.squeeze_().cpu().numpy()[2], anno.bbox3d.shifts.squeeze_().cpu().numpy()[0]]
    w = anno.bbox3d.width.item()
    l = anno.bbox3d.length.item()
    size = [l, w]
    c = np.cos(anno.bbox3d.yaw.item())
    s = np.sin(anno.bbox3d.yaw.item())
    return ObjectInfo([*centers, *size, c, s])


def create_from_predict(predict):
    return ObjectInfo(predict)


