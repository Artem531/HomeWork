import numpy as np
import math


class ObjectInfo(object):
    """ Main object structure to be updated all through code """

    def __init__(self, bbox, min_h=0, max_h=2):
        """ Param info
        bbox =[center_x, center_y, size_x, size_y, cos, sin]
        """
        self.classname = "not_set"
        self.points = None
        self.bbox_center = [bbox[0], bbox[1], (max_h + min_h)/2]
        self.bbox_size   = [bbox[2], bbox[3],  max_h - min_h]
        self.cos_t = bbox[4]
        self.sin_t = bbox[5]
        self.bbox = get_bbox_properties(self, min_h, max_h)

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

    def get_bbox_properties(obj, min_z=-2, max_z=1):
        """
        """
        rear_left_x = obj.bbox_center[0] - obj.bbox_size[0] / 2 * obj.cos_t - obj.bbox_size[1] / 2 * obj.sin_t
        rear_left_y = obj.bbox_center[1] - obj.bbox_size[0] / 2 * obj.sin_t + obj.bbox_size[1] / 2 * obj.cos_t
        rear_right_x = obj.bbox_center[0] - obj.bbox_size[0] / 2 * obj.cos_t + obj.bbox_size[1] / 2 * obj.sin_t
        rear_right_y = obj.bbox_center[1] - obj.bbox_size[0] / 2 * obj.sin_t - obj.bbox_size[1] / 2 * obj.cos_t
        front_right_x = obj.bbox_center[0] + obj.bbox_size[0] / 2 * obj.cos_t + obj.bbox_size[1] / 2 * obj.sin_t
        front_right_y = obj.bbox_center[1] + obj.bbox_size[0] / 2 * obj.sin_t - obj.bbox_size[1] / 2 * obj.cos_t
        front_left_x = obj.bbox_center[0] + obj.bbox_size[0] / 2 * obj.cos_t - obj.bbox_size[1] / 2 * obj.sin_t
        front_left_y = obj.bbox_center[1] + obj.bbox_size[0] / 2 * obj.sin_t + obj.bbox_size[1] / 2 * obj.cos_t

        bbox = np.asarray([[rear_left_x, rear_left_y, min_z],
                           [rear_left_x, rear_left_y, max_z],
                           [rear_right_x, rear_right_y, max_z],
                           [rear_right_x, rear_right_y, min_z],
                           [front_left_x, front_left_y, min_z],
                           [front_left_x, front_left_y, max_z],
                           [front_right_x, front_right_y, max_z],
                           [front_right_x, front_right_y, min_z]])
        return bbox


def get_bbox_properties(obj, min_z=-2, max_z=1):
    """
    """
    rear_left_x   = obj.bbox_center[0] - obj.bbox_size[0]/2 * obj.cos_t - obj.bbox_size[1]/2 * obj.sin_t
    rear_left_y   = obj.bbox_center[1] - obj.bbox_size[0]/2 * obj.sin_t + obj.bbox_size[1]/2 * obj.cos_t
    rear_right_x  = obj.bbox_center[0] - obj.bbox_size[0]/2 * obj.cos_t + obj.bbox_size[1]/2 * obj.sin_t
    rear_right_y  = obj.bbox_center[1] - obj.bbox_size[0]/2 * obj.sin_t - obj.bbox_size[1]/2 * obj.cos_t
    front_right_x = obj.bbox_center[0] + obj.bbox_size[0]/2 * obj.cos_t + obj.bbox_size[1]/2 * obj.sin_t
    front_right_y = obj.bbox_center[1] + obj.bbox_size[0]/2 * obj.sin_t - obj.bbox_size[1]/2 * obj.cos_t
    front_left_x  = obj.bbox_center[0] + obj.bbox_size[0]/2 * obj.cos_t - obj.bbox_size[1]/2 * obj.sin_t
    front_left_y  = obj.bbox_center[1] + obj.bbox_size[0]/2 * obj.sin_t + obj.bbox_size[1]/2 * obj.cos_t

    bbox = np.asarray([[rear_left_x,   rear_left_y,   min_z],
                       [rear_left_x,   rear_left_y,   max_z],
                       [rear_right_x,  rear_right_y,  max_z],
                       [rear_right_x,  rear_right_y,  min_z],
                       [front_left_x,  front_left_y,  min_z],
                       [front_left_x, front_left_y , max_z],
                       [front_right_x, front_right_y, max_z],
                       [front_right_x, front_right_y, min_z]])
    return bbox
