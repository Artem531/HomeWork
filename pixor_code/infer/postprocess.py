'''
Non Max Suppression
IOU, Recall, Precision, Find overlap and Average Precisions
Source Code is adapted from github.com/matterport/MaskRCNN

'''

import numpy as np
from shapely.geometry import Polygon
import shapely


def get_corners(bbox):
    centre_x = bbox[0]
    centre_y = bbox[1]
    l = bbox[2]
    w = bbox[3]
    cos_t = bbox[4]
    sin_t = bbox[5]
    rear_left_x = centre_x - l / 2 * cos_t - w / 2 * sin_t
    rear_left_y = centre_y - l / 2 * sin_t + w / 2 * cos_t
    rear_right_x = centre_x - l / 2 * cos_t + w / 2 * sin_t
    rear_right_y = centre_y - l / 2 * sin_t - w / 2 * cos_t
    front_right_x = centre_x + l / 2 * cos_t + w / 2 * sin_t
    front_right_y = centre_y + l / 2 * sin_t - w / 2 * cos_t
    front_left_x = centre_x + l / 2 * cos_t - w / 2 * sin_t
    front_left_y = centre_y + l / 2 * sin_t + w / 2 * cos_t
    corners = [[rear_left_x, rear_left_y],
               [rear_right_x, rear_right_y],
               [front_right_x, front_right_y],
               [front_left_x, front_left_y]]
    return np.array(corners)


def convert_format(boxes_array, compute_corners=False):
    """
    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """
    if compute_corners:
        corners = [get_corners(box) for box in boxes_array]
        polygons = [Polygon([(corner[i, 0], corner[i, 1]) for i in range(4)]) for corner in corners]
    else:
        polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]

    return np.array(polygons)


def compute_iou(box, boxes):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    iou = []
    for b in boxes:
        try:
            union = box.union(b).area
            if  union > 0:
                iou.append(box.intersection(b).area / union)
            else:
                iou.append(1.)
        except ValueError:
            iou.append(1.)
        except shapely.errors.TopologicalError:
            iou.append(1.)

    return np.array(iou, dtype=np.float32)


def bbox_in_roi(bbox, Config):
    geometry = Config.geometry
    if bbox[0] < geometry.length_max \
            and bbox[0] > geometry.length_min \
                and bbox[1] < geometry.width_max \
                    and bbox[1] > geometry.width_min:
        return True
    return False


def correct_box_size(bbox, Config):
    if max(bbox[2], bbox[3]) < Config.post_proc.object_max_size_thr:
        return True
    return False


def filter_non_valid_boxes(boxes, scores, Config):
    filtered_boxes = []
    filtered_scores = []
    for i in range(len(boxes)):
        if np.sum(np.isinf(boxes[i])) == 0 \
                and bbox_in_roi(boxes[i], Config) \
                and correct_box_size(boxes[i], Config):
            filtered_boxes.append(boxes[i])
            filtered_scores.append(scores[i])
    return np.array(filtered_boxes), np.array(filtered_scores)


def non_max_suppression(boxes, scores, Config, calc_bbox_corners=False):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.

    return an numpy array of the positions of picks
    """
    boxes  = boxes.squeeze_(0).cpu().numpy()
    # scores = scores.squeeze_(0)


    threshold = Config.post_proc.nms_iou_threshold
    assert boxes.shape[0] > 0
    if boxes.dtype != "f":
        boxes = boxes.cpu().numpy()
    boxes, scores = filter_non_valid_boxes(boxes, scores, Config)

    polygons = convert_format(boxes.astype(np.float32), compute_corners=calc_bbox_corners)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)

        # Compute IoU of the picked box with the rest
        iou = compute_iou(polygons[i], polygons[ixs[1:]])

        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1

        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    boxes  = boxes [np.array(pick, dtype=np.int32)]
    scores = scores[np.array(pick, dtype=np.int32)]
    return boxes, scores
