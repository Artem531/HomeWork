import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import datetime



from utils.utils import plot_label_map
from augmentation.utils.vis_3d import visualize
from augmentation.detector.pixor_detector import PixorDetector
from augmentation.utils import paths_manager as pm
from augmentation.utils import load_annotation
from argparse import ArgumentParser
from utils.utils import iou_bboxes
from itertools import product

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = ArgumentParser()

# parser.add_argument('-cfg',
#                     '--config_file',
#                     type=str,
#                     default=os.path.join(".\\configs\\config.json"),
#                     help="Path to config file")
# parser.add_argument('-pcs',
#                     '--clouds_folder',
#                     type=str,
#                     default=os.path.join(r"E:\Docs\Datasets\valeo_dynamic_obj_2019_dataset", "clouds"),
#                     help="Path to folder with poinclouds")
# parser.add_argument('-l',
#                     '--labels_folder',
#                     type=str,
#                     default=os.path.join(r"E:\Docs\Datasets\valeo_dynamic_obj_2019_dataset", "labels"),
#                     help="Path to label folder (can be None)")
# parser.add_argument('-s',
#                     '--test_file',
#                     type=str,
#                     default=r"E:\Docs\Datasets\valeo_dynamic_obj_2019_dataset\val.json",
#                     help="Path to for come txt file with filenames fo test (without extensions)")
# parser.add_argument('-m',
#                     '--model',
#                     type=str,
#                     default=r"E:\Docs\Datasets\valeo_dynamic_obj_2019_dataset\weights\model_0040.pth",
#                     help="Path to model .pth file")

# parser.add_argument('-vf',
#                     '--valeo_flag',
#                     type=str,
#                     default=True,
#                     help="")

parser.add_argument('-cfg',
                    '--config_file',
                    type=str,
                    default=os.path.join("..\\configs\\config.json"),
                    help="Path to config file")
parser.add_argument('-pcs',
                    '--clouds_folder',
                    type=str,
                    default=r"E:\Docs\Datasets\kitti\lidar_pixor\data_object_velodyne\training\velodyne",
                    help="Path to folder with poinclouds")
parser.add_argument('-l',
                    '--labels_folder',
                    type=str,
                    default=r"E:\Docs\Datasets\kitti\lidar_pixor\data_object_label_2\training\label_2",
                    help="Path to label folder (can be None)")
parser.add_argument('-s',
                    '--test_file',
                    type=str,
                    default=r"E:\Docs\Datasets\kitti\val.txt",
                    help="Path to for come txt file with filenames fo test (without extensions)")
parser.add_argument('-m',
                    '--model',
                    type=str,
                    default=r"E:\Docs\Datasets\kitti\weights_kitti\model_0040.pth",
                    # default=r"D:\Docs\Tasks\428_postprocess_filtration\test_model_weights\validate_sum_loss.pth",
                    help="Path to model .pth file")

parser.add_argument('-vf',
                    '--valeo_flag',
                    type=str,
                    default=False,
                    help="")


def get_metrics_auto(y_true, y_score):
    """
    Calculation precision, recall, average precision metrics.
    :param y_true: binary labels
    :param y_score: Estimated probabilities or decision function.
    :return: precision, recall, list of thresholds, average precision metrics.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    return precision, recall, thresholds, average_precision


def matching_bboxes(gr_list, pred_list, threshold):
    """
    Create objects pairs list
    :param gr_list: list of ground truth labels
    :param pred_list: list of predicted labels
    :param threshold: list of thresholds
    :return: list of tuples with matched objects
    """
    matched_list = []
    combination_mat = np.array(list(product(gr_list, pred_list))).reshape((len(gr_list), len(pred_list), 2))

    iou = lambda obj1, obj2: iou_bboxes(obj1, obj2)
    foo = np.vectorize(iou)
    iou_scores = foo(combination_mat[:, :, 0], combination_mat[:, :, 1])
    iou_scores[iou_scores < threshold] = 0
    idx_col = 0
    while combination_mat.size > 0 and idx_col < combination_mat.shape[0]:
        col = iou_scores[idx_col]
        max_iou = np.max(col)
        idx_row = np.where(col == max_iou)
        if max_iou > 0:
            pair = combination_mat[idx_col][idx_row][0].tolist()
            pair[0], pair[1] = pair[1], pair[0]
            sample = [max_iou, tuple(pair)]
            matched_list.append(sample)
            combination_mat = np.delete(combination_mat, idx_col, axis=0)
            combination_mat = np.delete(combination_mat, idx_row, axis=1)

            iou_scores = np.delete(iou_scores, idx_col, axis=0)
            iou_scores = np.delete(iou_scores, idx_row, axis=1)
            idx_col -= 1
        idx_col += 1

    if combination_mat.size > 0:
        gr = combination_mat[:, 0, 0]
        pr = combination_mat[0, :, 1]
        matched_list.extend([[0, (None, gro)] for gro in gr])
        matched_list.extend([[0, (pro, None)] for pro in pr])

    try:
        result_list = np.array(matched_list)[:, 1]
    except IndexError:
        result_list = [(0, 0)]
    return result_list


def get_objects_detection_results(gr_list_obj, pred_list_obj, scores, threshold=0.5):
    """
    Create lists of binary labels and estimated probabilities.
    :param gr_list_obj: list of ground truth labels
    :param pred_list_obj: list of predicted labels
    :param scores: scores of predicted labels
    :param threshold: iou valid threshold
    :return: binary labels and estimated probabilities.
    """
    matched_list = matching_bboxes(gr_list_obj, pred_list_obj, threshold) #[(pred, gr),...]

    y_true   = []
    y_scores = []
    for pair in matched_list:
        if None not in pair:
            y_true.append(1)
            p = pair[0]
            idx = pred_list_obj.index(p)
            score = scores[idx]
            y_scores.append(score)
        elif pair[0] is None and pair[1] is not None:
            y_true.append(1)
            y_scores.append(0)
        elif pair[1] is None and pair[0] is not None:
            y_true.append(0)
            score = scores[pred_list_obj.index(pair[0])]
            y_scores.append(score)
    return y_true, y_scores


class Metrics:
    """
    Class for store metrics
    """
    def __init__(self, thresholds, precision, recall, average_precision):
        self.precision_dynamic = precision[1:]
        self.recall_dynamic = recall[1:]
        self.average_precision = average_precision
        self.thresholds = thresholds

    def build_general_plot(self, iou_threshold, save_flag=True):
        """
        Build precision-recall curve
        :param iou_threshold: min iou valid value
        :param save_flag: optional; flag for save plots
        """
        plt.plot(self.recall_dynamic, self.precision_dynamic, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall curve: AP={self.average_precision: {0}.{2}}, IoU={iou_threshold}')
        list_chars = [" ", ":", "-"]
        time_now = str(datetime.datetime.now())
        for elem in list_chars:
             time_now = time_now.replace(elem, "_")
        from sklearn.utils import fixes

        step_kwargs = ({'step': 'post'}
                       if 'step' in fixes.signature(plt.fill_between).parameters
                       else {})
        plt.step(self.recall_dynamic, self.precision_dynamic, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(self.recall_dynamic, self.precision_dynamic, alpha=0.2, color='b', **step_kwargs)
        plt.legend((f"Precision = {round(self.precision_dynamic[0], 2)}",
                    f"Recall = {round(self.recall_dynamic[0], 2)}"),
                   loc="upper right", fontsize='small')
        if save_flag:
            out_path = os.path.normpath("..\\out_files")
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            plt.savefig(os.path.join(out_path, f"precision_recall_{time_now[:-10]}.png"))

        plt.show()


def process_cloud(cloud_name, Detector, paths_dict,
                  threshold=0.5, visualize_flag=False):
    """
    Get prediction for cloud
    :param cloud_name: cloud filename
    :param Detector: Detector class object
    :param paths_dict: dictionary with main paths(clouds and labels paths, model path, configuration file, test file)
    :param threshold: min value of iou
    :param visualize_flag: flag for visualization cloud with labels
    :return: cloud, label_list, predicted_objects, y_true, y_scores
    """
    clouds = []
    frames_objs = []
    cloud, label_list = load_annotation.get_processed_cloud(cloud_name, paths_dict)
    # cloud, label_list = load_annotation.get_source_cloud(cloud_name, paths_dict)
    min_points = 10
    if cloud is not None and cloud.shape[0] >= min_points:
        pc_feature, predicted_objects, heatmap, scores = Detector.detect_birdview(cloud,
                                                                                  return_heatmap=True,
                                                                                  enable_3d_postproc=True)

        clouds.append(cloud)
        if predicted_objects is not None:
            if visualize_flag:
                visualize([cloud], [label_list], predicted_objects)
                plot_label_map(heatmap)
            frames_objs.append(predicted_objects)
            # bev_image = get_bev_image(pc_feature,
            #                          config['voxel_size'],
            #                          label_list=predicted_objects,
            #                          gt_label_list=label_list,
            #                          map_height=config['input_shape'][0])
            # cv2.imshow('Detection_results', np.zeros((200, 200)))
            y_true, y_scores = get_objects_detection_results(label_list,
                                                             predicted_objects,
                                                             scores, threshold=threshold)
            return cloud, label_list, predicted_objects, y_true, y_scores
        else:
            frames_objs.append([])

    return [None]*5


def process_model(paths_dict,
                  visualize_flag=False,
                  threshold=0.5):
    """
    Calculate metrics for model
    :param valeo_flag: if True - valeo dataset else - kitti
    :param paths_dict: dictionary with main paths(clouds and labels path, model path, configuration file, test file, valeo_falg)
    :param visualize_flag: flag for visualization cloud with labels
    :param threshold: min value of iou
    :return: object Metrics class
    """
    y_scores_list = []
    y_true_list = []

    config, model_path, filelist, clouds_folder, labels_folder, valeo_flag = pm.unpack_paths_dict(paths_dict)

    Detector = PixorDetector(config, model_path, cpu_flag=False)
    for pc_filename in tqdm(filelist[:30]):
        cloud, label_list, predicted_objects, y_true, y_scores = process_cloud(pc_filename, Detector, paths_dict,
                                                                               threshold=threshold,
                                                                               visualize_flag=visualize_flag)
        if cloud is not None and y_true is not None:
            y_scores_list.extend(y_scores)
            y_true_list.extend(y_true)

    precision, recall, thresholds, average_precision = get_metrics_auto(y_true_list, y_scores_list)
    metric_obj = Metrics(thresholds, precision, recall, average_precision)

    return metric_obj


def main():
    iou_threshold = 0.5
    args = parser.parse_args()

    config, model_path, filelist, clouds_folder, labels_folder, valeo_flag = pm.get_main_paths(args)

    # ------------
    paths_dict = {"config":        config,
                  "model_path":    model_path,
                  "filelist":      filelist,
                  "clouds_folder": clouds_folder,
                  "labels_folder": labels_folder,
                  "valeo_flag":    valeo_flag}

    metric_obj = process_model(paths_dict,
                               visualize_flag=True,
                               threshold=iou_threshold)
    metric_obj.build_general_plot(iou_threshold)


if __name__ == '__main__':

    main()
