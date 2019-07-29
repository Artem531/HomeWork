import os
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import datetime
import json
from itertools import product
from utils.utils import iou_bboxes


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
    # TODO: delete it
    # print(iou_scores)
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


class Metrics:
    """
    Class for store metrics
    """
    def __init__(self):
        self.type = "Car"
        self.precision_dynamic = None
        self.recall_dynamic    = None
        self.average_precision = None
        self.thresholds        = None
        self.iou_threshold     = None

        self.y_true = []
        self.y_scores = []

        self.general_dict = {}

    def get_metrics_auto(self):
        """
        Calculation precision, recall, average precision metrics.
        :param y_true: binary labels
        :param y_score: Estimated probabilities or decision function.
        :return: precision, recall, list of thresholds, average precision metrics.
        """
        y_true, y_score = self.y_true, self.y_scores
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        average_precision =  average_precision_score(y_true, y_score)
        self.precision_dynamic = precision[1:]
        self.recall_dynamic    = recall[1:]
        self.average_precision = average_precision
        self.thresholds = thresholds

        print("precision: ", precision[0])
        print("recall: ", recall[0])

        # return precision, recall, thresholds, average_precision
    def agregate_dict(self):
        pass

    def create_json_results(self, filename):
        obj_type = {"boxes": {
                               "cars":{
                                   f"{self.iou_threshold}":{
                                       "dont_care":{
                                           "AP": self.average_precision,
                                           "score":  list(map(float, self.y_scores)),
                                           "y_true": list(map(float, self.y_true)),
                                       }
                                   }
                               }
                            }
                    }
        result_structure = obj_type
        with open(filename, "w") as f:
           json.dump(result_structure, f, indent=2, sort_keys=True)
        print(f"JSON {filename} was saved!")

    def get_objects_detection_results(self, gr_list_obj, pred_list_obj, scores, threshold=0.5):
        """
        Create lists of binary labels and estimated probabilities.
        :param gr_list_obj: list of ground truth labels
        :param pred_list_obj: list of predicted labels
        :param scores: scores of predicted labels
        :param threshold: iou valid threshold
        :return: binary labels and estimated probabilities.
        """
        self.iou_threshold = threshold

        matched_list = matching_bboxes(gr_list_obj, pred_list_obj, threshold)  # [(pred, gr),...]

        y_true = []
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
        self.y_true.extend(y_true)
        self.y_scores.extend(y_scores)

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
        # from sklearn.utils.fixes import signature
        # step_kwargs = ({'step': 'post'}
        #                if 'step' in signature(plt.fill_between).parameters
        #                else {})
        plt.step(self.recall_dynamic, self.precision_dynamic, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(self.recall_dynamic, self.precision_dynamic, alpha=0.2, color='b')#, **step_kwargs)
        plt.legend((f"Precision = {round(self.precision_dynamic[0], 2)}",
                    f"Recall = {round(self.recall_dynamic[0], 2)}"),
                   loc="upper right", fontsize='small')
        if save_flag:
            out_path = os.path.normpath(".\\out_files")
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            plt.savefig(os.path.join(out_path, f"precision_recall_{time_now[:-10]}.png"))

        plt.show()


