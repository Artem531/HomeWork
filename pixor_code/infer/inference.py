import os
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import torch


from config.config_dict import get_default_config
from structures.object_info import create_from_src_anno, create_from_predict
from structures.metrics import Metrics
from utils.visualize_utils import build_box_plot, plot_label_map
from structures.addition_net_structures import Period, PixorModel
from infer.postprocess import non_max_suppression

from research_file_heatmap import match_label_heatmap

Config = get_default_config()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = ArgumentParser()


parser.add_argument('-cfg',
                    '--config_file',
                    type=str,
                    default=os.path.join("..\\configs\\config.json"),
                    help="Path to config file")
parser.add_argument('-pcs',
                    '--clouds_folder',
                    type=str,
                    # default=r"E:\Docs\Datasets\kitti\lidar_pixor\data_object_velodyne\training\velodyne",
                    default=r"E:\Docs\Datasets\pixor\data_object_velodyne\training\velodyne",
                    help="Path to folder with poinclouds")
parser.add_argument('-l',
                    '--labels_folder',
                    type=str,
                    # default=r"E:\Docs\Datasets\kitti\lidar_pixor\data_object_label_2\training\label_2",
                    default=r"E:\Docs\Datasets\pixor\data_object_label_2\training\label_2",

                    help="Path to label folder (can be None)")
parser.add_argument('-s',
                    '--test_file',
                    type=str,
                    default=r"E:\Docs\Datasets\kitti\val.txt",
                    help="Path to for come txt file with filenames fo test (without extensions)")
parser.add_argument('-m',
                    '--model',
                    type=str,
                    # default=r"E:\Docs\Datasets\kitti\weights_kitti\model_0040.pth",
                    # default=r"D:\Docs\Tasks\428_postprocess_filtration\test_model_weights\validate_sum_loss.pth",
                    # default=r"D:\Docs\Tasks\524_pixor_deep_dive\model_weights_devbox\test_sum_loss.pth",
                   default = r"D:\Docs\Tasks\575_main_version_pixor\model_weights\9_july_pixor_models"
                             r"\validate_sum_loss.pth",
                    help="Path to model .pth file")


def read_list_of_files(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()  # get rid of \n symbol
        names = []
        for line in lines[:-1]:
            names.append(line[:-1] + '.bin')
    return names


def process_predictions(cls_pred, reg_pred, cls_threshold_val=None):
    heatmap = cls_pred.cpu().numpy()

    cls_threshold = Config.post_proc.cls_threshold_val if cls_threshold_val is None else cls_threshold_val
    activation = cls_pred > cls_threshold
    num_boxes = int(activation.sum())
    if num_boxes == 0:
        print("No bounding box found")
    else:
        corners = torch.zeros((reg_pred.shape[0], num_boxes, reg_pred.shape[-1]))
        for i in range(reg_pred.shape[-1]):
            corners[..., i] = torch.masked_select(reg_pred[..., i], activation)


        scores = (torch.masked_select(cls_pred.unsqueeze(0), activation))
        predicted_boxes, scores = non_max_suppression(corners, scores, Config, calc_bbox_corners=True)

        return predicted_boxes, heatmap, scores
    return None, heatmap, None


def get_predictins(data_loaders_test,
                   period,
                   date_time='',
                   model_path=None,
                   show_heatmap=False):
    dataset_label_list        = []
    dataset_predicted_objects = []
    dataset_scores            = []
    heatmap_collection = []

    mt = PixorModel(Config, train_flag=False, date_time=date_time)

    mt.models_path = Path(model_path) if model_path is not None else mt.models_path
    addition = date_time if date_time == '' else '_' + date_time
    model_weights = mt.models_path / f'validate_sum_loss{addition}.pth'
    mt.load_model(model_weights)

    for (list_grid, list_output_class, list_output_reg, list_anno) in tqdm(data_loaders_test,
                                                      desc=f'iter ({period})',
                                                      total=len(data_loaders_test),
                                                      leave=False):
        for idx in range(len(list_grid)):
            # unpack augment data
            print(idx)
            grid, cls_target, reg_target, annos = list_grid[idx], list_output_class[idx], list_output_reg[idx], \
                                                  list_anno[idx]
            # the same inference
            if annos:
                cls_target, reg_target, cls_pred, reg_pred = mt.get_predict(grid, cls_target, reg_target)
                predicted_boxes, heatmap, scores = process_predictions(cls_pred, reg_pred)
                if predicted_boxes is not None:
                    heatmap_collection.append(heatmap)
                    dataset_label_list.append([create_from_src_anno(anno) for anno in annos])
                    dataset_predicted_objects.append([create_from_predict(box) for box in predicted_boxes])
                    dataset_scores.append([score.item() for score in scores])
                    # ---------------------------
                    # match_label_heatmap(heatmap_collection[-1],
                    #                     pcloud.squeeze(0).cpu().numpy(),
                    #                     dataset_label_list[-1],
                    #                     dataset_predicted_objects[-1])
                    # ---------------------------
                    if show_heatmap:
                        plot_label_map(heatmap)


    torch.cuda.empty_cache()
    return dataset_label_list, dataset_predicted_objects, dataset_scores, heatmap_collection


def create_kpi_json(data_loaders_test,
                    threshold=0.5,
                    date_time='',
                    model_path=None):
    period = Period.test

    dataset_label_list, dataset_predicted_objects, dataset_scores, _ = get_predictins(data_loaders_test,
                                                                                      period,
                                                                                      date_time=date_time,
                                                                                      model_path=model_path)
    #
    print("Predictions is ready.\nStart metric calculation.")
    from structures.object_info import save_json_annottions
    metric_obj = Metrics()
    for idx in tqdm(range(len(dataset_label_list))):
        # save_json_annottions(dataset_label_list[idx], dataset_scores[idx], anno_name)
        # exit()
        metric_obj.get_objects_detection_results(dataset_label_list[idx],
                                                 dataset_predicted_objects[idx],
                                                 dataset_scores[idx],
                                                 threshold)
        # build_box_plot(dataset_label_list[idx], dataset_predicted_objects[idx])
    print("Calculation...")
    metric_obj.get_metrics_auto()
    print("Metric was calculated.")

    metric_obj.build_general_plot(threshold, True)
    metric_obj.create_json_results(".\\result_metrics_pixor.json")
    print("JSON was saved.")

