import numpy as np
import cv2

from utils.visualize_utils import build_box_plot, plot_label_map,  visualize
from config.config_dict import get_default_config
from structures.object_info import ObjectInfo
Config = get_default_config()


def heatmap2rect(heatmap):
	heatmap_coords = np.where(heatmap > Config.post_proc.cls_threshold_val)
	if not heatmap_coords:
		return []
	rect_list = []
	w, l = Config.geometry.grid_size, Config.geometry.grid_size
	for idx in range(len(heatmap_coords[0])):
		c = heatmap_coords[1][idx] * Config.geometry.grid_size, \
			heatmap_coords[0][idx] * Config.geometry.grid_size - 40
		obj = ObjectInfo([*c, w, l, 0, 1])
		rect_list.append(obj)
	return rect_list



def match_label_heatmap(heatmap, pcloud, label_list, pred_list):
	print(f"heatmap sizes = {heatmap.shape}\npcloud: {pcloud.shape}")

	# visualize([pcloud], [label_list], pred_list)
	heatmap = np.squeeze(heatmap, 0)
	rect_list = heatmap2rect(heatmap)
	visualize([pcloud], [rect_list], pred_list)
	# exit()



