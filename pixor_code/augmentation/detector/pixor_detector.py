import torch
import numpy as np
from models.model import PIXOR
from utils.utils import get_model_name
from post_process.postprocess import non_max_suppression


class PixorDetector:

    def __init__(self, config, weights):
        print("Load model ...")
        self.config = config
        if torch.cuda.is_available():
            self.net = PIXOR(config).cuda()
            self.device = 'cuda'
        else:
            self.net = PIXOR(config).cpu()
            self.device = 'cpu'
        self.net.set_decode(True)
        # net = torch.nn.DataParallel(net)
        state_dict = torch.load(weights)
        self.net.load_state_dict(state_dict)
        print("Done!")

    def point_in_roi(self, point):
        if (point[0] - self.config['geometry'][2]) < 0.01 \
                or (self.config['geometry'][3] - point[0]) < 0.01:
            return False
        if (point[1] - self.config['geometry'][0]) < 0.01 \
                or (self.config['geometry'][1] - point[1]) < 0.01:
            return False
        if (point[2] - self.config['geometry'][4]) < 0.01 \
                or (self.config['geometry'][5] - point[2]) < 0.01:
            return False
        return True

    def lidar_preprocess(self, cloud):

        velo = cloud
        velo_processed = np.zeros(self.config['input_shape'], dtype=np.float32)
        intensity_map_count = np.zeros((velo_processed.shape[0], velo_processed.shape[1]))
        for i in range(velo.shape[0]):
            if self.point_in_roi(velo[i, :]):
                x = int((velo[i, 1] - self.config['geometry'][0]) / self.config['voxel_size'])
                y = int((velo[i, 0] - self.config['geometry'][2]) / self.config['voxel_size'])
                z = int((velo[i, 2] - self.config['geometry'][4]) / 0.1)
                #print("velo_processed shape = ", velo_processed.shape, x, y, z)
                velo_processed[x, y, z] = 1
                velo_processed[x, y, -1] += velo[i, 3]
                intensity_map_count[x, y] += 1
        velo_processed[:, :, -1] = np.divide(velo_processed[:, :, -1],
                                             intensity_map_count,
                                             where = intensity_map_count != 0)
        return velo_processed



    def detect_birdview(self, cloud, return_heatmap = False):

        pc_feature = torch.from_numpy(self.lidar_preprocess(cloud))
        pc_feature = pc_feature.to(self.device)
        predicted_boxes = None
        heatmap = None
        with torch.no_grad():
            prediction = self.net(pc_feature.unsqueeze(0)).squeeze_(0)
            cls_pred = prediction[..., 0]
            activation = cls_pred > self.config['cls_threshold']
            num_boxes = int(activation.sum())
            if num_boxes == 0:
                print("No bounding box found")
            else:
                #print(num_boxes)
                corners = torch.zeros((num_boxes, 8))
                for i in range(1, 9):
                    corners[:, i - 1] = torch.masked_select(prediction[..., i], activation)
                corners = corners.view(-1, 4, 2)
                scores = (torch.masked_select(prediction[..., 0], activation)).cpu().numpy()
                predicted_boxes = non_max_suppression(corners, scores, self.config)
            pc_feature = pc_feature.cpu().numpy()  # (800, 700, 36)
            if return_heatmap:
                heatmap = cls_pred.cpu().numpy()
        return pc_feature, predicted_boxes, heatmap
