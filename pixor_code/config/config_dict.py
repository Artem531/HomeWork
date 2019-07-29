from addict import Dict
from pathlib import Path
import numpy as np

# TODO: удалить заглушки
def get_default_config(main_dataset_path='/home/artem/MAIN/test_dataset'):
    Config = Dict()
    Config.random_state = 42
    Config.use_npy = False

    Config.dataset.frame_range = None
    Config.dataset.path = Path(main_dataset_path).expanduser().absolute()
    Config.dataset.path_str = str(Config.dataset.path)

    # hack parameters
    Config.dataset.add_width  = 0.1  # hand-picked
    Config.dataset.add_height = 0.0  # hand-picked
    Config.dataset.add_length = 0.1  # hand-picked

    Config.settings.lru_cache_size = 10000

    Config.geometry.width_min  = -40.0  # x
    Config.geometry.width_max  =  40.0  # x
    Config.geometry.length_min =   0.0  # y
    Config.geometry.length_max =  70.0  # y
    Config.geometry.height_min =  -2.5  # z
    Config.geometry.height_max =   1.0  # z
    Config.geometry.discretization = 0.1
    Config.geometry.grid_size = 0.4#TODO: Найти и заменить

    Config.dataset.test_val_size = 0.30
    Config.dataset.remove_cls = -1
    # TODO: multiply class support
    Config.dataset.type2cls = {
        'DontCare': Config.dataset.remove_cls,
        'Misc': Config.dataset.remove_cls,

        'Car': 1,
        'Van': 1,
        'Truck': Config.dataset.remove_cls,
        'Tram': Config.dataset.remove_cls,

        'Pedestrian': Config.dataset.remove_cls,
        'Person_sitting': Config.dataset.remove_cls,

        'Cyclist': Config.dataset.remove_cls,
    }

    # https://matplotlib.org/api/colors_api.html
    Config.dataset.cls2color = ['g', 'c', 'b', 'y']

    # calculated for KittiDataset:
    Config.dataset.reg.mean = np.array([-0.0061, -0.0485, -0.0338, 0.1905, 0.5317, 1.4167])
    Config.dataset.reg.std = np.array([0.4638, 0.8839, 0.7861, 1.0861, 0.1485, 0.144])
    Config.dataset.reg.min = np.array([-1., -1., -3.448, -3.7617, 0.2151, 0.8286])
    Config.dataset.reg.max = np.array([1., 1., 3.4683, 3.233, 1.1019, 1.9473])

    # Complexity
    Config.dataset.complexity.easy.min_height = 40
    Config.dataset.complexity.easy.max_occlusion = 0
    Config.dataset.complexity.easy.max_truncated = 0.15
    Config.dataset.complexity.moderate.min_height = 25
    Config.dataset.complexity.moderate.max_occlusion = 1
    Config.dataset.complexity.moderate.max_truncated = 0.3
    Config.dataset.complexity.hard.min_height = 25
    Config.dataset.complexity.hard.max_occlusion = 2
    Config.dataset.complexity.hard.max_truncated = 0.5
    Config.dataset.complexity.min_height = [40, 25, 25]
    Config.dataset.complexity.max_occlusion = [0, 1, 2]
    Config.dataset.complexity.max_truncated = [0.15, 0.3, 0.5]

    Config.network.version = 1.00
    Config.network.input_shape = (800, 700, 36)  # lwh
    Config.network.output_class_shape = (200, 175, 1)  # Multi-task Learning
    Config.network.output_reg_shape = (200, 175, 6)  # Multi-task Learning
    Config.network.in_out_ratio = Config.network.input_shape[0] / Config.network.output_class_shape[0]

    Config.network.batch_size = 1
    Config.network.lr = 1e-3
    Config.network.lr_multiplier = 0.1
    Config.network.step = 50
    Config.network.step_size = 50
    Config.network.epochs = 1#Config.network.step + 2 * Config.network.step_size
    Config.network.weight_decay = 1e-5
    Config.network.momentum = 0.9

    Config.network.reg_loss_alpha = 1
    Config.network.device = 'cpu'


    Config.post_proc.cls_threshold_val = 0.5
    Config.post_proc.nms_iou_threshold = 0.001
    Config.post_proc.object_max_size_thr = 10

    Config.dataset.chinese = False
    return Config

