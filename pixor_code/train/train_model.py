import torch
from tqdm import tqdm
import datetime
import os

from config.config_dict import get_default_config
from train import model_selection
from structures.addition_net_structures import Period, PixorModel
from data_processing.prepare_dataset import prepare_data


def train_net(Config, data_loaders):
    periods = [Period.train, Period.validate]
    mt = PixorModel(Config, train_flag=True)

    for epoch in tqdm(range(Config.network.epochs), desc='epoch'):
        step = Config.network.step

        for period in periods:
            mt.train_model(data_loaders, period)

        print(f'{epoch:05d} ::', end='')
        for period in periods:
            history = mt.summary[str(period)].history
            print(
                f'{period:>8} :: sum: {history.sum_loss[-1]:.5f} | '
                f'cls: {history.cls_loss[-1]:.5f} | '
                f'reg: {history.reg_loss[-1]:.5f} ::',
                end='')
        if epoch == step:
            step += Config.network.step_size
            mt.optimizer.param_groups[0]['lr'] *= Config.network.lr_multiplier

    torch.cuda.empty_cache()

    model_selection.get_best_model(mt, data_loaders)
    date_time = str(datetime.datetime.now())[:10]
    out_path = mt.models_path / date_time
    if not os.path.exists(out_path): os.mkdir(out_path)
    mt.model.load_state_dict(torch.load(out_path / f'validate_sum_loss_{date_time}.pth'))

    print("Model was trained")


if __name__ == "__main__":
    Config = get_default_config()

    path_to_filelist = r"D:\Docs\Tasks\575_main_version_pixor\train_val_info"
    path_to_velodyne = 'training/velodyne'
    _, data_loaders = prepare_data(Config, path_to_filelist, path_to_velodyne)
    # ## Train model
    train_net(Config, data_loaders)
