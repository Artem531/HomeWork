import os
from pathlib import Path
from argparse import ArgumentParser
from config.config_dict import get_default_config
from infer.inference import create_kpi_json

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
                    default="/home/artem/MAIN/test_dataset/clouds/training",
                    help="Path to folder with poinclouds")
parser.add_argument('-l',
                    '--labels_folder',
                    type=str,
                    default=r"/home/artem/MAIN/test_dataset/label/training",

                    help="Path to label folder (can be None)")
parser.add_argument('-s',
                    '--test_file',
                    type=str,
                    default=".\\train_val_info",
                    help="Path to for come txt file with filenames fo test (without extensions)")
parser.add_argument('-m',
                    '--model',
                    default="..\\model_weights",#\9_july_pixor_models",
                    help="Path to model .pth file")


if __name__ == '__main__':
    args = parser.parse_args()
    from data_processing.prepare_dataset import prepare_data

    _, data_loaders = prepare_data(Config, args.test_file, args.clouds_folder, create_new_dataset=False)
    create_kpi_json(data_loaders.test,
                    threshold=0.5,
                    # date_time='2019-07-08')
                    model_path=Path(args.model))
