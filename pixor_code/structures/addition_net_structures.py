from tqdm import tqdm
from addict import Dict
from pathlib import Path
import os
import datetime
import copy
import enum
import numpy as np
import torch
import torch.nn as nn

from net.pixor_model import SplitOutputPixor
from net.loss_il  import CustomLoss as CustomLoss_ildik
from net.loss_src import CustomLoss as CustomLoss_src

@enum.unique
class Period(enum.Enum):
    train = enum.auto()
    test = enum.auto()
    validate = enum.auto()
    full = enum.auto()

    def __str__(self):
        return self.name


class PixorModel:
    def __init__(self, Config, train_flag=True, decode_flag=True, date_time=''):
        self.config_params = Config
        self.optimizer = None
        self.periods   = None
        self.device    = None #Config.network.device
        self.model     = None

        args = train_flag if train_flag else train_flag, decode_flag
        self.initialization_model(Config, *args)
        self.summary      = self.initialization_summary()
        self.models_path  = self.create_out_folder(date_time=date_time)
        self.loss_fn      = self.set_loss_fn


    def initialization_model(self, Config, *args):
        #TODO: args -> kwargs
        model = SplitOutputPixor(Config)
        train_flag = args[0]
        if not train_flag:
            decode_flag = args[1]
            model.pixor.set_decode(decode_flag)
        self.device = 'cpu'
        # self.device = 'cuda: 0' if torch.cuda.is_available() else 'cpu'
        # if self.device == 'cuda: 0' and torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        self.model = model

    def initialization_summary(self):
        summary = Dict()
        summary.train.history.cls_loss = []
        summary.train.history.reg_loss = []
        summary.train.history.sum_loss = []
        summary.test = copy.deepcopy(summary.train)
        summary.validate = copy.deepcopy(summary.train)
        return summary

    def create_out_folder(self, date_time=''):
        current_path = Path.cwd()
        models_path = current_path / 'model_weights' / date_time

        models_path.mkdir(exist_ok=True)
        # for pth in models_path.glob('*.pth'):
        #     pth.unlink()  # remove
        return models_path

    def set_optimizer(self):
        # optimizer = torch.optim.SGD(net.parameters(),
        #                             lr=self.config_params.network.lr,
        #                             momentum=self.config_params.network.momentum,
        #                             weight_decay=self.config_params.network.weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config_params.network.lr,
                                          weight_decay=self.config_params.network.weight_decay,
                                          amsgrad=True)

    def set_periods(self, test=True):
        p = Period()
        self.periods = [p.train, p.test] if test else [p.train]

    def set_loss_fn(self, *args, chinese_ver=False):
        # TODO: for research
        if chinese_ver:
            loss_fn = CustomLoss_src(self.device)
        else:
            loss_fn = CustomLoss_ildik().to(self.device)
        return loss_fn(*args)

    def get_predict(self, grid, cls_target, reg_target):
        self.model.eval()
        requires_grad = False
        with torch.set_grad_enabled(requires_grad):
            grid = grid.to(self.device)
            cls_target = cls_target.to(self.device)
            reg_target = reg_target.to(self.device)
            cls_pred, reg_pred = self.model(grid)
        return cls_target, reg_target, cls_pred, reg_pred

    def load_model(self, model_weights):
        state_dict = torch.load(model_weights, map_location=torch.device('cpu'))
        self.model.to(self.device)
        self.model.load_state_dict(state_dict)

    def train_model(self, data_loaders, period):
        """
        :param data_loaders: Dict()
        :param period: Period object
        :return:
        """
        if period == Period.train:
            self.model.train()
            requires_grad = True
            dataset = data_loaders.train
        elif period == Period.test or period == Period.validate:
            self.model.eval()
            requires_grad = False
            dataset = data_loaders.test if period == Period.test else data_loaders.val
        else:
            raise NotImplementedError(period)
        self.set_optimizer()
        total_losses = np.zeros(3)

        for (grid, cls_target, reg_target, annos, pcloud) in tqdm(dataset,
                                                          desc=f'iter ({period})',
                                                          total=len(dataset),
                                                          leave=False):

            with torch.set_grad_enabled(requires_grad):
                grid = grid.to(self.device)
                cls_target = cls_target.to(self.device)
                reg_target = reg_target.to(self.device)

                cls_pred, reg_pred = self.model(grid)
                cls_loss, reg_loss = self.loss_fn(cls_pred, reg_pred, cls_target, reg_target)
                sum_loss = cls_loss + self.config_params.network.reg_loss_alpha * reg_loss

                if period == Period.train:
                    sum_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_losses += np.array([cls_loss.item(), reg_loss.item(), sum_loss.item()])

        total_losses /= len(dataset)

        period_str = str(period)
        self.summary[period_str].history.cls_loss.append(total_losses[0])
        self.summary[period_str].history.reg_loss.append(total_losses[1])
        self.summary[period_str].history.sum_loss.append(total_losses[2])

        for (k, v) in self.summary[period_str].history.items():
            loss_hist = self.summary[period_str].history[k]
            if min(loss_hist) == loss_hist[-1]:
                date_time = str(datetime.datetime.now())[:10]
                out_path = self.models_path / date_time
                if not os.path.exists(out_path): os.mkdir(out_path)
                torch.save(self.model.state_dict(), str(out_path /
                                                        f'{period}_{k}_{date_time}.pth'))

