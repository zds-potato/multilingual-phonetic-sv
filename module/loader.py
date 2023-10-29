import os
from typing import Any, Callable, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import Evaluation_Dataset, Train_Dataset


class SPK_datamodule(LightningDataModule):
    def __init__(
        self,
        train_csv_path,
        trial_path,
        unlabel_csv_path = None,
        second: int = 2,
        num_workers: int = 16,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        num_classes: int = 1211,
        speed_perturb_flag: bool = False,
        add_reverb_noise: bool = False,
        spec_aug_flag: bool = False,
        semi: bool = False,
        asnorm: bool = False,
        cohort_path: str = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.train_csv_path = train_csv_path
        self.unlabel_csv_path = unlabel_csv_path
        self.second = second
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.trial_path = trial_path

        #数据增强
        self.speed_perturb_flag = speed_perturb_flag
        self.add_reverb_noise = add_reverb_noise
        self.spec_aug_flag = spec_aug_flag
        print("second is {:.2f}".format(second))

        #asnorm
        self.asnorm = asnorm
        self.cohort_path = cohort_path

    def train_dataloader(self) -> DataLoader:
        train_dataset = Train_Dataset(self.train_csv_path, self.second, num_classes= self.num_classes,
                                      speed_perturb_flag=self.speed_perturb_flag,
                                      add_reverb_noise=self.add_reverb_noise,
                                      spec_aug_flag=self.spec_aug_flag)
        loader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                pin_memory=True,
                drop_last=False,
                )
        return loader

    def val_dataloader(self) -> DataLoader:
        trials = np.loadtxt(self.trial_path, str)
        self.trials = trials
        if self.asnorm:
            eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2], np.loadtxt(self.cohort_path, str))))
            print("number of cohort: {}".format(len(np.loadtxt(self.cohort_path, str))))
        else:
            eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2])))
        print("number of enroll: {}".format(len(set(trials.T[1]))))
        print("number of test: {}".format(len(set(trials.T[2]))))
        print("number of evaluation: {}".format(len(eval_path)))
        eval_dataset = Evaluation_Dataset(eval_path, second=-1)
        loader = torch.utils.data.DataLoader(eval_dataset,
                                             num_workers=10,
                                             shuffle=False, 
                                             batch_size=1)
        return loader

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()


