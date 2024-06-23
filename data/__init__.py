import torch.utils.data
from data.base_data_loader import BaseDataLoader
import numpy as np
import h5py
import random

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'nii_3D_fine':
        from data.nii_3D_finetune_dataset import Nii3DPREDataset
        dataset = Nii3DPREDataset()
    elif opt.dataset_mode == 'nii_3D_pre_train_3':
        from data.nii_3D_pretrain_dataset_train_3 import Nii3DPREDataset
        dataset = Nii3DPREDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)
    dataset.initialize(opt)
    return dataset 


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
