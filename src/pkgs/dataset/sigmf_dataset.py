import math as mth
from numpy.random import Generator, MT19937, SeedSequence
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from memory_mapper import SigMFDataMMapper
except: 
    from pkgs.dataset.memory_mapper import SigMFDataMMapper

class SigMFDataset(Dataset):
    
    def __init__(self,
                 datasplit,
                 mmapper,
                 transform=None):

        self.h_labels = mmapper.get_h_labels(datasplit, "r")
        self.h_data = mmapper.get_h_data(datasplit, "r")
        self.transform = transform
        
    def __len__(self):
        
        return self.h_labels.shape[0]
    
    def __getitem__(self, idx):
        
        sample_data = self.h_data[idx, :, :].copy()

        if self.transform:
            sample_data = self.transform(sample_data)

        sample_data = torch.from_numpy(sample_data)
        sample_label = torch.from_numpy(self.h_labels[idx].copy())

        # https://discuss.pytorch.org/t/
        #     runtimeerror-multi-target-not-supported-newbie/10216
        return sample_data, sample_label.long()
    

class IQDataAugmenter(object):

    def __init__(self,
                 **kwargs):

        # 3-sigma pi/4 phase error
        self.phase_error_stdev =\
            kwargs.get("phase_error_stdev", (mth.pi / 4) / 3)
        
        # Gaussian noise floor (dB below average envelope)
        self.awgn_stddev_scaling =\
            np.power(10, kwargs.get("awgn_relative_db", -20) / 10)

        sg = SeedSequence(kwargs.get("rngseed", 773764502))
        self.rng = Generator(MT19937(sg))

    def __call__(self, sample_data):
        
        phase_error_draw =\
            self.rng.normal(scale=self.phase_error_stdev,
                            size=1) * self.phase_error_stdev

        complex_sample = sample_data.copy()
        complex_sample = complex_sample[0, :] + 1j * complex_sample[1, :]
        complex_sample *= np.exp(1j * phase_error_draw)        

        scale =\
            np.abs(complex_sample).mean() *\
            self.awgn_stddev_scaling / np.sqrt(2)

        awgn =\
            self.rng.normal(size=len(complex_sample), scale=scale) +\
            1j * self.rng.normal(size=len(complex_sample), scale=scale)

        complex_sample += awgn
        
        augmented_sample_data =\
            np.concatenate([np.real(complex_sample).reshape(1, -1),
                            np.imag(complex_sample).reshape(1, -1)],
                           axis=0)

        return augmented_sample_data


class SigMFDataModule(pl.LightningDataModule):
    """
    https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    """
    def __init__(self, parameters):
        
        self.mmapper = SigMFDataMMapper(parameters)

        self.batch_size = parameters.get("batch_size", 512)
        self.num_workers = parameters.get("num_workers", 8)

    def setup(self, stage: str):
        
        self.mmapper.map()

        self.sigmf_training =\
            SigMFDataset("training",
                         self.mmapper,
                         IQDataAugmenter())
        
        self.sigmf_validation =\
            SigMFDataset("validation", self.mmapper)
        
        self.sigmf_test = SigMFDataset("test", self.mmapper)
        
    def train_dataloader(self,
                         shuffle=True):

        return DataLoader(self.sigmf_training,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle,
                          pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.sigmf_validation,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.sigmf_test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)