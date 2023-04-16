"""
#++++++++++++++++++++++++++++++++++++++++++++++

    Project: Part of final project for Georgia Tech Institute of Technology course DL, CS 7643. 

    Totality of this code is non-proprietary and may be used at will. 

#++++++++++++++++++++++++++++++++++++++++++++++


Description: 

@brief Defines a PyTorch Lightning dataset module for Oracle RF Fingerprinting dataset: https://genesys-lab.org/oracle. 

@author: Greg Zdor (gzdor@icloud.com)

@date Date_Of_Creation: 4/16/2023 

@date Last_Modification 4/16/2023 

No Copyright - use at will

"""

import os
import yaml
import sys

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

sys.path.append(f'..{os.sep}..{os.sep}')

# Local imports
from pkgs.dataset.memory_mapper import SigMFDataMMapper
from pkgs.dataset.sigmf_dataset import SigMFDataset, IQDataAugmenter


# Create Pytorch Lightning module for dataset 
class LightningOracleDataset(pl.LightningDataModule): 

    def __init__(self, parameters): 
        """
        @brief constructor 

        @type parameters dict 
        @param parameters the input parameters from the mem mapper yaml config file.
        """

        super(LightningOracleDataset, self).__init__() 

        self.mmapper = SigMFDataMMapper(parameters)
        self.batch_size = parameters.get("batch_size", 1024)
        self.num_workers = parameters.get("num_workers", 2)
        print(f'\nUsing {self.num_workers} workers for data loading in current Lightning data module.\n')


    def setup(self, stage: str):
        
        self.mmapper.map()

        self.sigmf_training =\
            SigMFDataset("training",
                         self.mmapper,
                         IQDataAugmenter())
        
        self.sigmf_validation =\
            SigMFDataset("validation", self.mmapper)
        
        self.sigmf_test = SigMFDataset("test", self.mmapper)


    def train_dataloader(self):
        return DataLoader(self.sigmf_training,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
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
        
    def predict_dataloader(self):
        return DataLoader(self.sigmf_test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)