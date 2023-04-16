# """
# #++++++++++++++++++++++++++++++++++++++++++++++

#     Project: Part of final project for Georgia Tech Institute of Technology course DL, CS 7643. 

#     Totality of this code is non-proprietary and may be used at will. 

# #++++++++++++++++++++++++++++++++++++++++++++++


# Description: 

# @brief Defines top level model training and evaluation run module. 

# How to use: 

#       python3 model_trainer.py -cfg ../../../configs/rf_fingerprinting_cfg.yaml

# @author: Greg Zdor (gzdor@icloud.com)

# @date Date_Of_Creation: 4/16/2023 

# @date Last_Modification 4/16/2023 

# No Copyright - use at will

# """

# System level imports 
import os 
import sys
import time 
import yaml
import argparse
from datetime import datetime

# ML imports 
import pytorch_lightning as pl

# Add local source files to namespace 
sys.path.append(f'..{os.sep}..{os.sep}')

# Local imports
from pkgs.utils.yaml_loader import custom_yaml_loader
from pkgs.dataset.oracle_lightning_dataset import LightningOracleDataset
from pkgs.ml.neural_net_search_spaces import (
    lightning_vanilla_cnn_classifier_cfg,
    lightning_tunable_cnn_classifier_cfg,
)
from pkgs.ml.neural_net_definitions import LightningCNNClassifier
from pkgs.ml.model_evaluation_analysis import model_evaluation


def setup_dataset(cfg:dict) -> pl.LightningDataModule:
    return LightningOracleDataset(cfg) 


def setup_model(cfg:dict) -> pl.LightningModule:
    if cfg['use_hyper_param_tuning']:
        # Get hyper parameter tuning search space parameters 
        cfg = lightning_tunable_cnn_classifier_cfg()
    else:
        # Get default, fixed-valued model training and architecture parameters 
        cfg = lightning_vanilla_cnn_classifier_cfg()
    
    return LightningCNNClassifier(cfg)  


def setup_trainer(cfg:dict) -> pl.Trainer:    
    # Define Lightning callbacks
    lightning_callbacks = [ 
        pl.callbacks.ModelCheckpoint(monitor = "validation_loss", mode = "min"), 
        pl.callbacks.EarlyStopping(monitor='validation_loss', patience=10),]

    # Define loggers 
    exper_name = "experiment_" + "{:%Y_%m_%d_%H_%M_%S_%MS}".format(datetime.now())
    logger = pl.loggers.CSVLogger("lightning_logs", name = exper_name)

    n_epochs = cfg['epochs']
    devices = cfg['devices']
    accelerator = cfg['accelerator']

    # Configure trainer
    trainer = pl.Trainer(
        enable_progress_bar          = True, 
        enable_checkpointing         = True, 
        enable_model_summary         = True,
        callbacks                    = lightning_callbacks, 
        max_epochs                   = n_epochs, 
        precision                    = 32, # options are 16 or 32 
        logger                       = logger,
        strategy                     = "ddp", # =  distributed data parallel 
        accelerator                  = accelerator,
        devices                      = devices, # uses all available GPUs
    ) 
    
    return trainer


def run(
    dataset_cfg: dict,
    model_cfg  : dict,
    trainer_cfg: dict, 
    ) -> str: 
    
    dataset = setup_dataset(dataset_cfg)
    
    model = setup_model(model_cfg)
    
    trainer = setup_trainer(trainer_cfg)
    
    # execute training 
    n_epochs = trainer_cfg['epochs']
    t_start = time.time()
    trainer.fit(model, dataset)
    tr_t_sec = time.time() - t_start
    print(
        f'\n\nTotal training time was {tr_t_sec} seconds, @ {str(tr_t_sec/n_epochs)} seconds per epoch.\n')
    
    # execute testing 
    test_results = trainer.test(model = model, datamodule = dataset, verbose = True)
    print(f'\n\n test results are:  \n\n {test_results}\n\n')
    
    experiment_path = os.path.join(os.getcwd(), trainer._loggers[0].log_dir)
    current_logs_csv = os.path.join(os.getcwd(), experiment_path, "metrics.csv")
    
    # execute training results analysis 
    model_evaluation(current_logs_csv, experiment_path, n_epochs)
    
    return experiment_path


if __name__=='__main__':
    
    # Start timer 
    t_start = time.time()
        
    # Get repo root path 
    start_dir = os.getcwd() 
    os.chdir(f'..{os.sep}..{os.sep}..{os.sep}')
    repo_root_path = os.getcwd() 
    os.chdir(start_dir)
    print(f'\n Repo location of:   {repo_root_path}')
    
    # Get configuration parameters 
    default_cfg_path = os.path.join(repo_root_path, "configs", "rf_fingerprinting_cfg.yaml")
    
    argparser_cfg_filename_help = (
        f'The full path + name to the top level YAML config file, located under /configs/'
        f'Defaults to Radio_Frequency_Fingerprinting_802.11/configs/rf_fingerprinting_cfg.yaml'
    )
    
    parser = argparse.ArgumentParser(description='Top level neural net training and evaluation run script.')
    
    parser.add_argument('-cfg', '--config', dest='cfg_path',
        default=default_cfg_path, help = argparser_cfg_filename_help)
    
    args = parser.parse_args()
    
    # Load configuration files 
    cfg_path = args.cfg_path
    cfg = custom_yaml_loader(cfg_path) 

    dataset_cfg = cfg['dataset_parameters'] 
    model_cfg = cfg['model_parameters'] 
    trainer_cfg = cfg['lightning_trainer_parameters'] 

    print(
        f'\n\nUsing config parameters of\n\n{yaml.dump(cfg, allow_unicode=True, default_flow_style=False)}\n\n'
        )
    
    # Run training and evaluation 
    experiment_path = run(
        dataset_cfg,
        model_cfg,
        trainer_cfg,
    )
    
    print(f'Current experiment results are under \n\n{experiment_path}')
    
    print(f"\n\nTotal run time was: \n\n {str(time.time() - t_start)} seconds\n\nProgram ended.")