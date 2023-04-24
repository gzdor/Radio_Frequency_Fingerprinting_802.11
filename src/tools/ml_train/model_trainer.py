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

# @date Last_Modification 4/24/2023 

# No Copyright - use at will

# """

# System level imports 
import os 
import sys
import time 
import yaml
import shutil
import string 
import random
import argparse
from datetime import datetime

# ML imports 
import torchsummary
import pytorch_lightning as pl

# Hyper parameter tuning imports 
import ray 
from ray import tune 
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# Add local source files to python path
sys.path.append(f'..{os.sep}..{os.sep}')

# Local imports
from pkgs.utils.yaml_loader import custom_yaml_loader
from pkgs.dataset.oracle_lightning_dataset import LightningOracleDataset
from pkgs.ml.neural_net_search_spaces import (
    lightning_vanilla_cnn_classifier_cfg,
    lightning_tunable_cnn_classifier_cfg,
    variable_layers_cnn_cfg, 
)
from pkgs.ml.neural_net_definitions import LightningCNNClassifier, VariableLayersCNN
from pkgs.ml.model_evaluation_analysis import model_evaluation


def trial_dir_name_creator(trial): 

    # defaults to --> "{}_{}".format(trial.trainable_name, trial.trial_id)
    options = string.ascii_lowercase
    rnd_str = ''.join(str(random.choice(options)) for _ in range(4))
    trial_dir_name = "{}_{}".format(trial.trainable_name, rnd_str)
    return trial_dir_name


def trial_name_creator(trial):
    
    trial_name_str = 'trainable'
    return trial_name_str


def tune_experiment_name_creator():
    
    exper_name = 'hyper_tuning_experiment_' + "{:%Y_%m_%d_%H_%M_%S_%MS}".format(datetime.now())
    return exper_name


def setup_dataset(cfg:dict) -> pl.LightningDataModule:
    
    return LightningOracleDataset(cfg) 


def setup_model(cfg:dict) -> pl.LightningModule:
    
    if cfg['use_hyper_parameter_tuning']:
        # Get variable parameters, layers model 
        model = VariableLayersCNN(cfg)
    else:
        # Get default, fixed-valued model training and architecture parameters 
        model = LightningCNNClassifier(cfg)  
    
    return model


def setup_trainer(cfg:dict) -> pl.Trainer:    
    
    # Define Lightning callbacks
    lightning_callbacks = [ 
        pl.callbacks.ModelCheckpoint(monitor = "validation_loss", mode = "min"), 
        pl.callbacks.EarlyStopping(monitor='validation_loss', patience=cfg['lightning_patience']),]
    
    # Check if to use Ray Tune hyper parameter tuning 
    if "use_hyper_parameter_tuning" in trainer_cfg.keys():
        if trainer_cfg['use_hyper_parameter_tuning']:
            metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"} 
            ray_tune_callback = TuneReportCallback(metrics, on="validation_end")
            lightning_callbacks.append(ray_tune_callback)

    # Define loggers 
    logger = pl.loggers.CSVLogger("lightning_logs", name = trainer_cfg['exper_name'])

    n_epochs = cfg['epochs']
    devices = cfg['devices']
    accelerator = cfg['accelerator']
    strategy = cfg['strategy']
    enable_progress_bar = cfg['enable_progress_bar']

    # Configure trainer
    trainer = pl.Trainer(
        enable_progress_bar          = enable_progress_bar, 
        enable_checkpointing         = True, 
        enable_model_summary         = True,
        callbacks                    = lightning_callbacks, 
        max_epochs                   = n_epochs, 
        precision                    = 32, # options are 16 or 32 
        logger                       = logger,
        strategy                     = strategy,
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
    
    torchsummary.summary(model.cuda(), (2, 128))
    
    # execute training 
    n_epochs = trainer_cfg['epochs']
    t_start = time.time()
    trainer.fit(model, dataset)
    tr_t_sec = time.time() - t_start
    print(
        f'\n\nTotal training time was {tr_t_sec} seconds, @ {str(tr_t_sec/n_epochs)} seconds per epoch.\n')

    # execute testing 
    test_trainer = pl.Trainer(logger = False)
    test_results = test_trainer.test(model, dataset, verbose = True)
    print(f'\n\n test results are:  \n\n {test_results}\n\n')
    
    experiment_path = os.path.join(os.getcwd(), trainer._loggers[0].log_dir)
    current_logs_csv = os.path.join(os.getcwd(), experiment_path, "metrics.csv")
    print(f'Experiment path:  {experiment_path}\n\n')
    
    # make copy of training history data - fixes trainer overwrites metrics.csv with testing results
    if os.path.isfile(current_logs_csv):
        dest_csv = os.path.join(os.getcwd(), experiment_path, "train_val_metrics.csv")
        shutil.copy(
            current_logs_csv, 
            dest_csv, 
        )
    
    # execute training results analysis 
    model_evaluation(current_logs_csv, experiment_path, n_epochs)
    
    return experiment_path


def tunable_trainer(
    config:      dict,
    dataset_cfg: dict,
    trainer_cfg: dict,
    ): 
    
    dataset = setup_dataset(dataset_cfg)

    config['use_hyper_parameter_tuning'] = True
    model = setup_model(config)
    
    trainer_cfg['use_hyper_parameter_tuning'] = True
    trainer = setup_trainer(trainer_cfg)
    
    torchsummary.summary(model.cuda(), (2, 128))
    
    trainer.fit(model, dataset)
    

def run_tuning(
    dataset_cfg: dict,
    model_cfg  : dict,
    trainer_cfg: dict,
    ): 
    
    search_space = variable_layers_cnn_cfg()
    
    # Distributed Asynchronous Hyper-parameter Optimizer (HyperOptSearch)
    # for efficient hyperpameter value down selection  
    search_algorithm = HyperOptSearch(metric = model_cfg['metric'],
        mode = model_cfg['mode'])
    
    # AsyncHyperBand scheduler enables aggressive early stopping of bad trials  
    scheduler = AsyncHyperBandScheduler(max_t = trainer_cfg['epochs'],
        grace_period = model_cfg['grace_period'])
    
    # Ray runs each trial/actor in a new process as part of its parallel processing, so 
    # since sys.path.append only applies to the current process and this change is not 
    # propogated to the worker processes, we manually add the path to $PYTHONPATH env variable.
    # Calling ray.init() inherits the $PYTHONPATH env variable for all its worker processes
    # so the change to the path is propogated to all worker processes.

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
    
    trainable = tune.with_parameters(
        tunable_trainer,
        dataset_cfg = dataset_cfg, 
        trainer_cfg = trainer_cfg,
    )
    
    ray.init(include_dashboard = False)
    
    analysis = tune.run(
        trainable, 
        resources_per_trial = {
            "cpu": model_cfg['cpus_per_trial'], 
            "gpu": model_cfg['gpus_per_trial'], 
        }, 
        metric = "loss",
        mode = "min",
        config = search_space, 
        num_samples = model_cfg['n_tuning_trials'], 
        local_dir = os.getcwd(), 
        fail_fast = False, 
        scheduler = scheduler, 
        search_alg = search_algorithm,
        trial_name_creator = trial_name_creator,
        trial_dirname_creator = trial_dir_name_creator, 
        name = tune_experiment_name_creator(),
        verbose = 0,
    )
    
    print(
        f'\n\nbest tuning cfg:  {analysis.best_config}\n'
        f'best tuning trial path:  {analysis.best_trial}\n'
        f'best tuning trial logdir:  {analysis.best_logdir}\n'
        f'best tuning trial result:  {analysis.best_result}\n\n'
        )
    

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
    
    # Check if to run default, vanilla model with fixed parameters or to do hyper parameter tuning with 1 or more models
    
    if model_cfg['use_hyper_parameter_tuning']:
         
        # Define output experiment path 
        exper_name = "experiment_" + "{:%Y_%m_%d_%H_%M_%S_%MS}".format(datetime.now())
        trainer_cfg['exper_name'] = exper_name

        # Run Ray Tune - based hyper parameter tuning - based training 
        experiment_path = run_tuning(
            dataset_cfg,
            model_cfg,
            trainer_cfg,
        )
    
    else:
        
        # Define output experiment path 
        exper_name = "experiment_" + "{:%Y_%m_%d_%H_%M_%S_%MS}".format(datetime.now())
        trainer_cfg['exper_name'] = exper_name
    
        # Run default, vanilla training and evaluation 
        experiment_path = run(
            dataset_cfg,
            model_cfg,
            trainer_cfg,
        )
    
    print(f'Current experiment results are under \n\n{experiment_path}')
    
    print(f"\n\nTotal run time was: \n\n {str(time.time() - t_start)} seconds\n\nProgram ended.")
    
    
    
    
    
    
    
    
#       File "/MLSR_Shared_Folder/gzdor/Miniconda_Install/envs/rf-fingerprinting-proj-pytorch-cs7643/lib/python3.10/site-packages/ray/tune/execution/trial_runner.py", line 1047, in _on_training_result
#     self._process_trial_results(trial, result)
#   File "/MLSR_Shared_Folder/gzdor/Miniconda_Install/envs/rf-fingerprinting-proj-pytorch-cs7643/lib/python3.10/site-packages/ray/tune/execution/trial_runner.py", line 1130, in _process_trial_results
#     decision = self._process_trial_result(trial, result)
#   File "/MLSR_Shared_Folder/gzdor/Miniconda_Install/envs/rf-fingerprinting-proj-pytorch-cs7643/lib/python3.10/site-packages/ray/tune/execution/trial_runner.py", line 1167, in _process_trial_result
#     self._validate_result_metrics(flat_result)
#   File "/MLSR_Shared_Folder/gzdor/Miniconda_Install/envs/rf-fingerprinting-proj-pytorch-cs7643/lib/python3.10/site-packages/ray/tune/execution/trial_runner.py", line 1263, in _validate_result_metrics
#     raise ValueError(
# # ValueError: Trial returned a result which did not include the specified metric(s) `loss` that `tune.TuneConfig()` expects. Make sure your calls to `tune.report()` include the metric, or set the TUNE_DISABLE_STRICT_METRIC_CHECKING environment variable to 1. Result: {'trial_id': 'af1fa306', 'experiment_id': '76cb6dd9c79e4fe6ab08995ecf8bb825', 'date': '2023-04-24_01-52-18', 'timestamp': 1682301138, 'pid': 35907, 'hostname': 'ash3-mlrd-nvidia01.lnx.boozallencsn.com', 'node_ip': '10.137.224.83', 'done': True, 'config/conv_layers': 4, 'config/dense_layers': 4, 'config/input_channels': 2, 'config/num_filters': (108, 75, 81, 48, 42), 'config/filter_size': (3, 3, 3, 3, 3), 'config/pool_size': 3, 'config/dense_layer_sizes': (476, 291, 216, 185, 108), 'config/input_size': 128, 'config/in_channels': 1, 'config/pool_stride': 1, 'config/dropout': 0.5221329751701351, 'config/num_classes': 16, 'config/learning_rate': 0.0011963725382409184, 'config/momentum': 0.9154664600997693}