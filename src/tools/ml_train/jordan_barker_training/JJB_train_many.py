
import os 
import sys 
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger


# Add local source files to python path
sys.path.append(f'..{os.sep}..{os.sep}..{os.sep}')

from pkgs.dataset.memory_mapper import load_parameters
from pkgs.dataset.oracle_lightning_dataset import LightningOracleDataset

from JJB_networks import OracleClassifier, CNNClassifier, EfficientNetTL, ResNet50TL



if __name__=='__main__':
    
    # Params to edit 
    num_workers = 12
    batch_size = 512
    max_epochs = 35
    sys.argv = ["", "./memory_mapper.yml"]

    parameters = load_parameters()
    
    data_module = LightningOracleDataset(parameters)
    data_module.setup("")
    train_dl = data_module.train_dataloader()
    val_dl = data_module.val_dataloader()

    torch.manual_seed(42) 
    torch.set_float32_matmul_precision("high")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: %s" % device)

    models = {
        # 'CNNClassifier' : CNNClassifier(),
        # 'OracleClassifier' : OracleClassifier(),
        'EfficientNetTL': EfficientNetTL(),
        'ResNet50TL' : ResNet50TL()
    }

    total_count = len(models.keys())
    for i, model in enumerate(models.keys()):
        model_name = model
        print("------------------------------")
        print(f"{i}/{total_count} - {model_name}")
        print("------------------------------")
        
        experiment_name = model_name + "_" + "{:%Y_%m_%d_%H_%M_%S_%MS}".format(datetime.now())
        save_dir = os.path.join(os.getcwd(), experiment_name)
        
        if not os.path.isdir(save_dir): 
            os.mkdir(save_dir)
        csv_logger = CSVLogger(save_dir = save_dir) 
        
        lightning_callbacks = [ 
        pl.callbacks.ModelCheckpoint(monitor = "validation_loss", mode = "min")]

        trainer = pl.Trainer(max_epochs=max_epochs,
                            logger=csv_logger,
                            devices = [7], # the gpu at the second index
                            callbacks = lightning_callbacks)

        curr_model = models[model]
                        
        trainer.fit(curr_model, train_dataloaders=train_dl, val_dataloaders=val_dl)    