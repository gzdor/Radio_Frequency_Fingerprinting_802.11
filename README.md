# Deep Learning for Identification of RF Emitters Based on Transmitter Artifacts

### How to use this code base 

    To get started, install the required Python packages from the requirements_base.txt. This can be done with the following steps. 

        conda create --name rf-fingerprinting-proj-pytorch-cs7643 python=3.10.9
        conda activate rf-fingerprinting-proj-pytorch-cs7643
        pip install -r requirements-base.txt

#### Running neural net training and evaluation 

    Training and evaluation is executed via running the following from the src/tools/ml_train folder. 

        python3 model_trainer.py -cfg ../../../configs/rf_fingerprinting_cfg.yaml

    To change training parameters such as number of epochs, learning rate, whether to use hyper parameter 
    tuning or more, modify the contents of /configs/rf_fingerprinting_cfg.yaml. 

### About the models

#### Architecture diagrams 

#### Architecture explanations and motivations 

### About the dataset 

    This WiFi signal classification dataset from Northeastern University's GENESYS Laboratory consists of raw IQ samples 
    from 16 different Ettus USRP X310 transmitters (TX) (SDRs) and one B210 receiver (RX). 

    In particular, the WiFi is IEEE802.11a protocol bursts, and recorded in SigMF standard data format. There are 20 million
    samples per class, recorded open air at TX <--> RX separations ranging from 2 feet to 62 feet apart. Refer to the ORACLE d
    dataset link in the last section of this page for links to download the dataset.

### About the dataset, in detail

### Links and resources 

    ORACLE dataset home page: https://genesys-lab.org/oracle 

    This project was done in partial fulfillment of course requirements for https://omscs.gatech.edu/cs-7643-deep-learning, project "team Electromagnetics!". 
