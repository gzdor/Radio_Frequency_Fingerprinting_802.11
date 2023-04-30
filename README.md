# Deep Learning for Identification of RF Emitters Based on Transmitter Artifacts

### About this code base 

This repository was created to do radio frequency (RF) emitter identification (classification) of Ettus software defined radios via neural networks in PyTorch Lightning. 

The **final project report** provides details and may be found in this repo under docs/TeamElectromagnetics!ProjectReport.pdf

The dataset used here was released in conjunction with Northeastern University GENESYS Laboratory’s 2018 IEEE INFOCOM journal article “ORACLE: Optimized Radio clAssification through Convolutional Neural nEtworks”.

This repository contains modules defining several convolutional-based neural network architectures. A main training module is defined, and this module allows for training a single model or a hyper parameter tuning a large number of models in parallel via Ray Tune. Analysis Jupyter Notebooks are used to point to a specific model checkpoint, load the weights, and perform test dataset evaluation and plot corresponding confusion matrix results. Initial dataset exploratory notebooks and also PyTorch Lightning dataset instantiator and loader utility classes are also defined. 

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

### About the dataset 

This WiFi signal classification dataset from Northeastern University's GENESYS Laboratory consists of raw IQ samples 
from 16 different Ettus USRP X310 transmitters (TX) (SDRs) and one B210 receiver (RX). 

In particular, the WiFi is IEEE802.11a protocol bursts, and recorded in SigMF standard data format. There are 20 million
samples per class, recorded open air at TX <--> RX separations ranging from 2 feet to 62 feet apart. Refer to the ORACLE d
dataset link in the last section of this page for links to download the dataset.

### Links and resources 

ORACLE dataset home page: https://genesys-lab.org/oracle 
The dataset used for this project may be downloaded via the following links. (first link is demodulated IQ symbols (not used) and second link is raw IQ data, which is what was used for this project).

        wget -O demod_iq.zip https://repository.library.northeastern.edu/downloads/neu:m044q523j?datastream_id=content
        wget -O raw_iq.zip https://repository.library.northeastern.edu/downloads/neu:m044q5210?datastream_id=content
    
This project was done in partial fulfillment of course requirements for https://omscs.gatech.edu/cs-7643-deep-learning, project "team Electromagnetics!", spring 2023. 
