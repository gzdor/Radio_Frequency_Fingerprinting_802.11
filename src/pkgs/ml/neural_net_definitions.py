#!/usr/bin/env python3
"""
#++++++++++++++++++++++++++++++++++++++++++++++

    Project: Part of final project for Georgia Tech Institute of Technology course DL, CS 7643. 

    Totality of this code is non-proprietary and may be used at will. 

#++++++++++++++++++++++++++++++++++++++++++++++


Description: 

@brief a module defining various neural networks. 

@author: Greg Zdor (gzdor@icloud.com)

@date Date_Of_Creation: 4/12/2023 

@date Last_Modification 4/12/2023 

No Copyright - use at will

"""


import torch
import torchmetrics
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl



class LightningCNNClassifier(pl.LightningModule): 
    
    def __init__(self, cfg):
        """
        @brief class constructor for convolution-based neural network 
        classifier. 

        @type cfg dict 
        @param cfg a dictionary of network configuration parameter values 
        """

        super(LightningCNNClassifier, self).__init__()

        self.residual_block1 = ResidualBlock(cfg['conv_layers']['conv_2_n_filters'], cfg['conv_layers']['conv_2_n_filters'])
        
        self.residual_block2 = ResidualBlock(cfg['conv_layers']['conv_2_n_filters'], cfg['conv_layers']['conv_2_n_filters'])

        self.conv_1 = nn.Conv1d(
            cfg['n_features'],
            cfg['conv_layers']['conv_1_n_filters'],
            kernel_size = cfg['conv_layers']['conv_1_kernal_size'],
            stride = cfg['conv_layers']['conv_1_n_stride'], 
            padding = cfg['conv_layers']['conv_1_padding']
            )

        self.conv_2 = nn.Conv1d(
            cfg['conv_layers']['conv_1_n_filters'],
            cfg['conv_layers']['conv_2_n_filters'],
            kernel_size = cfg['conv_layers']['conv_2_kernal_size'],
            stride = cfg['conv_layers']['conv_2_n_stride'], 
            padding = cfg['conv_layers']['conv_2_padding']
            )
        
        self.batch_norm1 = nn.BatchNorm1d(cfg['conv_layers']['conv_1_n_filters'])
        self.batch_norm2 = nn.BatchNorm1d(cfg['conv_layers']['conv_2_n_filters'])

        self.conv_activation = nn.ReLU()

        self.maxpool = nn.MaxPool1d(cfg['max_pool_kernel_size'], cfg['max_pool_stride']) 

        dense_layer_input_size = 3048 

        self.dense_1 =  nn.Linear(dense_layer_input_size,
                                cfg['dense_layers']['dense_1_hidden_size'])

        self.dense_2 =  nn.Linear(cfg['dense_layers']['dense_2_hidden_size'],
                                cfg['last_dense_layer_size'])

        self.dense_drop_1 = nn.Dropout(cfg['dense_layers']['dense_1_dropout'])

        self.dense_drop_2 = nn.Dropout(cfg['dense_layers']['dense_2_dropout'])

        self.output_dense_layer = nn.Linear(cfg['last_dense_layer_size'], cfg['num_classes'])

        self.cfg = cfg
        
        self.get_softmax = nn.Softmax(dim = 1)
        
        self.loss = nn.CrossEntropyLoss()
        
        self.lr = 1e-3 
        self.momentum = 0.9
        
        self.tr_batch_loss = [] 
        self.val_batch_loss = [] 
        self.test_batch_loss = [] 
        
        self.tr_batch_acc = [] 
        self.val_batch_acc = [] 
        self.test_batch_acc = []
        
    def forward(self, x):
        """"
        @brief executes computation graph forward pass 

        @type x tensor 
        @param x neural network input tensor, of shape [batch_size, 2, 128]
        
        @type logits tensor 
        @return logits unnormalized, raw dense layer outputs, not put thru softmax layer yet 
        """

        # Convolution layers
        x = self.conv_1(x)
        x = self.batch_norm1(x)
        x = self.conv_activation(x)
        x = self.maxpool(x)

        if self.cfg['n_conv_layers'] > 1: 
            x = self.conv_2(x)
            x = self.batch_norm2(x)
            x = self.conv_activation(x)
            
            # Residual blocks 
            x = self.residual_block1(x)
            x = self.residual_block2(x)

        x = torch.flatten(x, 1) 

        # Fully connected layers 
        x = self.dense_1(x)
        x = self.dense_drop_1(x)

        if self.cfg['n_dense_layers'] > 1: 
            x = self.dense_2(x)
            x = self.dense_drop_2(x)

        logits = self.output_dense_layer(x)

        return logits


    def configure_optimizers(self): 
        optimizer = torch.optim.SGD(params = self.parameters(),
            lr = self.lr,
            momentum = self.momentum,
            ) #TODO use ADAM torch.optim.Adam(self.parameters(), lr  = 1e-3) 
        return optimizer 


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self(x)        
        loss = self.loss(logits, y.squeeze())
        probabilities = self.get_softmax(logits)
        acc = self._compute_accuracy(y, probabilities)
        
        self.log('train_loss', loss, prog_bar = True, sync_dist=True)
        self.log('train_accuracy', acc, prog_bar = True, sync_dist=True)
        
        self.tr_batch_loss.append(loss)
        self.tr_batch_acc.append(acc)
        
        return loss


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)
        probabilities = self.get_softmax(logits)
        loss = self.loss(logits, y.squeeze())
        acc = self._compute_accuracy(y, probabilities)
        
        self.log('validation_loss', loss, prog_bar = True, sync_dist=True)
        self.log('validation_accuracy', acc, prog_bar = True, sync_dist=True)
        
        self.val_batch_loss.append(loss)
        self.val_batch_acc.append(acc)
        
        return {"validation_loss" : loss, "validation_accuracy" : acc}
    
    
    def test_step(self, val_batch, batch_idx): 
        x, y = val_batch
        logits = self(x)        
        loss = self.loss(logits, y.squeeze())
        probabilities = self.get_softmax(logits)
        acc = self._compute_accuracy(y, probabilities)
        
        self.log('test_loss', loss, prog_bar = True, sync_dist=True)
        self.log('test_accuracy', acc, prog_bar = True, sync_dist=True)
        
        self.test_batch_loss.append(loss)
        self.test_batch_acc.append(acc)
        
        return {"test_loss" : loss, "test_accuracy" : acc}
    
    
    def predict_step(self, batch, batch_idx): 
        x, y = batch
        logits = self(x)
        probabilities = self.get_softmax(logits)
        
        return probabilities, y
    
        
    def on_train_epoch_end(self): 
        
        avg_tr_acc = torch.mean(torch.stack(self.tr_batch_acc))
        
        avg_tr_loss = torch.mean(torch.stack(self.tr_batch_loss))
        
        self.log("avg_tr_acc", avg_tr_acc, sync_dist=True)
        self.log("avg_tr_loss", avg_tr_loss, sync_dist=True)
        
        self.tr_batch_acc.clear()
        self.tr_batch_loss.clear()
    

    def on_validation_epoch_end(self): 
        
        avg_val_acc = torch.mean(torch.stack(self.val_batch_acc))
        
        avg_val_loss = torch.mean(torch.stack(self.val_batch_loss))
        
        self.log("avg_val_acc", avg_val_acc, sync_dist=True)
        self.log("avg_val_loss", avg_val_loss, sync_dist=True)
        
        self.val_batch_acc.clear()
        self.val_batch_loss.clear()
    
    
    def on_test_epoch_end(self): 
        
        avg_test_acc = torch.mean(torch.stack(self.test_batch_acc))
        
        avg_test_loss = torch.mean(torch.stack(self.test_batch_loss))
        
        self.log("avg_test_acc", avg_test_acc, sync_dist=True)
        self.log("avg_test_loss", avg_test_loss, sync_dist=True)
        
        self.test_batch_acc.clear()
        self.test_batch_loss.clear()

    
    def _compute_accuracy(self, labels, predictions): 
        
        acc = torchmetrics.functional.accuracy(
            torch.argmax(predictions, dim = 1), 
            labels.squeeze(), 
            task = "multiclass",
            num_classes = self.cfg['num_classes'],
        )
        
        return acc 
    
    
class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride = 1):
        
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU())
        
        self.conv2 = nn.Sequential(
                        nn.Conv1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm1d(out_channels))
        
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out
    
    

class VariableLayersCNN(pl.LightningModule):
    
    def __init__(self, config_dict: dict):
        super(VariableLayersCNN, self).__init__()
        
        #initialize variables for number of layers
        conv_layers = config_dict.pop('conv_layers')
        dense_layers = config_dict.pop('dense_layers')
        in_channels = config_dict['input_channels']
        filter_sizes = config_dict['filter_size']
        num_filters = config_dict['num_filters']
           
        #create convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(conv_layers):
            out_channels = num_filters[i]
            kernel_size = filter_sizes[i]
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size))
            in_channels = out_channels
        
        #create pooling layers
        pool_size = config_dict['pool_size']
        pool_stride = config_dict['pool_stride']
        self.pool_layers = nn.ModuleList()
        for i in range(conv_layers):
            self.pool_layers.append(nn.MaxPool1d(pool_size, stride=pool_stride))
        
        #initialize feature size variables for dense layers
        dense_layer_sizes = config_dict['dense_layer_sizes']
                
        #create dense layers 
        # https://pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html - to create varying size input layers
        self.dense_layers = nn.ModuleList()
        for i in range(dense_layers):
            out_features = dense_layer_sizes[i]
            self.dense_layers.append(nn.LazyLinear(out_features))  
        
        #create final output layer
        self.output_layer = nn.LazyLinear(config_dict['num_classes'])
        
        #dropout probability
        self.dropout = nn.Dropout(config_dict['dropout'])
        
        #define training parameters 
        self.lr = config_dict['learning_rate']
        self.momentum = config_dict['momentum']
        
        #define helper util
        self.get_softmax = nn.Softmax(dim = 1)
        
        #define optimizer
        self.loss = nn.CrossEntropyLoss()
        
        #define number of classes
        self.num_classes = config_dict['num_classes']
        
        #define lists to store training data 
        self.tr_batch_loss = [] 
        self.val_batch_loss = [] 
        self.test_batch_loss = [] 
        
        self.tr_batch_acc = [] 
        self.val_batch_acc = [] 
        self.test_batch_acc = []

    def forward(self, x):
        #convolutional layers
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = F.relu(conv(x))
            x = pool(x)

        #flatten tensor before passing to dense layers
        x = torch.flatten(x, start_dim=1)

        #dense layers
        for dense in self.dense_layers:
            x = F.relu(dense(x))
            x = self.dropout(x)

        #final output layer - outputs logits, not probabilities
        output = self.output_layer(x)
        
        return output
    
    def configure_optimizers(self): 
        optimizer = torch.optim.SGD(params = self.parameters(),
            lr = self.lr,
            momentum = self.momentum)
        return optimizer 


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self(x)        
        loss = self.loss(logits, y.squeeze())
        probabilities = self.get_softmax(logits)
        acc = self._compute_accuracy(y, probabilities)
        
        self.log('train_loss', loss, prog_bar = True, sync_dist=True)
        self.log('train_accuracy', acc, prog_bar = True, sync_dist=True)
        
        self.tr_batch_loss.append(loss)
        self.tr_batch_acc.append(acc)
        
        return loss


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)
        probabilities = self.get_softmax(logits)
        loss = self.loss(logits, y.squeeze())
        acc = self._compute_accuracy(y, probabilities)
        
        self.log('validation_loss', loss, prog_bar = True, sync_dist=True)
        self.log('validation_accuracy', acc, prog_bar = True, sync_dist=True)
        
        self.val_batch_loss.append(loss)
        self.val_batch_acc.append(acc)
        
        return {"validation_loss" : loss, "validation_accuracy" : acc}
    
    
    def test_step(self, val_batch, batch_idx): 
        x, y = val_batch
        logits = self(x)        
        loss = self.loss(logits, y.squeeze())
        probabilities = self.get_softmax(logits)
        acc = self._compute_accuracy(y, probabilities)
        
        self.log('test_loss', loss, prog_bar = True, sync_dist=True)
        self.log('test_accuracy', acc, prog_bar = True, sync_dist=True)
        
        self.test_batch_loss.append(loss)
        self.test_batch_acc.append(acc)
        
        return {"test_loss" : loss, "test_accuracy" : acc}
    
    
    def predict_step(self, batch, batch_idx): 
        x, y = batch
        logits = self(x)
        probabilities = self.get_softmax(logits)
        
        return probabilities, y
    
        
    def on_train_epoch_end(self): 
        
        avg_tr_acc = torch.mean(torch.stack(self.tr_batch_acc))
        
        avg_tr_loss = torch.mean(torch.stack(self.tr_batch_loss))
        
        self.log("avg_tr_acc", avg_tr_acc, sync_dist=True)
        self.log("avg_tr_loss", avg_tr_loss, sync_dist=True)
        
        self.tr_batch_acc.clear()
        self.tr_batch_loss.clear()
    

    def on_validation_epoch_end(self): 
        
        avg_val_acc = torch.mean(torch.stack(self.val_batch_acc))
        
        avg_val_loss = torch.mean(torch.stack(self.val_batch_loss))
        
        self.log("avg_val_acc", avg_val_acc, sync_dist=True)
        self.log("avg_val_loss", avg_val_loss, sync_dist=True)
        
        self.val_batch_acc.clear()
        self.val_batch_loss.clear()
    
    
    def on_test_epoch_end(self): 
        
        avg_test_acc = torch.mean(torch.stack(self.test_batch_acc))
        
        avg_test_loss = torch.mean(torch.stack(self.test_batch_loss))
        
        self.log("avg_test_acc", avg_test_acc, sync_dist=True)
        self.log("avg_test_loss", avg_test_loss, sync_dist=True)
        
        self.test_batch_acc.clear()
        self.test_batch_loss.clear()

    
    def _compute_accuracy(self, labels, predictions): 
        
        acc = torchmetrics.functional.accuracy(
            torch.argmax(predictions, dim = 1), 
            labels.squeeze(), 
            task = "multiclass",
            num_classes = self.num_classes,
        )
        
        return acc 