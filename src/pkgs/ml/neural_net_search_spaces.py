#!/usr/bin/env python3
"""
#++++++++++++++++++++++++++++++++++++++++++++++

    Project: Part of final project for Georgia Tech Institute of Technology course DL, CS 7643. 

    Totality of this code is non-proprietary and may be used at will. 

#++++++++++++++++++++++++++++++++++++++++++++++


Description: 

@brief a module defining neural network architecture and training parameters. 

@author: Greg Zdor (gzdor@icloud.com)

@date Date_Of_Creation: 4/12/2023 

@date Last_Modification 4/12/2023 

No Copyright - use at will

"""

import ray 
from ray import tune 


def lightning_vanilla_cnn_classifier_cfg(): 

    cfg = {

        "num_classes"                 :        16, 
        "vector_len"                  :        128, # length in time of input signal sequence
        "n_features"                  :        2, # I and Q signal dimensions 
        "max_pool_kernel_size"        :        2, 
        "max_pool_stride"             :        1,
        "n_conv_layers"               :        2,
        "conv_layers"                 :  
            {
                "conv_1_n_filters"    :        256, 
                "conv_1_n_stride"     :        1, 
                "conv_1_kernal_size"  :        3,
                "conv_1_padding"      :        1, 
                "conv_2_n_filters"    :        128, 
                "conv_2_n_stride"     :        1, 
                "conv_2_kernal_size"  :        3,
                "conv_2_padding"      :        1, 
            }, 
        "conv_activations"            :        "Relu",
        "n_dense_layers"              :        2, 
        "dense_layers"                :
            {
                "dense_1_hidden_size" :        1024, 
                "dense_2_hidden_size" :        1024, 
                "dense_1_dropout"     :        0.1, 
                "dense_2_dropout"     :        0.1, 
            },
        "last_dense_layer_size"       :        64,
    }
    
    return cfg 

def lightning_tunable_cnn_classifier_cfg(): 

    cfg = {

        "num_classes"                 :        16, 
        "vector_len"                  :        128, # length in time of input signal sequence
        "n_features"                  :        2, # I and Q signal dimensions 
        "max_pool_kernel_size"        :        2, 
        "max_pool_stride"             :        1,
        "n_conv_layers"               :        2,
        "conv_layers"                 :  
            {
                "conv_1_n_filters"    :        tune.randint(64, 512), 
                "conv_1_n_stride"     :        1, 
                "conv_1_kernal_size"  :        3,
                "conv_1_padding"      :        1, 
                "conv_2_n_filters"    :        tune.randint(64, 256), 
                "conv_2_n_stride"     :        1, 
                "conv_2_kernal_size"  :        3,
                "conv_2_padding"      :        1, 
            }, 
        "conv_activations"            :        "Relu",
        "n_dense_layers"              :        2, 
        "dense_layers"                :
            {
                "dense_1_hidden_size" :        tune.randint(512, 2048), 
                "dense_2_hidden_size" :        tune.randint(128, 512), 
                "dense_1_dropout"     :        tune.uniform(0.0, 0.75), 
                "dense_2_dropout"     :        tune.uniform(0.0, 0.75), 
            },
        "last_dense_layer_size"       :        tune.randint(32, 128),
        "training_parameters"         :
            {
                "momentum"            :        tune.uniform(0.85, 0.99), 
                "learning_rate"       :        tune.uniform(1e-6, 1e-2), 
                "optimizer"           :        tune.choice(["Adam", "SGD", "AdaGrad"]), 
         }
    }
    
    return cfg 


def variable_layers_cnn_cfg(): 

    cfg = {
        "conv_layers"                 : tune.randint(1, 5), # varying num conv layers
        "dense_layers"                : tune.randint(2, 5), # varying num FCN layers 
        "input_channels"              : 2, 
        "num_filters"                 : [ # defined per layer
            tune.randint(64, 128),
            tune.randint(64, 100),
            tune.randint(54, 84),
            tune.randint(32, 64),
            tune.randint(24, 48)],
        "filter_size"                 : [3, 3, 3, 3, 3], # defined per layer 
        "pool_size"                   : 1,
        "dense_layer_sizes"           : [ # defined per layer 
            tune.randint(400, 512),
            tune.randint(256, 400),
            tune.randint(200, 256),
            tune.randint(128, 200),
            tune.randint(32, 128)],
        "input_size"                  : 128, 
        "in_channels"                 : 1,
        "pool_size"                   : 3,
        "pool_stride"                 : 1,
        "dropout"                     : tune.uniform(0.1, 0.7),
        "num_classes"                 : 16, 
        "learning_rate"       :        tune.uniform(1e-6, 1e-2),
        "momentum"            :        tune.uniform(0.85, 0.99), 
    }
    
    return cfg 

