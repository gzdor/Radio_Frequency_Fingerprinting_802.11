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
                "conv_1_n_filters"    :        128, 
                "conv_1_n_stride"     :        1, 
                "conv_1_kernal_size"  :        3,
                "conv_1_padding"      :        1, 
                "conv_2_n_filters"    :        16, 
                "conv_2_n_stride"     :        1, 
                "conv_2_kernal_size"  :        3,
                "conv_2_padding"      :        1, 
            }, 
        "conv_activations"            :        "Relu",
        "n_dense_layers"              :        2, 
        "dense_layers"                :
            {
                "dense_1_hidden_size" :        128, 
                "dense_2_hidden_size" :        128, 
                "dense_1_dropout"     :        0.2, 
                "dense_2_dropout"     :        0.2, 
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
                "conv_1_n_filters"    :        128, 
                "conv_1_n_stride"     :        1, 
                "conv_1_kernal_size"  :        3,
                "conv_1_padding"      :        1, 
                "conv_2_n_filters"    :        16, 
                "conv_2_n_stride"     :        1, 
                "conv_2_kernal_size"  :        3,
                "conv_2_padding"      :        1, 
            }, 
        "conv_activations"            :        "Relu",
        "n_dense_layers"              :        2, 
        "dense_layers"                :
            {
                "dense_1_hidden_size" :        128, 
                "dense_2_hidden_size" :        128, 
                "dense_1_dropout"     :        tune.uniform(0.0, 0.75), 
                "dense_2_dropout"     :        tune.uniform(0.0, 0.75), 
            },
        "last_dense_layer_size"       :        64,
    }
    
    return cfg 