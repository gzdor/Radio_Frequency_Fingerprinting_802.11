#!/usr/bin/env python3
"""
#++++++++++++++++++++++++++++++++++++++++++++++

    Project: Part of final project for Georgia Tech Institute of Technology course DL, CS 7643. 

    Totality of this code is non-proprietary and may be used at will. 

#++++++++++++++++++++++++++++++++++++++++++++++


Description: 

@brief a module defining a variety of file loading and manipulation utilities. 

@author: Greg Zdor (gzdor@icloud.com)

@date Date_Of_Creation: 3/15/2023 

@date Last_Modification 3/15/2023 

No Copyright - use at will

"""

import json
import numpy as np 


# Define SigMF data parsing utility 
def load_sigmf_iq_meta(data_path_file: str, metadata_path_file: str) -> dict: 
    """
    @brief intakes the paths to SigMF data and 
    metadata files and returns a dictionary containing 
    the output raw samples and associated metadata. 


    @type  data_path_file str
    @param data_path_file path to SigMF data file 

    @type  metadata_path_file str
    @param metadata_path_file path to SigMF metadata file

    @type output dict
    @return output dictionary where raw samples are under output['samples']
    and metadata is under output['metadata']

    """
    
    output = {}


    # Load metadata
    with open(metadata_path_file, 'r') as f: 
        output['metadata'] = json.loads(f.read())

    # Load samples 
    with open(data_path_file, 'rb') as f: 
        bytes_data = f.read()
        output['samples'] = np.frombuffer(bytes_data, dtype = np.complex128)

    return output