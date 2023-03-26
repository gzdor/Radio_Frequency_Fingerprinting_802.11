"""
"""
from argparse import ArgumentParser
from glob import glob
import json
import os
from pathlib import Path

from numpy.random import Generator, MT19937, SeedSequence
import numpy as np
from tqdm import tqdm
import yaml


class SigMFDataMMapper(object):
    
    def __init__(self,
                 parameters):
        
        self.parameters = parameters.copy()

        sg = SeedSequence(self.parameters.get("rngseed", 932079473))
        self.rng = Generator(MT19937(sg))

        self.init_mmap_datadir_params()

        self.init_datafile_prefix_list()

        self.init_nsamples_per_file()

        self.init_file_datasplit_idx()
        
    def map_data(self):

        self.init_datasplit_labels()

        self.init_datasplit_data()

    def init_mmap_datadir_params(self):

        self.mmap_data_prefix =\
            self.parameters["datadir_pth"].split("neu_m044q5210" +
                                                 os.path.sep)[1]

        self.mmap_data_prefix = "-".join(os.path.split(self.mmap_data_prefix))

        pathobj = Path(os.path.join(self.parameters["base_mmapdir_pth"],
                                    self.mmap_data_prefix))

        if not pathobj.exists():
            pathobj.mkdir(parents=True)

        self.mmap_datadir_pth = str(pathobj)
        self.mmap_data_prefix += "_" + self.parameters["runid"]
        
    def init_datafile_prefix_list(self):

        datafile_prefix_list =\
            os.listdir(self.parameters['datadir_pth'])
        
        datafile_prefix_list = [os.path.join(self.parameters['datadir_pth'], file) for file in datafile_prefix_list if "sigmf-data" in file]

        datafile_prefix_list =\
            [elem.replace(".sigmf-data", "") for elem in datafile_prefix_list]

        self.datafile_prefix_label_map =\
            {key: value for value, key in enumerate(datafile_prefix_list)}

        labels_pth =\
            os.path.join(self.mmap_datadir_pth, 
                         self.mmap_data_prefix  + "_labels.json")

        with open(labels_pth, "wt") as h_file:

            json.dump(self.datafile_prefix_label_map, h_file)

    def init_nsamples_per_file(self):

        self.nsamples_per_file = {}

        glob_arg =\
            os.path.join(self.parameters["datadir_pth"],
                         "*" + self.parameters["runid"] + ".sigmf-meta")

        datafile_prefix_list =\
            os.listdir(self.parameters['datadir_pth'])
        metafile_list = [os.path.join(self.parameters['datadir_pth'], file) for file in datafile_prefix_list if "sigmf-meta" in file]

        for metafile_pth in metafile_list: # glob(glob_arg):

            key = os.path.split(metafile_pth)[1].replace(".sigmf-meta", "")

            with open(metafile_pth, "r") as h_file:

                key1 = "_metadata"
                key2 = "annotations"
                key3 = "core:sample_count"
                
                metadata = json.loads(h_file.read())

                self.nsamples_per_file[key] =\
                    int(metadata[key1][key2][0][key3] / 128)
                
    def init_file_datasplit_idx(self):
                
        training_precent = self.parameters.get("training_precent", 0.8)
        
        self.datasplit_idx = {"training": {},
                              "validation": {},
                              "test": {}}

        for prefix in self.nsamples_per_file:

            file_sample_idx = np.arange(self.nsamples_per_file[prefix])
            self.rng.shuffle(file_sample_idx)

            ntraining_samples =\
                int(self.nsamples_per_file[prefix] * training_precent)

            cur_training_idx = file_sample_idx[:ntraining_samples].copy()

            self.datasplit_idx["test"][prefix] =\
                file_sample_idx[ntraining_samples:].copy()

            ntraining_samples = int(ntraining_samples * training_precent)
            self.rng.shuffle(cur_training_idx)

            self.datasplit_idx["training"][prefix] =\
                cur_training_idx[:ntraining_samples].copy()

            self.datasplit_idx["validation"][prefix] =\
                cur_training_idx[ntraining_samples:]
            
        device_datasplit_nsamples = {}
        self.total_datasplit_nsamples = {}
            
        for key in self.datasplit_idx.keys():

            device_datasplit_nsamples =\
                {key: len(values) for key, values in
                 self.datasplit_idx[key].items()}

            self.total_datasplit_nsamples[key] =\
                np.sum(list(device_datasplit_nsamples.values()))

    def init_mmap_file_pth(self,
                           datasplit,
                           filetype):

        return os.path.join(self.mmap_datadir_pth,
                            self.mmap_data_prefix + "_" +
                            datasplit + f"_{filetype}.mmap")

    def init_datasplit_labels(self):

        for datasplit in self.datasplit_idx.keys():

            total_datasplit_samples = self.total_datasplit_nsamples[datasplit]

            mmap_labels_pth =\
                self.init_mmap_file_pth(datasplit,
                                        "labels")

            h_labels = np.memmap(mmap_labels_pth,
                                 dtype=np.int32,
                                 mode="w+",
                                 shape=(total_datasplit_samples, 1))

            write_idx = 0

            for file_prefix in self.datasplit_idx[datasplit].keys():


                for key in self.datafile_prefix_label_map.copy(): 
                    new_key = os.path.basename(key)
                    self.datafile_prefix_label_map[new_key] = self.datafile_prefix_label_map[key]

                device_label = self.datafile_prefix_label_map[file_prefix]
                device_nsamples = len(self.datasplit_idx[datasplit][file_prefix])

                h_labels[write_idx:(write_idx + device_nsamples)] = device_label
                write_idx += device_nsamples
                
    def init_datasplit_data(self):
        
        for datasplit in self.datasplit_idx.keys():

            print(f"Memory mapping {self.mmap_data_prefix}'s " +
                  f"{datasplit} data ....")

            total_datasplit_samples = self.total_datasplit_nsamples[datasplit]
            
            print(f' Current dataset split total number of samples: {total_datasplit_samples}\n')

            print(f'on datasplit {datasplit}')

            mmap_data_pth = self.init_mmap_file_pth(datasplit, "data")

            h_data = np.memmap(mmap_data_pth,
                               dtype=np.float32,
                               mode="w+",
                               shape=(total_datasplit_samples, 2, 128))            

            write_idx = 0


            all_sigmf_data_files = [f for f in os.listdir(self.parameters['datadir_pth']) if f.endswith('.sigmf-data')]
            all_sigmf_data_files = [os.path.join(self.parameters['datadir_pth'], file) for file in all_sigmf_data_files]

            for file_prefix in tqdm(all_sigmf_data_files): # enumerate(self.datafile_prefix_label_map.keys()):

                file_prefix = os.path.basename(file_prefix)
                file_prefix = os.path.splitext(file_prefix)[0]

                rawdata_file_pth =\
                    os.path.join(self.parameters["datadir_pth"],
                                 file_prefix + ".sigmf-data")

                with open(rawdata_file_pth, 'rb') as f: 
                    bytes_data = f.read()
                    device_data = np.frombuffer(bytes_data, dtype = np.complex128)

                device_data =\
                    device_data.reshape((self.nsamples_per_file[file_prefix], 128))

                device_data =\
                    device_data[self.datasplit_idx[datasplit][file_prefix], :]

                device_datasplit_nsamples = device_data.shape[0]

                device_data_i =\
                    np.real(device_data).reshape((device_datasplit_nsamples, 1, 128))

                device_data_q =\
                    np.imag(device_data).reshape((device_datasplit_nsamples, 1, 128))
        

                h_data[write_idx:(write_idx + device_datasplit_nsamples), :, :] =\
                    np.concatenate([device_data_i,
                                    device_data_q], axis=1)

                write_idx += device_datasplit_nsamples


def load_parameters():
    """
    https://realpython.com/command-line-interfaces-python-argparse/
    https://python.land/data-processing/python-yaml
    """
    parser = ArgumentParser(description="ORACLE Raw Data Memory Mapper")

    parser.add_argument("yaml_file",
                        type=str,
                        help="Memory mapping parameters *.yml file path")

    arguments = parser.parse_args()
    
    with open(arguments.yaml_file, "rt") as h_file:
        parameters = yaml.safe_load(h_file)

    return parameters


def memory_map_rundata():

    parameters = load_parameters()

    mmaper = SigMFDataMMapper(parameters)
    mmaper.map_data()


if __name__ == "__main__":

    memory_map_rundata()
