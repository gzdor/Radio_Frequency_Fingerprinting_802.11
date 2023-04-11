"""
"""
from argparse import ArgumentParser
from collections import defaultdict
import json
from joblib import dump, load
import re
import os 
from pathlib import Path

from numpy.random import Generator, MT19937, SeedSequence
import numpy as np
from tqdm import tqdm
import yaml
import pdb


class SigMFDataMMapper(object):
    """
    Class that generates numpy memory mapped files to store "raw IQ samples
    of over-the-air transmissions from 16 X310 USRP radios" from the
    Northeastern University Institute for the Wireless Internet of Things
    Genesys Lab ORACLE RF Fingerprinting Dataset.
    https://genesys-lab.org/oracle
    """    

    def __init__(self,
                 parameters):
        """
        Class constructor
        """
        self.parameters = parameters.copy()
        self.nsamples_per_file = {}
        self.datafile_paths = defaultdict(list)
        
        sg = SeedSequence(self.parameters.get("rngseed", 932079473))
        self.rng = Generator(MT19937(sg))
        
        self.initialize_mmap_parameters()
        
    def initialize_mmap_parameters(self):
        """
        """
        self.mmap_dir =\
            self.parameters["base_mmapdir_pth"]
        
        print(f'\nmemmap base path: self.mmap_dir:  {self.mmap_dir}')

        # self.mmap_dir =\
        #     self.mmap_dir.joinpath(self.parameters["experimentid"])
        
        self.mmap_dir =\
            os.path.join(self.mmap_dir , self.parameters["experimentid"])
        
        self.mmap_dir = Path(self.mmap_dir)

        if not self.mmap_dir.exists():
            self.mmap_dir.mkdir(parents=True)
        
        params_copy_path = self.mmap_dir.joinpath("parameters.yaml")

        if not params_copy_path.exists():
            self.save_parameters(params_copy_path)

        self.initialize_memory_mapping()

        self.initialize_deviceid_number_map()

    def save_parameters(self,
                        params_copy_path):

        params_copy = self.parameters.copy()
                
        for key in params_copy.keys():

            if str(type(params_copy[key])).find("pathlib") != -1:
                params_copy[key] = str(params_copy[key])

        with open(params_copy_path, "wt") as h_file:
            yaml.dump(params_copy, h_file)
        
    def initialize_memory_mapping(self):
        """
        """
        indexing_jl_pth =\
            self.mmap_dir.joinpath("rawdata_indexing.joblib")

        # https://stackoverflow.com/questions/56312742/
        #    can-use-joblib-dump-many-object
        if indexing_jl_pth.exists():

            (self.datafile_paths,
             self.nsamples_per_file,
             self.rawdata_indexing) = load(indexing_jl_pth)
        # ----------------------------------------------------------
        else:

            for datasplit_key in ["train_val_data",
                                  "test_data"]:

                self.initialize_datafile_paths(datasplit_key)

            for data_pth in self.datafile_paths["test_data"]:

                assert data_pth not in self.datafile_paths["train_val_data"],\
                    f"{data_pth} is included in both training & test data"

            self.init_rawdata_indexing()

            dump([self.datafile_paths,
                  self.nsamples_per_file,
                  self.rawdata_indexing],
                  indexing_jl_pth)
            
    def init_rawdata_indexing(self):
        """
        """
        test_set_only = self.parameters.get("test_set_only", False)

        training_precent =\
            self.parameters.get("training_precent", 0.8)

        test_data_mmap_frac =\
            self.parameters.get("test_data_mmap_frac", 0.2)

        self.rawdata_indexing = {"training": {},
                                 "validation": {},
                                 "test": {}}

        h_get_prefix =\
            lambda elem: elem.parts[-1].replace(".sigmf-data", "")

        for datasplit in self.datafile_paths.keys():

            for cur_pth in self.datafile_paths[datasplit]:

                prefix = h_get_prefix(cur_pth)

                file_sample_idx =\
                    np.arange(self.nsamples_per_file[prefix])
                
                self.rng.shuffle(file_sample_idx)

                # ------------------------------------------------------
                # Default is 80% / 20% training & validation data split
                # ------------------------------------------------------
                if datasplit == "train_val_data":

                    ntraining_samples =\
                        int(self.nsamples_per_file[prefix] *\
                            training_precent)

                    # ------------------------------------------------------
                    # Work around to minimize memory mapper complexity
                    # ------------------------------------------------------
                    if test_set_only:

                        self.rawdata_indexing["training"][prefix] =\
                            file_sample_idx[:1].copy()

                        self.rawdata_indexing["validation"][prefix] =\
                            file_sample_idx[:1].copy()
                    # -------------------------------------------------
                    else:
                        self.rawdata_indexing["training"][prefix] =\
                            file_sample_idx[:ntraining_samples].copy()

                        self.rawdata_indexing["validation"][prefix] =\
                            file_sample_idx[ntraining_samples:].copy()
                # ------------------------------------------------------
                # Default is to use 20% of test data samples
                # ------------------------------------------------------
                elif datasplit == "test_data":
            
                    ntest_samples =\
                        int(self.nsamples_per_file[prefix] *\
                            test_data_mmap_frac)
            
                    self.rawdata_indexing["test"][prefix] =\
                        file_sample_idx[:ntest_samples].copy()

    def initialize_datafile_paths(self,
                                  datasplit_key):
        """
        """
        for dist_key in self.parameters[datasplit_key].keys():

            distance_pth = self.parameters["datadir_pth"].joinpath(dist_key)

            for run_key in self.parameters[datasplit_key][dist_key]:

                cur_data_files =\
                    list(distance_pth.glob("*" + run_key + "*.sigmf-data"))

                for cur_data_pth in cur_data_files:
                
                    cur_meta_pth = cur_data_pth.with_suffix(".sigmf-meta")
                    
                    data_file_prefix =\
                        cur_meta_pth.parts[-1].replace(".sigmf-meta", "")
                    
                    with open(cur_meta_pth, "r") as h_file:

                        key1 = "_metadata"
                        key2 = "annotations"
                        key3 = "core:sample_count"

                        metadata = json.load(h_file)

                        self.nsamples_per_file[data_file_prefix] =\
                            int(metadata[key1][key2][0][key3] / 128)
                    
                self.datafile_paths[datasplit_key].extend(cur_data_files)
            
    @staticmethod
    def initialize_device_regexp():
        """
        """
        return re.compile(r"WiFi_air_X310_(?P<deviceid>[0-9A-Z]" +
                          r"{7})_[0-9]+ft_run[0-9]+$")
    
    def initialize_deviceid_number_map(self):
        """
        """
        json_file_pth =\
            self.mmap_dir.joinpath("deviceid_number_map.json")
        
        if json_file_pth.exists():
            
            with open(json_file_pth, "r") as h_file:
                self.deviceids = json.load(h_file)
        # ---------------------------------------------
        else:
            po = SigMFDataMMapper.initialize_device_regexp()

            self.deviceids =\
                [po.match(elem).groupdict()["deviceid"]
                 for elem in self.nsamples_per_file.keys()]

            self.deviceids = set(self.deviceids)
        
            self.deviceids =\
                {key: index for index, key in enumerate(self.deviceids)}
            
            with open(json_file_pth, "w") as h_file:
                json.dump(self.deviceids, h_file)
                
    def get_total_datasplit_nsamples(self):
        """
        """
        total_datasplit_nsamples = defaultdict(int)

        for datasplit in self.rawdata_indexing.keys():

            for prefix in self.rawdata_indexing[datasplit].keys():
        
                cur_nsamples =\
                    len(self.rawdata_indexing[datasplit][prefix])
        
                total_datasplit_nsamples[datasplit] += cur_nsamples

        return total_datasplit_nsamples
    
    def get_mmap_data_dir(self):
        """
        """
        key1 = "base_mmapdir_pth"
        key2 = "experimentid"

        # mmap_data_dir =\
        #     self.parameters[key1].joinpath(self.parameters[key2])

        mmap_data_dir =\
            os.path.join(self.parameters[key1], self.parameters[key2])
        
        mmap_data_dir = Path(mmap_data_dir)

        if not mmap_data_dir.exists():
            mmap_data_dir.mkdir(parents=True)

        return mmap_data_dir
    
    def get_mmap_data_file(self,
                           datasplit):
        
        mmap_data_file =\
                self.get_mmap_data_dir().joinpath(datasplit +
                                                  "_data.mmap")
        return mmap_data_file
    
    def get_mmap_labels_file(self,
                             datasplit):
        
        mmap_labels_file =\
                self.get_mmap_data_dir().joinpath(datasplit +
                                                  "_labels.mmap")
        
        return mmap_labels_file

    def get_h_data(self,
                   datasplit,
                   mode):
        """
        """
        total_datasplit_nsamples = self.get_total_datasplit_nsamples()

        h_data =\
            np.memmap(self.get_mmap_data_file(datasplit),
                      dtype=np.float32,
                      mode=mode,
                      shape=(total_datasplit_nsamples[datasplit], 2, 128))        

        return h_data

    def get_h_labels(self,
                     datasplit,
                     mode):
        """
        """
        total_datasplit_nsamples = self.get_total_datasplit_nsamples()
        
        h_labels =\
            np.memmap(self.get_mmap_labels_file(datasplit),
                      dtype=np.int32,
                      mode=mode,
                      shape=(total_datasplit_nsamples[datasplit], 1))        

        return h_labels

    def map_datasplit_data(self,
                           datasplit):
        
        data_ready =\
            self.get_mmap_data_file(datasplit).exists() and\
            self.get_mmap_labels_file(datasplit).exists()
        
        if self.parameters.get("overwrite_data", False):
            data_ready = False
        
        return data_ready is False

    def map(self):

        po = SigMFDataMMapper.initialize_device_regexp()

        for datasplit in self.rawdata_indexing.keys():

            if self.map_datasplit_data(datasplit):

                h_data = self.get_h_data(datasplit, "w+")
                h_label = self.get_h_labels(datasplit, "w+")

                start_write_idx = 0
            
                for measurementid in\
                    tqdm(self.rawdata_indexing[datasplit].keys()):

                    deviceid =\
                        po.match(measurementid).groupdict()["deviceid"]

                    meas_nsamples =\
                        len(self.rawdata_indexing[datasplit][measurementid])

                    meas_indexing =\
                        self.rawdata_indexing[datasplit][measurementid]
                
                    h_data[start_write_idx:(start_write_idx +\
                                            meas_nsamples)] =\
                        self.get_device_data(measurementid,
                                             meas_indexing)

                    h_label[start_write_idx:(start_write_idx +\
                                             meas_nsamples)] =\
                        self.deviceids[deviceid]
                
                    start_write_idx += meas_nsamples

    def get_device_data(self,
                        measurementid,
                        meas_indexing):

        measid_po =\
            re.compile(r"(?P<prefix>WiFi_air_X310_[0-9A-Z]{7})" +
                       r"_(?P<distance>[0-9]+ft)_(?P<run>run[0-9]+)$")

        meas_params = measid_po.match(measurementid).groupdict()

        rawdata_file_pth =\
            self.parameters["datadir_pth"].joinpath(meas_params["distance"],
                                                    measurementid +
                                                    ".sigmf-data")
    
        with open(rawdata_file_pth, 'rb') as f: 
            bytes_data = f.read()
            device_data = np.frombuffer(bytes_data, dtype = np.complex128)

        device_data =\
            device_data.reshape((self.nsamples_per_file[measurementid], 128))

        device_data = device_data[meas_indexing, :]
        device_datasplit_nsamples = device_data.shape[0]

        device_data_i =\
            np.real(device_data).reshape((device_datasplit_nsamples, 1, 128))

        device_data_q =\
            np.imag(device_data).reshape((device_datasplit_nsamples, 1, 128))
        
        # Need to normalize the average envelope when processing data
        # across multiple distances (RF path loss is a function of
        # transmitter / receiver distance)
        average_envelope = self.parameters.get("average_envelope", None)
        
        if average_envelope is not None:
            
            device_data_envelope =\
                np.sqrt(np.power(device_data_i, 2) +\
                        np.power(device_data_q, 2))
            
            norm_factor =\
                average_envelope / device_data_envelope.mean(axis=2)
            
            norm_factor = norm_factor.reshape(-1,1,1)

            device_data_i = device_data_i * norm_factor 
            device_data_q = device_data_q * norm_factor
        
        return np.concatenate([device_data_i, device_data_q], axis=1)


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
    
    with open(Path(arguments.yaml_file), "rt") as h_file:
        parameters = yaml.safe_load(h_file)

    for key in [elem for elem in parameters.keys()
                if elem.find("pth") != -1]:
        
        parameters[key] = Path(parameters[key])

    return parameters


def memory_map_rundata():

    parameters = load_parameters()

    mmaper = SigMFDataMMapper(parameters)
    mmaper.map()


if __name__ == "__main__":

    memory_map_rundata()