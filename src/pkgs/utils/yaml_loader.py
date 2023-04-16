# """
# #++++++++++++++++++++++++++++++++++++++++++++++

#     Project: Part of final project for Georgia Tech Institute of Technology course DL, CS 7643. 

#     Totality of this code is non-proprietary and may be used at will. 

# #++++++++++++++++++++++++++++++++++++++++++++++


# Description: 

# @brief Defines customer yaml loader.

# @author: Greg Zdor (gzdor@icloud.com)

# @date Date_Of_Creation: 4/16/2023 

# @date Last_Modification 4/16/2023 

# No Copyright - use at will

# """

import yaml
import os

class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super(Loader, self).__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, Loader)

Loader.add_constructor('!include', Loader.include)

def custom_yaml_loader(fname):
    with open(fname, 'r') as f:
        data = yaml.load(f, Loader)
    return data