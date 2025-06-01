
import os
import argparse
from argparse import ArgumentParser
import yaml
import numpy as np


class Parser(ArgumentParser):
    def __init__(self, yaml):
        super().__init__()
        self.yaml = yaml
        self.config = self.configure()

    def configure(self):
        with open(self.yaml, "r") as stream:
            dictionary = yaml.safe_load(stream)
        for item in list(dictionary.keys()):
            i_default = dictionary[item] 
            i_type = type(dictionary[item]) 
            if i_type == "bool":
                self.add_argument("--" + item, default=i_default, action="store_true")
            else:
                self.add_argument("--" + item, default=i_default, type=type(i_default))
        return self.parse_args(args=[])
