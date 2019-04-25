#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:46:53 2019

@author: mohamed
"""

import json
from bunch import Bunch
import os

def get_config_from_json(file):
    
    with open(file , 'r') as config_file:
        config_dict = json.load(config_file)
    
    config = Bunch(config_dict)
    
    return config , config_dict

def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.summaries_dir = os.path.join("../experiments", config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join("../experiments", config.exp_name, "checkpoint/")
    return config