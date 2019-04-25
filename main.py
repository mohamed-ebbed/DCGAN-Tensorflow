#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:47:12 2019

@author: mohamed
"""

import tensorflow as tf
from trainers.trainer import Trainer
from models.model import DCGAN
from data.data_generator import DataGenerator
from utils import config

config = config.process_config("config.json")
data = DataGenerator(config)
c = tf.ConfigProto()
c.gpu_options.allow_growth = True
sess = tf.Session(config=c)
model = DCGAN(config)
trainer = Trainer(config , model , data, sess)
trainer.train()