#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:52:58 2019

@author: mohamed
"""

import tensorflow as tf
from base.base_train import BaseTrain
from tqdm import trange
import math
class Trainer(BaseTrain):
    def __init__(self , config , model , data , sess):
        super(Trainer , self).__init__(config , model , data ,sess)
        self.steps_per_epoch = int(math.ceil(data.data_size) / self.config.batch_size)
    def train_epoch(self):
        for _ in trange(self.steps_per_epoch):
            self.train_step()
        self.model.save(self.sess)
    
    def train_step(self):
        global_step = self.sess.run(self.model.global_step)
        real_images , curr_noise , next_noise  = self.sess.run(self.next_batch)
        feed_dict = {self.model.z : curr_noise , self.model.x : real_images , self.model.is_training : True}
        if global_step % self.config.summaries_period == 0:
            _ , summaries = self.sess.run([self.model.disc_step , self.summaries] , feed_dict = feed_dict)
            self.train_writer.add_summary(summaries , global_step)
        else:
            self.sess.run(self.model.disc_step , feed_dict = feed_dict)
        feed_dict = {self.model.z : next_noise , self.model.is_training:True}
        self.sess.run(self.model.gen_step , feed_dict = feed_dict)
        
