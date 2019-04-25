#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:37:20 2019

@author: mohamed
"""

import tensorflow as tf


class BaseModel:
    def __init__(self , config):
        self.config = config
        self.init_curr_epoch()
        self.init_global_step()
    
    def save(self , sess):
        self.saver.save(sess , self.config.checkpoint_dir , self.global_step)
        
    def load(self , sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if(latest_checkpoint):
            print("Loading model checkpoint {}".format(latest_checkpoint))
            self.saver.restore(sess , self.config.checkpoint_dir)
            print("Model loaded successfully")
    
    def init_curr_epoch(self):
        with tf.variable_scope("curr_epoch"):
            self.curr_epoch = tf.Variable(1 , trainable = False, name = 'cur_epoch')
            self.curr_epoch_increment = tf.assign(self.curr_epoch , self.curr_epoch+1)
    
    def init_global_step(self):
        with tf.variable_scope("global_step"):
            self.global_step = tf.Variable(0 , trainable = False , name = 'global_step')
        
    def init_saver(self):
        raise NotImplementedError
    
    def build_model(self):
        raise NotImplementedError