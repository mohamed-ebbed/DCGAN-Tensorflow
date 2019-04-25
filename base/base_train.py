#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:00:42 2019

@author: mohamed
"""

import tensorflow as tf

class BaseTrain:
    def __init__(self , config , model , data , sess):
        self.model = model
        self.config = config
        self.data = data.build_dataset()
        self.data_iterator = self.data.make_one_shot_iterator()
        self.next_batch = self.data_iterator.get_next()
        self.sess = sess
        self.summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.config.summaries_dir+"/train" , sess.graph)
        self.init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
        sess.run(self.init)
        
    def train(self):
        for curr_epoch in range(self.sess.run(self.model.curr_epoch) , self.config.num_epochs+1):
            self.train_epoch()
            self.sess.run(self.model.curr_epoch_increment)
    
    def train_epoch(self):
        raise NotImplementedError
    
    def train_step(self):
        raise NotImplementedError
        
