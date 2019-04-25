#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:13:52 2019

@author: mohamed
"""
import tensorflow as tf
import os

class DataGenerator:
    def __init__(self , config):
        self.config = config
        self.images = self.get_images()
        self.data_size = len(self.images)
    def get_images(self):
        data_dir = self.config.data_dir
        files = os.listdir(data_dir)
        files = ["{}/{}".format(data_dir , f) for f in files]
        return files
    def process_image(self , img_dir):
        img_string = tf.read_file(img_dir)
        img_decoded = tf.image.decode_jpeg(img_string , channels = 3)
        img_resized = tf.image.resize_images(img_decoded , [self.config.img_size , self.config.img_size])
        img_resized = img_resized / 255.0
        img_resized = img_resized* 2 - 1
        curr_noise = tf.random.uniform([self.config.noise_shape] , -1 , 1)
        next_noise = tf.random.uniform([self.config.noise_shape] , -1 , 1)
        return img_resized , curr_noise , next_noise
    def build_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.images)
        dataset = dataset.shuffle(buffer_size = self.data_size)
        dataset = dataset.repeat(self.config.num_epochs)
        dataset = dataset.map(self.process_image , num_parallel_calls = 12)
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(1)
        return dataset