#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:39:22 2019

@author: mohamed
"""

import tensorflow as tf
from base.base_model import BaseModel

class DCGAN(BaseModel):
    def __init__(self , config):
        super(DCGAN , self).__init__(config)
        self.build_model()
        self.init_saver()
    
    def build_model(self):
        self.z = tf.placeholder(tf.float32 , shape = [None, 100])
        self.x = tf.placeholder(tf.float32 , shape= [None, self.config.img_size, self.config.img_size , 3])
        self.is_training = tf.placeholder(tf.bool)
        fake_images = self._build_generator(self.z)
        tf.summary.image("generated" , fake_images , max_outputs = 10)
        disc_real_output= self._build_discriminator(self.x)
        disc_fake_output = self._build_discriminator(fake_images , reuse = True)
        vars = tf.trainable_variables()
        gvars = [var for var in vars if "gen" in var.name]
        dvars = [var for var in vars if "disc" in var.name]
        
        with tf.name_scope("desc_loss"):
            dreal_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_real_output,
                                                                 labels = tf.ones_like(disc_real_output)))
            dfake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_fake_output,
                                                                 labels = tf.zeros_like(disc_fake_output)))
            self.dloss = dreal_loss + dfake_loss
            tf.summary.scalar("disc_loss" , self.dloss)
        with tf.name_scope("gen_loss"):
            self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_fake_output ,
                                                                    labels = tf.ones_like(disc_fake_output)))
            tf.summary.scalar("gen_loss" , self.gen_loss)
            
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        gen_update_ops = [op for op in update_ops if "gen" in op.name]
        disc_update_ops = [op for op in update_ops if "disc" in op.name]
        
        with tf.control_dependencies(gen_update_ops):
            self.gen_step = tf.train.AdamOptimizer(learning_rate= self.config.lr,
                                                   beta1 = self.config.beta1,).minimize(self.gen_loss, var_list = gvars
                                                                             , global_step = self.global_step)
        
        with tf.control_dependencies(disc_update_ops):
            self.disc_step = tf.train.AdamOptimizer(learning_rate= self.config.lr,
                                                    beta1 = self.config.beta1).minimize(self.dloss , var_list = dvars,
                                                                             global_step = self.global_step)
        
        
        
        
    def _build_generator(self, z , reuse = None):
        with tf.variable_scope("gen" , reuse = reuse):
            img_size = self.config.img_size
            s16 = img_size // 16
            gf_dim = self.config.gf_dim
            with tf.name_scope("project_reshape"):
                z = tf.layers.Dense(units = s16*s16*gf_dim*8)(z)
                z = tf.keras.layers.Reshape([s16,s16,gf_dim*8])(z)
                z = tf.layers.batch_normalization(z, training = self.is_training)
                z = tf.nn.relu(z)
            with tf.name_scope("conv1"):
                z = tf.layers.Conv2DTranspose(filters = gf_dim * 4 , kernel_size = 5 , strides = 2, padding="same")(z)
                z = tf.layers.batch_normalization(z, training = self.is_training)
                z = tf.nn.relu(z)
            with tf.name_scope("conv2"):
                z = tf.layers.Conv2DTranspose(filters = gf_dim * 2 , kernel_size = 5 , strides = 2, padding="same")(z)
                z = tf.layers.batch_normalization(z, training = self.is_training)
                z = tf.nn.relu(z)
            with tf.name_scope("conv3"):
                z = tf.layers.Conv2DTranspose(filters = gf_dim , kernel_size = 5, strides = 2, padding="same")(z)
                z = tf.layers.batch_normalization(z, training = self.is_training)
                z = tf.nn.relu(z)
            with tf.name_scope("output"):
                z = tf.layers.Conv2DTranspose(filters = 3 , kernel_size = 5 , strides = 2 , 
                                              padding = "same", activation = tf.nn.tanh)(z)
        return z
    
    def _build_discriminator(self , x , reuse = None):
        lrelu = lambda x : tf.nn.leaky_relu(x , alpha = 0.2)
        df_dim = self.config.df_dim
        with tf.variable_scope("disc" , reuse = reuse):
            with tf.name_scope('conv1'):
                x = tf.layers.Conv2D(filters = df_dim , kernel_size = 5 ,  padding = "same" , strides = 2 , 
                                     activation = lrelu)(x)
            with tf.name_scope('conv2'):
                x = tf.layers.Conv2D(filters = df_dim*2 , kernel_size = 5 , strides = 2 , padding = "same")(x)
                x = lrelu(tf.layers.batch_normalization(x, training = self.is_training))
            with tf.name_scope('conv3'):
                x = tf.layers.Conv2D(filters = df_dim*4 , kernel_size = 5 , strides = 2 , padding = "same")(x)
                x = lrelu(tf.layers.batch_normalization(x, training = self.is_training))
            with tf.name_scope('conv4'):
                x = tf.layers.Conv2D(filters = df_dim*8 , kernel_size = 5 , strides = 2 , padding = "same")(x)
                x = lrelu(tf.layers.batch_normalization(x, training = self.is_training))
            with tf.name_scope('output'):
                x = tf.layers.Flatten()(x)
                x = tf.layers.Dense(units = 1)(x)
        return x

            
    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep = self.config.max_to_keep)
    