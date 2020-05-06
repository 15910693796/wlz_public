#import
import random
import tensorflow as tf
import numpy as np
import pandas as pd

#class
class dqn():
    def __init__(self,scope):
        self.scope = scope
        self.sess = tf.Session()
        self.random_seed = 1234
        self.greedy=0.99
        self.gamma = 0.90
        self.memory_size=200
        self.batch_size=32
        self.n_actions = 4
        self.n_features = 50*50*3
        self.lr = 0.01
        self.width = 50
        self.height = 50
        self.color = 3
        with tf.variable_scope(scope):
            self.x = tf.placeholder(tf.float32, shape = (None,self.height,self.width,self.color),name = 'x')
            self.g = tf.placeholder(tf.float32, shape = (None,),name = 'g')
            self.actions = tf.placeholder(tf.int32, shape = (None,),name = 'actions')
            z = self.x/255.0
            cnn1 = tf.contrib.layers.conv2d(z,32,8,4,activation_fn = tf.nn.relu)

    