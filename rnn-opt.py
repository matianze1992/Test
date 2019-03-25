#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
=====================
rnn 单元的使用
=====================
使用辅助函数优化程序流程
"""

import numpy as np
import tensorflow as tf

np.random.seed(0)
indata = tf.constant(np.random.random([1,10,6]))
cell = tf.nn.rnn_cell.BasicLSTMCell(6, state_is_tuple=True)
#cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2, state_is_tuple=True)

state = cell.zero_state(1, tf.float64)

outputs, last_state = tf.nn.dynamic_rnn(cell, indata, initial_state=state)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(outputs))