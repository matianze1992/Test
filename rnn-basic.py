#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
=====================
rnn 单元的使用
=====================
基础使用
"""

import numpy as np
import tensorflow as tf

np.random.seed(0)
batch_size = 1

inputx = np.random.random([batch_size,10,6])

indata = tf.constant(inputx)
cell = tf.nn.rnn_cell.BasicRNNCell(6)
#cell = tf.nn.rnn_cell.BasicLSTMCell(6, state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([
    tf.nn.rnn_cell.BasicRNNCell(6),
    tf.nn.rnn_cell.BasicRNNCell(6)], state_is_tuple=True)

state = cell.zero_state(batch_size, tf.float64)   # y0
#状态向量

outputs = []
for time_step in range(10):
    #if time_step > 0: tf.get_variable_scope().reuse_variables()
    (cell_output, state) = cell(indata[:, time_step, :], state)
    outputs.append(cell_output)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for idx, itr in enumerate(sess.run(outputs)):
    print("step%d:"%idx, itr)

varlist = tf.trainable_variables()

w, b = sess.run(varlist)

s = np.zeros([1, 6])
for time_step in range(10):
    x = inputx[:, time_step, :]
    x = np.concatenate([x, s], axis=1)
    s = np.tanh(np.dot(x, w) + b)
    print("step%d"%time_step, s)
