#文本向量化
import numpy as np
import tensorflow as tf 

input_data = tf.constant([[1,2,3],[1,2,3]])
n_words = 5000
embedding_size = 256
w = tf.constant(np.ones([n_words, embedding_size]))
#[BatchSize=2,Time=3,EmbeddingSize=128]
inputs = tf.nn.embedding_lookup(w, input_data)

sess = tf.Session()
print(sess.run(inputs))