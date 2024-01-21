import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.framework import graph_util
from nets.inception_v3 import *
from sklearn.metrics.pairwise import cosine_similarity

sess = tf.Session()
input_tensor = tf.placeholder(shape=[None, 299, 299, 3], dtype=tf.float32)
slim = tf.contrib.slim
CHEKPOINT_FILE = "D:/Alexey/Projects/models/inception_v3.ckpt"
input_tensor = tf.placeholder(shape=[None, 299, 299, 3], dtype=tf.float32, name="input")
arg_scope = inception_v3_arg_scope()
with slim.arg_scope(arg_scope):
    logits, end_points = inception_v3(input_tensor ,1001, is_training=False)
# Restore variables from disk.
    saver = tf.train.Saver()
    saver.restore(sess, CHEKPOINT_FILE)
    print("Model restored.")
    #tf.train.write_graph(sess.graph_def, '.', 'waqas_graph.pb', False)
    output_nodes = ['InceptionV3/Predictions/Softmax']
    output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_nodes)
    tf.train.write_graph(output_graph_def, '.', 'inception_v3_08_16.txt', True)
    # tf.train.write_graph(sess.graph_def, '.', 'TF_inception_v3_08_16_graph.txt', True)
    print("Graph Freezed")
