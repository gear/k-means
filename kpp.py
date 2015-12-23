# Hoang NT
# 20151221 - Tokyo, Japan
# K-means++ implementation using Tensorflow

import tensorflow as tf
import random as rd
import numpy as np

class point(object):
# Container for data point. Add Variable node to graph

	def __init__(self, vector):
		assert type(vector) is np.ndarray
		self._vect = tf.Variable(vector, name="data_point")
		self._clus = -1    # Assign to default cluster number -1
		self._dist = -1	   # Distance to current cluster assignment

	@property
	def vect(self):
		return self._vect

	@property
	def clus(self):
		return self._clus

	@property
	def dist(self):
		return self._dist

def kpp(data, n_clusters):
	# Performs k-pp on a given data
	### INPUT
	# data: numpy 2-D array contains all data vectors
	# n_clusters: integer indicate number of clusters required
	### OUTPUT
	# clusters: dictionary {data index : cluster assignment}

	# Check data type and dimensions
	assert type(data) is np.ndarray 
	assert type(n_clusters) is int
	assert n_clusters < len(data)
	dim = data.shape[1];

	# Create list of data points and initial center
	vectors = [point(vect) for vect in data]
	c0_index = rd.randint(0,len(vectors)-1) 

	# Create list of centroids with initial first center
	centroids = [tf.Variable(tf.zeros([dim], tf.float64, name="centroid") for _ in range(n_clusters)]
	centroid_val = tf.placeholder(tf.float64, shape=[dim], name="centroid_pholder")
	centroid_assigners = [tf.assign(centroid, centroid_val) for centroid in centroids]
	
	# Euclid distance calculator
	
