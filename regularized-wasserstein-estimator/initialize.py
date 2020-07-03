import numpy as np
from bisect import bisect_left

### INITIALIZATION OF THE ALGORITHM

def cumu_sum(dist):
  ''' Compute the cumulative distribution function for a discrete distribution '''
  n = len(dist)
  cumu_dist = []
  S = 0
  for i in range(n):
      S += dist[i]
      cumu_dist.append(S)
  return np.array(cumu_dist)

def random_int_list(dist, length, replace=True):
  ''' Pick a random list of integers with a given distribution '''
  rand_array = np.random.rand(length)
  i = 0
  cumu_dist = cumu_sum(dist)
  random_list = []
  for x in rand_array:
    i = bisect_left(cumu_dist, x)
    random_list.append(i)
  return random_list

def initialize(a, b, M, numItermax, batch_size):
  ''' Pick the intitial values for the internal variables of the algorithm, and construct the list of random indices to be used for sgd '''
  random_list_a = random_int_list(a, numItermax * batch_size)
  random_list_b = random_int_list(b, numItermax * batch_size)
  n_source = np.shape(M)[0]
  n_target = np.shape(M)[1]
  cur_alpha = np.zeros(n_source)
  cur_beta = np.zeros(n_target)
  cur_S = 1
  alpha_list = []
  beta_list = []
  time_list = []
  return random_list_a, random_list_b, cur_alpha, cur_beta, cur_S, alpha_list, beta_list, time_list