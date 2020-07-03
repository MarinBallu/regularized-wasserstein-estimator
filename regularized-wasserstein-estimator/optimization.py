from initialize import initialize
import timeit
import numpy as np
import computations

### STOCHASTIC GRADIENT DESCENT

def sgd_entropic_regularization(a, b, M, reg1, reg2, numItermax, lr, maxTime):
  r'''
  Compute the sgd algorithm with one-sized batches to solve the regularized discrete measures
      ot dual estimation problem

  Parameters
  ----------
  a : ndarray, shape (ns,)
      source measure
  b : ndarray, shape (nt,)
      target measure
  M : ndarray, shape (ns, nt)
      cost matrix
  reg1, reg2 : float
      Regularization terms > 0
  numItermax : int
      number of iteration
  lr : float
      learning rate

  Returns
  -------
  alpha : ndarray, shape (ns,)
      dual variable
  beta : ndarray, shape (nt,)
      dual variable
  '''
  # Initialize variables
  random_list_a, random_list_b, cur_alpha, cur_beta, cur_S, alpha_list, beta_list, time_list = initialize(a, b, M, numItermax, 1)
  # Initialize time counter
  start = timeit.default_timer()
  for cur_iter in range(numItermax):
    # Receive the random indices
    i, j = random_list_a[cur_iter], random_list_b[cur_iter]
    # Compute the stepsize
    stepsize = lr / np.sqrt(cur_iter + 1)

    ## SGD
    # Compute gradients
    partial_target = computations.partial_target_meas(b[j], cur_beta[j], reg2, cur_S)
    grad_alpha, grad_beta = computations.partial_grad_dual(b[j], partial_target, M[i, j], reg1, cur_alpha[i], cur_beta[j])
    # Update dual variables
    cur_alpha[i], cur_beta[j], cur_S = computations.sgd_update(b[j], reg2, cur_alpha[i], cur_beta[j], cur_S, grad_alpha, grad_beta, stepsize)
    
    # Update memory for analysis
    alpha_list.append(np.array(cur_alpha))
    beta_list.append(np.array(cur_beta))
    t = timeit.default_timer() - start
    time_list.append(t)

    # Stopping time
    if maxTime and t > maxTime:
      break
  # Stop time counter
  stop = timeit.default_timer()
  # Print info
  print('Nb iter: ', cur_iter + 1)
  print('Time: ', stop - start)  
  print('Average iteration time: ', (stop - start) / numItermax)  
  # Return memory of dual variables and time
  return alpha_list, beta_list, time_list

### BATCHED GRADIENT DESCENT

def bgd_entropic_regularization(a, b, M, reg1, reg2, numItermax, batch_size, lr, maxTime):
  r'''
  Compute the batched sgd algorithm to solve the regularized discrete measures
      ot dual estimation problem

  Parameters
  ----------
  a : ndarray, shape (ns,)
      source measure
  b : ndarray, shape (nt,)
      target measure
  M : ndarray, shape (ns, nt)
      cost matrix
  reg1, reg2 : float
      Regularization terms > 0
  batch_size : int
      size of the batch
  numItermax : int
      number of iteration
  lr : float
      learning rate

  Returns
  -------
  alpha : ndarray, shape (ns,)
      dual variable
  beta : ndarray, shape (nt,)
      dual variable
  '''
  # Initialize variables
  random_list_a, random_list_b, cur_alpha, cur_beta, cur_S, alpha_list, beta_list, time_list = initialize(a, b, M, numItermax, batch_size)
  # Initialize time counter
  start = timeit.default_timer()
  for cur_iter in range(numItermax):
    # Receive the random batches of indices
    batch_a, batch_b = random_list_a[cur_iter * batch_size : (cur_iter + 1) * batch_size], random_list_b[cur_iter * batch_size : (cur_iter + 1) * batch_size]
    # Compute the stepsize
    stepsize = stepsize = min(lr / np.sqrt(cur_iter + 1), reg1) / batch_size
    
    ## SGD
    # Compute gradients
    partial_target = computations.partial_target_meas(b[batch_b], cur_beta[batch_b], reg2, cur_S)
    grad_alpha, grad_beta = computations.partial_grad_dual(b[batch_b], partial_target, M[batch_a, batch_b], reg1, cur_alpha[batch_a], cur_beta[batch_b])
    # Update dual variables
    cur_alpha, cur_beta, cur_S = computations.bgd_update(b, reg2, cur_alpha, cur_beta, cur_S, grad_alpha, grad_beta, batch_a, batch_b, stepsize)

    # Update memory for analysis
    alpha_list.append(np.array(cur_alpha))
    beta_list.append(np.array(cur_beta))
    t = timeit.default_timer() - start
    time_list.append(t)
    # Stopping time
    if maxTime and t > maxTime:
      break
  # Stop time counter
  stop = timeit.default_timer()
  # Print info
  print('Nb iter: ', cur_iter + 1)
  print('Time: ', stop - start)
  print('Average iteration time: ', (stop - start) / numItermax)
  # Return memory of dual variables and time
  return alpha_list, beta_list, time_list

### SEMI-STOCHASTIC GRADIENT DESCENT

def ssgd_entropic_regularization(a, b, M, reg1, reg2, numItermax, lr, maxTime):
  r'''
  Compute the semi-sgd algorithm to solve the regularized discrete measures
      ot dual estimation problem (sgd for alpha, full gradient for beta)

  Parameters
  ----------
  a : ndarray, shape (ns,)
      source measure
  b : ndarray, shape (nt,)
      target measure
  M : ndarray, shape (ns, nt)
      cost matrix
  reg1, reg2 : float
      Regularization terms > 0
  batch_size : int
      size of the batch
  numItermax : int
      number of iteration
  lr : float
      learning rate

  Returns
  -------
  alpha : ndarray, shape (ns,)
      dual variable
  beta : ndarray, shape (nt,)
      dual variable
  '''
  # Initialize variables
  random_list_a, useless_random_list_b, cur_alpha, cur_beta, cur_S, alpha_list, beta_list, time_list = initialize(a, b, M, numItermax, 1)
  # Initialize time counter
  start = timeit.default_timer()
  for cur_iter in range(numItermax):
    # Receive the random indices
    i = random_list_a[cur_iter]
    # Compute the stepsize
    stepsize = lr / np.sqrt(cur_iter + 1)

    ## SGD
    # Compute gradients
    target = computations.dual_to_target(b, reg2, cur_beta)
    grad_alpha, grad_beta = computations.semi_grad_dual(b, target, M[i], reg1, cur_alpha[i], cur_beta)
    # Update dual variables
    cur_alpha[i] += stepsize * grad_alpha
    cur_beta += stepsize * grad_beta
    
    # Update memory for analysis
    alpha_list.append(np.array(cur_alpha))
    beta_list.append(np.array(cur_beta))
    t = timeit.default_timer() - start
    time_list.append(t)
    # Stopping time
    if maxTime and t > maxTime:
      break
  # Stop time counter
  stop = timeit.default_timer()
  # Print info
  print('Nb iter: ', cur_iter + 1)
  print('Time: ', stop - start)  
  print('Average iteration time: ', (stop - start) / numItermax)  
  # Return memory of dual variables and time
  return alpha_list, beta_list, time_list