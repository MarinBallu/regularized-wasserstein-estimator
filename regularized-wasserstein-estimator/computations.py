import numpy as np

### INTERMEDIATE COMPUTATIONS FOR THE UPDATES

def dual_to_target(b, reg2, beta):
  ''' compute the target given the dual variable '''
  target = b * np.exp( - beta / reg2)
  target = target / target.sum()
  return target

def partial_target_meas(b, beta, reg2, S):
  ''' Compute one coefficient of the current target measure with one coefficient of the current dual variable, O(1)'''
  nu = b * np.exp(- beta / reg2) / S
  return nu

def partial_grad_dual(b, target, M, reg1, alpha, beta):
  ''' Compute one coefficient of the gradient for the dual variables, O(1) '''
  D = np.exp((alpha + beta - M) / reg1)
  grad_alpha = 1 - D
  grad_beta = target / b  - D
  return grad_alpha, grad_beta

def semi_grad_dual(b, target, M, reg1, alpha, beta):
  ''' Compute the stochastic gradients for the dual variable alpha and full gradient for beta '''
  D = np.exp((alpha + beta - M) / reg1) * b
  grad_alpha = 1 - D.sum()
  grad_beta = target  - D
  return grad_alpha, grad_beta

def sgd_update(b, reg2, alpha, beta, cur_S, grad_alpha, grad_beta, stepsize):
  ''' Update the dual variables as well as the latent memory-conserved variable '''
  cur_S -= b * np.exp(- beta / reg2)
  alpha += stepsize * grad_alpha
  beta += stepsize * grad_beta
  cur_S += b * np.exp(- beta / reg2)
  return alpha, beta, cur_S

def bgd_update(b, reg2, alpha, beta, cur_S, grad_alpha, grad_beta, batch_a, batch_b, stepsize):
  ''' Update the dual variables as well as the latent memory-conserved variable in the batch case '''
  batch_b_unique = list(np.unique(batch_b))
  cur_S -= (b[batch_b_unique] * np.exp(- beta[batch_b_unique] / reg2)).sum()
  for k in range(len(batch_a)):
    alpha[batch_a[k]] += stepsize * grad_alpha[k]
  for k in range(len(batch_b)):
    beta[batch_b[k]] += stepsize * grad_beta[k]
  cur_S += (b[batch_b_unique] * np.exp(- beta[batch_b_unique] / reg2)).sum()
  return alpha, beta, cur_S