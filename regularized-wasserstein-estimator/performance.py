import numpy as np
from computations import dual_to_target

### PERFORMANCE MEASURES FOR GRAPHS AND MISC

def norm_grad_dual(a, b, target, M, reg, alpha, beta):
    r'''
    Compute the gradient L1 norm of the dual optimal transport problem.
    '''
    D = np.exp((alpha[:, None] + beta[None, :] - M[:,:]) / reg) * a[:, None] * b[None, :]
    grad_alpha = (a - D.sum(1))
    grad_beta = (target  - D.sum(0))
    return np.sqrt(np.linalg.norm(grad_alpha)**2 + np.linalg.norm(grad_beta)**2)

def scalar_loss(a, b, M, reg1, reg2, alpha, beta):
  ''' compute the scalar loss of the classical ot problem'''
  target = dual_to_target(b, reg2, beta)
  pi = (np.exp((alpha[:, None] + beta[None, :] - M[:, :]) / reg1) * a[:, None] * target[None, :])
  pi = pi/pi.sum()
  return (M * pi).sum()

def reg1_loss(a, b, M, reg1, reg2, alpha, beta):
  ''' compute the optimization loss of the current target compared to the uniform law '''
  target = dual_to_target(b, reg2, beta)
  pi = (np.exp((alpha[:, None] + beta[None, :] - M[:, :]) / reg1) * a[:, None] * target[None, :])
  pi = pi/pi.sum()
  return (np.log((pi == 0) + pi / (a[:, None] * b[None, :])) * pi).sum()

def reg2_loss(a, b, M, reg1, reg2, alpha, beta):
  ''' compute the kl loss of the current target compared to the uniform law '''
  target = dual_to_target(b, reg2, beta)
  return ((np.log((target == 0) + target / b)) * target).sum()

def kl_div(a, b):
  return ((np.log((a == 0) + a / b)) * a).sum()