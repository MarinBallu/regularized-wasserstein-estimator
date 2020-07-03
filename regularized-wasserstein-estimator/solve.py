import optimization

### AVERAGING ALGORITHM

def avg_variable_list(list1):
  ''' Compute a list of the averages of all the items of a list '''
  list2 = [list1[0]]
  for i in range(len(list1))[1:]:
      tau = 1/(i + 1)
      list2.append(tau * list1[i] + (1 - tau) * list2[-1])
  return list2

def solve_dual_entropic(a, b, M, reg1, reg2, numItermax=10000, batch_size=1, lr=1, maxTime=None):
    r'''
    Compute the transportation matrix to solve the regularized discrete measures
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
    maxTime : float
        maximum run time for the gradient loop

    Returns
    -------
    '''
    if batch_size == 0:
      alpha_list, beta_list, time_list = optimization.ssgd_entropic_regularization(a, b, M, reg1, reg2, numItermax, lr, maxTime)
    elif batch_size == 1:
      alpha_list, beta_list, time_list = optimization.sgd_entropic_regularization(a, b, M, reg1, reg2, numItermax, lr, maxTime)
    else:
      alpha_list, beta_list, time_list = optimization.bgd_entropic_regularization(a, b, M, reg1, reg2, numItermax, batch_size, lr, maxTime)
    avg_alpha_list = avg_variable_list(alpha_list)
    avg_beta_list = avg_variable_list(beta_list)
    return avg_alpha_list, avg_beta_list, time_list