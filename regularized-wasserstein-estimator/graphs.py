import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from computations import dual_to_target
from performance import norm_grad_dual

### DISPLAY OF GRAPHS
def SetPlotRC():
    #If fonttype = 1 doesn't work with LaTeX, try fonttype 42.
    plt.rc('pdf',fonttype = 42)
    plt.rc('ps',fonttype = 42)

def ApplyFont(ax):

    ticks = ax.get_xticklabels() + ax.get_yticklabels()

    text_size = 12

    for t in ticks:
        t.set_fontname('Times New Roman')
        t.set_fontsize(text_size)

    txt = ax.get_xlabel()
    txt_obj = ax.set_xlabel(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

    txt = ax.get_ylabel()
    txt_obj = ax.set_ylabel(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

    txt = ax.get_title()
    txt_obj = ax.set_title(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

def graph_loglog(list0, time=[]):
  ''' draw a log-log graph '''
  if not time:
    time = list(range(len(list0)))
  start = 0
  end = len(list0)
  list1 = [np.log(i+1) for i in range(end - start)]
  list2 = [np.log(gn) for gn in list0[start:]]
  slope, intercept, r_value, p_value, std_err = linregress(list1, list2)
  plt.plot(time, list0[start:])
  plt.xscale('log')
  plt.yscale('log')
  plt.grid()
  print('Convergence slope:', slope)

def performance_graphs(a, b, M, reg1, reg2, avg_alpha_list, avg_beta_list, time_list):
  alpha_fin = avg_alpha_list[-1]
  beta_fin = avg_beta_list[-1]
  G = np.exp((alpha_fin[:, None] + beta_fin[None, :] - M) / reg1) * a[:, None] * b[None, :]
  pi = G / G.sum()
  fig, ax = plt.subplots()
  ax.matshow(pi, cmap=plt.cm.Blues)
  plt.axis('off')
  # fig.savefig('tmp/transport_matrix_reg'+str(reg1).replace('.', '')+'.pdf', bbox_inches='tight')
  print("Transportation matrix")
  plt.show()

  n_source = np.shape(M)[0]
  target_list = [dual_to_target(b, reg2, beta) for beta in avg_beta_list]
  barWidth = 0.4
  r1 = list(range(n_source))
  r2 = [x + barWidth for x in r1]
  bar1 = a
  bar2 = target_list[-1]
  
  SetPlotRC()
  fig, ax = plt.subplots()
  plt.bar(r1, bar1, width=barWidth)
  plt.bar(r2, bar2, width=barWidth)
  print("Target measure and estimate")
  ApplyFont(plt.gca())
  # fig.savefig('tmp/solution_reg'+str(reg1).replace('.', '')+'.pdf', bbox_inches='tight')
  plt.show()

  grad_norm_list = [norm_grad_dual(a, b, target_list[i], M, reg1, avg_alpha_list[i], avg_beta_list[i]) for i in range(len(avg_alpha_list))]
  print("Final gradient norm:", grad_norm_list[-1])

  graph_loglog(grad_norm_list)
  print("Convergence rate of the gradient norm")
  plt.show()
  
def compare_results(list_results_alpha, list_results_beta):
  fig, ax = plt.subplots()
  SetPlotRC()
  for j in range(len(list_results_alpha)):
    avg_alpha_list =  list_results_alpha[j]
    avg_beta_list =  list_results_beta[j]
    target_list = [dual_to_target(b, reg2, beta) for beta in avg_beta_list]
    error_list = [kl_div(a, target_list[i]) for i in range(len(avg_alpha_list))]
    print("Final error:", error_list[-1])
    graph_loglog(error_list)
    labels=['c = 1/8', 'c = 1/4', 'c = 1/2', 'c = 1', 'c = 2']
    plt.legend(labels)
    plt.ylabel('KL Loss', fontsize=12)
    plt.xlabel('Number of iterations', fontsize=12)
    print("Convergence rate of the gradient norm")
  ApplyFont(plt.gca())
  # fig.savefig('tmp/lr_choice.pdf', bbox_inches='tight')
  plt.show()