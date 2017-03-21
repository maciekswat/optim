import numpy as np
import time
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
# import matplotlib.pyplot as plt
from scipy.stats import signaltonoise
#from numba import jit
import helper_funcs
# import line_profiler


from helper_funcs import*
# import matplotlib
# matplotlib.rc('xtick', labelsize=20)
# matplotlib.rc('ytick', labelsize=20)
# simulation study


sigma = 0.3

def target_loc1(x):

    #return np.exp(-(x - 0.8) ** 2) + np.exp(-(x - 0.2) ** 2) - 0.5 * (x - 0.8) ** 3 - 2.0

    #return np.exp(-(x*10 - 2)**2) + np.exp(-(x*10 - 6)**2/10) + 1/ ((x*10)**2 + 1)
#    return 0.2*(np.exp(-(x-0.8)**2) + np.exp(-(x-0.2)**2) - 0.3*(x-0.8)**2 - 1.5) + 0.04
    return np.sin(x*10)

#    return np.exp(-(x-0.8)**2) + np.exp(-(x-0.2)**2) - 0.3*(x-0.8)**2 - 1.5

def target_loc2(x):
    #return np.exp(-(x - 0.8) ** 2) + np.exp(-(x - 0.2) ** 2) - 0.5 * (x - 0.8) ** 3 - 1.5

    return np.sin(x*10)

    #return np.exp(-(x*10 - 2)**2) + np.exp(-(x*10 - 6)**2/10) + 1/ ((x*10)**2 + 1)
    #return 0.2*(np.exp(-(x-0.8)**2) + np.exp(-(x-0.2)**2) - 0.3*(x-0.8)**2 - 1.5)
    #return np.sin(x*10)

#    return np.exp(-(x-0.8)**2) + np.exp(-(x-0.2)**2) - 0.3*(x-0.8)**2 - 1.5

def target_noise_loc1(x):

    noise = np.random.normal(0,sigma,1)[0]

    return target_loc1(x) + noise


def target_noise_loc2(x):

    noise = np.random.normal(0,sigma,1)[0]

    return target_loc2(x) + noise



bounds = np.array([[0.0,1.0]])
xp1, yp1, evaluated_loss1, times1, model_level1_noise = bayesian_optimization(n_iters = 100, target_func= target_noise_loc1, bounds = bounds, n_pre_samples = 26)

SNR_1 = np.mean(np.abs(yp1))/np.sqrt(model_level1_noise)
print "SNR_1", SNR_1

#
post_process(xp1,yp1, evaluated_loss1, target_loc1, bounds, sigma, 100)
#
#
bounds = np.array([[0.0,1.0]])
xp2, yp2, evaluated_loss2, times2, model_level2_noise = bayesian_optimization(n_iters = 100, target_func= target_noise_loc2, bounds = bounds, n_pre_samples = 26)
#
post_process(xp2,yp2, evaluated_loss2, target_loc2, bounds, sigma, 100)
#

SNR_2 = np.mean(np.abs(yp2))/np.sqrt(model_level2_noise)
print "SNR2", SNR_2

# two sample t-test


decision = choose_location(np.concatenate((xp1,yp1), axis = 1), np.concatenate((xp2,yp2), axis = 1), bounds)
print decision



x1 = np.arange(0,1, 1.0/1000, dtype=np.float)
y1 = np.zeros(len(x1))
# x1 = np.arange(250,1500, 1.0/100, dtype=np.float)
for i in xrange(len(x1)):
    y1[i] = target_noise_loc1(x1[i])


x2 = np.arange(0,1, 1.0/1000, dtype=np.float)
y2 = np.zeros(len(x2))
# x1 = np.arange(250,1500, 1.0/100, dtype=np.float)
for i in xrange(len(x1)):
    y2[i] = target_noise_loc2(x1[i])

# data_set1 = np.concatenate((x1,y1), axis =0 )
#
# data_set2 = np.concatenate((x2,y2), axis = 0)

data_set1 = np.vstack((x1,y1)).T

data_set2 = np.vstack((x2,y2)).T

print "test here"

result = choose_location(data_set1, data_set2)
print result