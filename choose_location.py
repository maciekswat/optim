import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from collections import OrderedDict
import sklearn.gaussian_process as gp


def target_noise(x):

    noise = np.random.normal(0, 0.2, 1)[0]
    return np.exp(-(x - 0.8) ** 2) + np.exp(-(x - 0.2) ** 2) - 0.5 * (x - 0.8) ** 3 - 2.0 + noise

def target_noise_2(x):

    noise = np.random.normal(0, 0.2, 1)[0]
    return np.exp(-(x - 0.8) ** 2) + np.exp(-(x - 0.2) ** 2) - 0.5 * (x - 0.8) ** 3 - 2.0 + noise



def target(x):


    return np.exp(-(x - 0.8) ** 2) + np.exp(-(x - 0.2) ** 2) - 0.5 * (x - 0.8) ** 3 - 2.0



def find_max(xp, yp, model):
    model.fit(xp,yp)
    evaluated_loss = model.predict(xp)
    opt_loc = np.argmax(evaluated_loss)
    x_max = xp[opt_loc]
    y_max, se = model.predict(x_max, return_std = True)
    return x_max,y_max, se

def choose_location(dataset_loc1, dataset_loc2, gp_params = None, alpha = 1e-3, epsilon = 1e-3, sig_level = 0.05):
    """
     Return a decision variable indicating which location to choose
        decision = 1, loc1 > loc2
                = 0 , inconclusive
                = -1, loc1 < loc2

    :param dataset_loc1: PS dataset for loc1
    :param dataset_loc2: PS dataset for loc2
    :param gp_params: params for gaussian process regressor (set to sklearn defaults)
    :param alpha:
    :param epsilon:
    :param sig_level: significant level for t-test

    """


    # set up kernels
    kernel_matern = Matern() + WhiteKernel(noise_level=1)
    model = gp.GaussianProcessRegressor(kernel=kernel_matern, alpha=alpha, n_restarts_optimizer= 5, normalize_y=True)


    # get predicted values for loc1
    xp_loc1 = np.array(map(lambda x: [x], dataset_loc1[:,0]))
    yp_loc1 = np.array(map(lambda y: [y], dataset_loc1[:,1]))


    x_max_loc1, y_max_loc1, se_1 = find_max(xp_loc1, yp_loc1, model)


    # get predicted values for loc2
    xp_loc2 = np.array(map(lambda x: [x], dataset_loc2[:,0]))
    yp_loc2 = np.array(map(lambda y: [y], dataset_loc2[:,1]))

    x_max_loc2, y_max_loc2, se_2 = find_max(xp_loc2, yp_loc2, model)

    t_stat = (y_max_loc1 - y_max_loc2)/np.sqrt(se_1**2 + se_2**2)
    p_val = norm.cdf(- np.abs(t_stat))

    if p_val < sig_level:
        if t_stat > 0:
            decision = {'Amplitude': x_max_loc1, 'Max Delta Classifier' : y_max_loc1, 'Location': 'Loc1', 'decision': 1, 'tie': 1}
            print "Loc 1 is better than Loc 2"
        else:
            print "Loc 2 is better than  Loc 1"
            decision = {'Amplitude': x_max_loc2, 'Max Delta Classifier': y_max_loc2 ,'Location': 'Loc2', 'decision': -1, 'tie': 1}

    else:
        if se_1 < se_2:
            print "Same but Loc1 is more reliable"
            decision = {'Amplitude': x_max_loc1, 'Max Delta Classifier' : y_max_loc1, 'Location': 'Loc1', 'decision': 1, 'tie': 1}
        else:
            print "Same but Loc2 is more reliable"
            decision = {'Amplitude': x_max_loc2, 'Max Delta Classifier': y_max_loc2 ,'Location': 'Loc2', 'decision': -1, 'tie': 1}


    result_dict = OrderedDict()
    result_dict ['decision'] = decision
    result_dict ['p_val'] = p_val
    result_dict ['loc1'] = {'Amplitude': x_max_loc1, 'Maximum Delta Classifier ' : y_max_loc1[0], 'se': se_1[0]}
    result_dict ['loc2'] = {'Amplitude': x_max_loc2, 'Maximum Delta Classifier' : y_max_loc2[0], 'se' : se_2[0]}

    return result_dict
    # return {'decision':decision, 'p_val': p_val, 'loc1':[y_max_loc1[0], se_1[0]], 'loc2':[y_max_loc2[0], se_2[0]]}


if __name__== '__main__':

    x = np.arange(0,2,0.001, dtype = np.float)

    x1 = np.arange(0,2, 1.0/100, dtype=np.float)
    # x1 = np.arange(250,1500, 1.0/100, dtype=np.float)

    y1 = target_noise(x1)
    x2 = np.copy(x1)
    y2 = target_noise_2(x2)

    y = target(x)
    print x[np.argmax(y)]
    print y[np.argmax(y)]

    # data_set1 = np.concatenate((x1,y1), axis =0 )
    #
    # data_set2 = np.concatenate((x2,y2), axis = 0)

    data_set1 = np.vstack((x1,y1)).T

    data_set2 = np.vstack((x2,y2)).T


    result = choose_location(data_set1, data_set2)

    print result

