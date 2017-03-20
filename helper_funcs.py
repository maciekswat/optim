import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
# import matplotlib.pyplot as plt
from scipy.stats import signaltonoise
import time
#from numba import jit
# sample next value


# import line_profiler


def expected_improvement(x, gaussian_process, evaluated_loss, n_params, delta = 0.01):
    """
    :param x: next x
    :param gaussian_process: GaussianProcessRegressor object
    :param evaluated_loss: all the historical loss data
    :param n_params: dimension of X
    :param delta: offset to improve faster convergence
    :return: negative expected improvement
    """

    x_to_predict = x.reshape(-1, n_params)
    mu, sigma = gaussian_process.predict(x_to_predict, return_std = True)
    loss_max = np.max(evaluated_loss)
    scalar = -1.0
    with np.errstate(divide = 'ignore'):
        Z = (mu - loss_max - delta)/sigma
        # calculate expected improvement E[f(x) - f(x_best)]
        ei = scalar*((mu- loss_max - delta)*norm.cdf(Z) + sigma*norm.pdf(Z))
        if (sigma == 0):
            ei = 0

    #print 'negative expected improvement', -ei
    return ei   # return negative to use minimize function

def sample_next_value(acquisition_func, gaussian_process, evaluated_loss, n_params, bounds, delta, n_restart = 1):
    """

    :param acquisition_func: function to optimize for each step
    :param gaussian_process: GPRegressor object
    :param evaluated_loss:
    :param n_params:
    :param bounds:
    :param n_restart: number of starting initial values
    :return:

    """

    x_best = None
    best_acquisition_value = 1.0
    for starting_point in np.random.uniform(bounds[:,0], bounds[:,1], size = (n_restart,n_params)):
        #print starting_point

        res = minimize(acquisition_func, x0 = starting_point, bounds = bounds, method = 'L-BFGS-B',
                       args = (gaussian_process, evaluated_loss, n_params, delta))

        #print 'minimal ei', res.fun

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            x_best = res.x

    return x_best


def bayesian_optimization(n_iters, target_func, bounds, x0 = None, n_pre_samples = 10,
                          gp_params = None, alpha = 1e-4, epsilon = 1e-4):

    # using matern kernel with white noise
    n_params = bounds.shape[0]
    x_list = []
    y_list = []

    x_pre_samples = np.random.uniform(bounds[:,0], bounds[:,1], (n_pre_samples, bounds.shape[0]))



    for x in x_pre_samples:

        print type(x)
        x_list.append(x)
        y_list.append(target_func(x))

    print x_list
    print y_list

    xp = np.array(x_list)
    yp = np.array(y_list)


    print xp
    print yp

    kernel_matern = Matern() + WhiteKernel(noise_level=1)
    model = gp.GaussianProcessRegressor(kernel=kernel_matern, alpha=alpha, n_restarts_optimizer= 1, normalize_y=True)

    times = np.zeros(n_iters)

    delta0 = 0.02 # jump control

    for n in range(n_iters):
        #print 'iter: ', n
        start_time = time.time()
        delta = delta0
        model.fit(xp, yp)
        evaluated_loss = model.predict(xp)
        next_sample = sample_next_value(expected_improvement, model, evaluated_loss, n_params, bounds, delta)

        if(np.any(np.abs(next_sample - xp) <= epsilon)):
            next_sample =  np.random.uniform(bounds[:,0], bounds[:,1], bounds.shape[0])

        x_list.append(next_sample)
        y_list.append(target_func(next_sample))


        xp = np.array(x_list)
        yp = np.array(y_list)



        duration = time.time() - start_time
        times[n] = duration
        print 'duration: ', duration



    evaluated_loss = model.predict(xp)
    print type(model)
    print model.kernel_

    noise_level = model.kernel_.k2.noise_level

    return xp, yp, evaluated_loss, times, noise_level






def bayesian_optimization2D(n_iters, target_func, bounds, x0 = None, n_pre_samples = 10,
                          gp_params = None, alpha = 1e-4, epsilon = 1e-4):

    # using matern kernel with white noise
    n_params = bounds.shape[0]
    xy_list = []
    z_list = []

    xy_pre_samples = np.random.uniform(bounds[:,0], bounds[:,1], (n_pre_samples, bounds.shape[0]))

    for x in xy_pre_samples:

        xy_list.append(x)
        z_list.append(target_func(x[0],x[1]))


    xyp = np.array(xy_list)
    zp = np.array(z_list)


    kernel_matern = Matern() + WhiteKernel(noise_level=1)
    model = gp.GaussianProcessRegressor(kernel=kernel_matern, alpha=alpha, n_restarts_optimizer= 10, normalize_y=True)

    times = np.zeros(n_iters)

    delta0 = 0.02 # jump control

    for n in range(n_iters):
        #print 'iter: ', n

        start_time = time.time()

        delta = delta0
        model.fit(xyp, zp)
        evaluated_loss = model.predict(xyp)
        next_sample = sample_next_value(expected_improvement, model, evaluated_loss, n_params, bounds, delta)

        epsilon_vec = map(lambda x: np.sum((next_sample-x)**2), xyp)


        if(np.any(epsilon_vec <= epsilon)):
            next_sample =  np.random.uniform(bounds[:,0], bounds[:,1], bounds.shape[0])

        xy_list.append(next_sample)
        z_list.append(target_func(next_sample[0], next_sample[1]))


        xyp = np.array(xy_list)
        zp = np.array(z_list)



        duration = time.time() - start_time
        times[n] = duration
        print 'duration: ', duration



    evaluated_loss = model.predict(xyp)


    model = model

    return xyp, zp, evaluated_loss, times, model


def find_max(xp, yp, model):
    model.fit(xp,yp)
    model.fit(xp, yp)
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
            decision = {'Amplitude': x_max_loc1, 'Location': 'Loc1', 'decision': 1, 'tie': 0 }
            print "Loc 1 is better than Loc 2"
        else:
            decision = {'Amplitude': x_max_loc2, 'Location': 'Loc2', 'decision': -1, 'tie': 0}

            print "Loc 2 is better than  Loc 1"
    else:
        if se_1 < se_2:
            print "Same but Loc1 is more reliable"
            decision = {'Amplitude': x_max_loc1, 'Location': 'Loc1', 'decision': 1, 'tie': 1}
        else:
            print "Same but Loc2 is more reliable"
            decision = {'Amplitude': x_max_loc2, 'Location': 'Loc2', 'decision': -1, 'tie': 1}


    return {'decision':decision, 'p_val': p_val, 'loc1':[y_max_loc1[0], se_1[0]], 'loc2':[y_max_loc2[0], se_2[0]]}





def post_process(xp, yp, evaluated_loss, target, bounds, sigma, n_samp):

    # generate true function values
    x_samp = np.linspace(bounds[:,0], bounds[:,1], num = n_samp)
    x_samp = np.array(x_samp).reshape(len(x_samp),1)
    y_samp =np.apply_along_axis(target, 0, x_samp)

    x_true = []
    y_true = []
    x_grid = np.linspace(bounds[:,0], bounds[:,1], n_samp)
    for x in x_grid:
        x_true.append([x])
        y_true.append([target(x)])

    x_true = np.array(x_true)
    y_true = np.array(y_true)

    x_max = x_true[np.argmax(y_true)]

    print 'x_max,', x_max

    print np.mean(np.abs(y_true))

    SNR = np.mean(np.abs(y_true))/sigma
    SNR = round(SNR,3)
    print 'SNR, ', SNR



    # fig, ax = plt.subplots()
    #
    # ax.plot(x_true[:,0], y_true[:,0], color = 'blue', label = 'true function')
    #
    # indices = np.argsort(xp, axis = 0)
    #
    # ax.plot(xp[indices[:,0],0], evaluated_loss[indices[:,0],0], color = 'red', label = 'estimated function')
    #
    #
    # ax.scatter(xp, yp, color = 'green', label = 'sampled points')
    # ax.axvline(x = x_max, color = 'black')
    # ax.set_ylabel('y')
    # ax.set_xlabel('x')
    # ax.set_title('SNR = ' + str(SNR) +  ", Number of probed points = " + str(xp.shape[0]))
    # ax.text(0.0, -0.95, '$y = -2.0 + e^{-(x-0.5)^2} + e^{-0.5(x-0.2)^2} - 0.5(x-0.5)^2 + \epsilon $', fontsize=15)
    #
    # legend = ax.legend(loc = 'upper right')
    # frame = legend.get_frame()
    #
    # plt.show()
    #
    # fig.savefig('/Users/tungphan/Box Sync/Pres/SNR.' + str(SNR) + '.' + '5' + '.pdf', dpi = 1000, figsize = (8,6))

    return SNR








