'''
#######################################################################
# Copyright (C) (1)                                                   #
# 2018 Hans Weytjens                                                  #
# Permission given to modify the code as long as you                  #
# mention the source clearly                                          #
#######################################################################
This program uses a substantial part of the code of the 
CH_06_ random_wakl program written by Shangtong Zhang
and Kenta Shimada, see declaration below.



#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# 0 is the left terminal state
# 6 is the right terminal state
# 1 ... 5 represents A ... E
VALUES = np.zeros(7)
VALUES[1:6] = 0.5
# For convenience, we assume all rewards are 0
# and the left terminal state has value 0, the right terminal state has value 1
# This trick has been used in Gambler's Problem
VALUES[6] = 1

# set up true state values
TRUE_VALUE = np.zeros(7)
TRUE_VALUE[1:6] = np.arange(1, 6) / 6.0
TRUE_VALUE[6] = 1

ACTION_LEFT = 0
ACTION_RIGHT = 1


# @values: current states value, will be updated if @batch is False
# @alpha: step size
# @batch: whether to update @values
def monte_carlo(values, alpha=0.1, batch=False):
    state = 3
    trajectory = [3]

    # if end up with left terminal state, all returns are 0
    # if end up with right terminal state, all returns are 1
    while True:
        if np.random.binomial(1, 0.5) == ACTION_LEFT:
            state -= 1
        else:
            state += 1
        trajectory.append(state)
        if state == 6:
            returns = 1.0
            break
        elif state == 0:
            returns = 0.0
            break

    if not batch:
        for state_ in trajectory[:-1]:
            values[state_] += alpha * (returns - values[state_])
    return values


def n_step_td(values, n=1, alpha=0.1 ):
    from collections import deque
    state = 3
    St = deque()
    St.append(state)
    Rt = deque()
    Rt.append(0)
    T = 99999999
    t = 0
    GAMMA = 1
    reward05 = 0
    reward6 = 1
    tau = 0
    while  tau <  T - 1:
        if t < T:
            if np.random.binomial(1, 0.5) == ACTION_LEFT:
                state -= 1
            else:
                state += 1
            St.append(state)
            if state != 6:
                Rt.append(reward05)
            else:
                Rt.append(reward6)
            if state == 0 or state == 6:         # terminal state
                T = t + 1
        tau = t - n + 1
        if tau >= 0:
            G = (np.array([Rt[i] * GAMMA ** (i-tau-1) for i in range(tau+1,1+min(tau+n, T))])).sum()
            if tau + n < T:
                G += GAMMA ** n * values[St[-1]]
            values[St[tau]] += alpha * (G - values[St[tau]])   
        t += 1
    return values


def n_step_td_error(values, n=1, alpha=0.1 ):
    from collections import deque
    state = 3
    St = deque()
    St.append(state)
    deltat = deque()
    deltat.append(0)
    T = 99999999
    t = 0
    GAMMA = 1
    reward = 0
    tau = 0
    while tau < T - 1:
        old_state = state
        if t < T:
            if np.random.binomial(1, 0.5) == ACTION_LEFT:
                state -= 1
            else:
                state += 1
            St.append(state)
            deltat.append(reward + GAMMA * values[state] - values[old_state] )
            if state == 0 or state == 6:         # terminal state
                T = t + 1
        tau = t - n + 1
        if tau >= 0:
            TD_error = (np.array([deltat[i] * GAMMA ** (i-tau-1) for i in range(tau+1,1+min(tau+n, T))])).sum()
            values[St[tau]] += alpha * TD_error   
        t += 1
    return values


# approach true values
def compute_state_value(method, n):
    episodes = [0, 1, 10, 100, 1000]
    current_values = np.copy(VALUES)
    plt.figure(1)
    for i in range(episodes[-1] + 1):
        if i in episodes:
            plt.plot(current_values, label=str(i) + ' episodes')
        if method == "TD":
            current_values = n_step_td(current_values, n=n)
        elif method == "TD_error":
            current_values = n_step_td_error(current_values, n=n)
    plt.plot(TRUE_VALUE, label='true values')
    plt.xlabel('state')
    plt.ylabel('estimated value')
    plt.title("method: "+method+", n="+ str(n))
    plt.legend()

# rms analysis
def rms_error():
    td_n = [1, 3, 10]
    td_error_n = [1, 3, 10]
    mc_alphas = [0.01, 0.05, 0.1]
    episodes = 100 + 1
    runs = 100
    for i, n in enumerate(td_n + td_error_n + mc_alphas):
        total_errors = np.zeros(episodes)
        if i < len(td_n):
            method = 'TD'
            linestyle = 'solid'
        elif i < len(td_n)+len(td_error_n):
            method = 'TD_error'
            linestyle = 'dotted'
        else:
            method = 'MC'
            linestyle = 'dashdot'
        for r in tqdm(range(runs)):
            errors = []
            current_values = np.copy(VALUES)
            for i in range(0, episodes):
                errors.append(np.sqrt(np.sum(np.power(TRUE_VALUE - current_values, 2)) / 5.0))
                if method == 'TD':
                    current_values = n_step_td(current_values, n=n)
                    param = "n"
                elif method == "TD_error":
                    current_values = n_step_td_error(current_values, n=n)
                    param = "n"                    
                else:
                    current_values = monte_carlo(current_values, alpha=n)
                    param = "alpha"
            total_errors += np.asarray(errors)
        total_errors /= runs
        plt.plot(total_errors, linestyle=linestyle, label=method + ', '+param+' = %.02f' % (n))
    plt.xlabel('episodes')
    plt.ylabel('RMS')
    plt.title("RMS comparison methods")
    plt.legend()


def exercise_7_2():
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    compute_state_value(method = "TD", n= 3)
    
    plt.subplot(2, 1, 2)
    compute_state_value(method = "TD_error", n = 3)
    
    plt.tight_layout()
    plt.savefig(PATH+'CH_07_exercise_7_2_a.png')

    plt.figure(figsize=(20, 20))
    #plt.subplot(3, 1, 3)
    rms_error()
    plt.tight_layout()
    
    plt.savefig(PATH+'CH_07_exercise_7_2_b.png')
    #plt.close()



if __name__ == '__main__':
    PATH = 'C:/Users/Hans/workspace/Reinforcement Learning/Files/images/Chapter 7/'
    exercise_7_2()