'''
#######################################################################
# Copyright (C)                                                       #
# 2018 Hans Weytjens ()                                               #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
'''


def read_racetrack(tr_nr):
    #define racetrack (left)
    # 1 is racetrack, 0 is off-track, 3 is start, 4 is finish 
    if tr_nr == 1:
        HEIGHT = 32
        WIDTH = 17
        RACETRACK = np.ones((HEIGHT,WIDTH), int)
        # define off-track fields
        RACETRACK[0:26,9:17] = 0
        RACETRACK[25,9] = 1
        RACETRACK[0:18,0] = 0
        RACETRACK[0:10,1] = 0   
        RACETRACK[0:4,2] = 0
        # define starting line
        START = []
        for j in range(3,9):
            RACETRACK[0,j] = 2
            START.append([0,j])
        # define finish line
        FINISH = []
        for i in range(26,32):
            RACETRACK[i,16] = 3
            FINISH.append([i,16])
    else:
        HEIGHT = 30
        WIDTH = 32
        RACETRACK = np.ones((HEIGHT,WIDTH), int)
        # define off-track fields
        for i in range(0,17):
            for j in range(23,WIDTH):
                RACETRACK[i,j] = 0
        RACETRACK[17,24:WIDTH] = 0
        RACETRACK[18,26:WIDTH] = 0
        RACETRACK[19,27:WIDTH] = 0
        RACETRACK[20,30:WIDTH] = 0
        for i in range(3, 16):
            for j in range (0, i-2):
                RACETRACK[i,j] = 0
        RACETRACK[16:21,0:14] = 0
        RACETRACK[21,0:13] = 0
        RACETRACK[22,0:12] = 0
        RACETRACK[23:27,0:11] = 0
        RACETRACK[27,0:12] = 0
        RACETRACK[28,0:13] = 0
        RACETRACK[29,0:16] = 0
        # define starting line
        START = []
        for j in range(0,23):
            RACETRACK[0,j] = 2
            START.append([0,j])
        # define finish line
        FINISH = []
        for i in range(21,HEIGHT):
            RACETRACK[i,WIDTH-1] = 3
            FINISH.append([i,WIDTH-1])
    return RACETRACK, HEIGHT, WIDTH, START, FINISH


ACTIONS = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]
nr_actions = len(ACTIONS)
MAX_SPEED = 5


def compute_projected_path(position, speed):
    path = [position]
    old_position = position
    if speed[1] > speed[0]:       # vertical speed is higher
        for _ in range(speed[1]):
            new_position = [old_position[0]+ speed[0]/speed[1], old_position[1]+1] 
            if abs(new_position[0] - int(new_position[0])) < .01:
                path.append([int(new_position[0]),int(new_position[1])])
            else:
                path.append([int(new_position[0]),new_position[1]])
                path.append([int(new_position[0]+.9),new_position[1]])
            old_position = new_position         
    else:                         # horizontal speed is higher or equal to vertical speed
        for _ in range(speed[0]):
            new_position = [old_position[0]+ 1, old_position[1]+speed[1]/speed[0]] 
            if abs(new_position[1] - int(new_position[1])) < .01:
                path.append([int(new_position[0]),int(new_position[1])])
            else:
                path.append([int(new_position[0]),int(new_position[1])])
                path.append([int(new_position[0]),int(new_position[1]+.9)])
            old_position = new_position         
    return path

def cross_finishline(path, finish_line):
    path_till_finish = []
    for position in path:
        path_till_finish.append(position)
        if position in finish_line:
            return True, path_till_finish
    return False, path

def out_of_bound(path, racetrack, height, width):
    for position in path:
        if position[0]<0 or position[0]> (height - 1) or position[1]<0 or position[1] > (width-1):
            return True
        if racetrack[position[0]][position[1]] == 0:
            return True
    
def speed_ok(speed, action):
    if speed[0]+action[0] > MAX_SPEED: return False
    if speed[0]+action[0] < 0: return False
    if speed[1]+action[1] > MAX_SPEED: return False
    if speed[1]+action[1] < 0: return False
    if speed[0]+action[0] == 0 and speed[1]+action[1] == 0: return False
    return True


def exec_table_policy(position, speed, policy = None):
    if policy == []:    
        # default case, if no policy (table) is given: random policy                          
        return np.random.randint(0,nr_actions)
    else:
        if len(policy.shape) == 4:
            # deterministic policy
            return policy[position[0],position[1],speed[0],speed[1]]
        elif len(policy.shape) == 5:
            # stochastic policy
            return np.random.choice(nr_actions, p=policy[position[0], position[1], speed[0], speed[1],:])
        else:
            return "error"

def drive(position, speed, racetrack, height, width, start_line, finish_line, noise = False, policy = [], plot = False):
    episode = []
    finish = False
    while finish == False:
        old_position = position[:]
        old_speed = speed[:]
        valid = False
        counter = 0
        while not valid:
            action = exec_table_policy(position, speed, policy)
            valid = speed_ok(speed, ACTIONS[action])
            #print("valid in drive:", valid)
            if plot and counter > 100:         # policy is not valid, return empty episode for plot
                return []
            counter += 1
        if noise:
            if np.random.choice(2, p=[0.9, 0.1]) == 1:
                action = ACTIONS.index([0,0])
        action_speed = ACTIONS[action]
        #episode.append([position, speed, action, -1])
        new_speed = [speed[0]+action_speed[0], speed[1]+action_speed[1]]
        path = compute_projected_path(position,new_speed)
        crossed_finish, path = cross_finishline(path, finish_line)
        r = -1
        if crossed_finish and not out_of_bound(path, racetrack, height, width):
            finish = True
        else:
            if out_of_bound(path, racetrack, height, width):
                position = start_line[np.random.randint(0,len(start_line))]
                speed = [0,0]
                r = -len(start_line)          # to avoid going out-of-bound as strategy to improve starting position
            else:
                position = [position[0]+new_speed[0], position[1]+new_speed[1]]
                speed = new_speed
        episode.append([old_position, old_speed, action, r])
    return episode

def exploring_start(racetrack, height, width):
    # ignores that certain high speeds cannot be reached close to starting line
    while True:
        init_pos = [np.random.randint(0,height),np.random.randint(0,width)]
        init_speed = [np.random.randint(0,MAX_SPEED),np.random.randint(0,MAX_SPEED)]
        if racetrack[init_pos[0],init_pos[1]] == 1:              # normal position on race track
            return init_pos, init_speed
        elif racetrack[init_pos[0],init_pos[1]] == 2:            # start line
            init_speed = [0, 0]                      
            return init_pos, init_speed
    
def monte_carlo_off_policy(racetrack, height, width, start_line, finish_line, gamma,
                           episodes, expl_start, start_position = None, noise=False):
    # racetrack 32 x 17, speed 5x5, actions 9
    # -20 in Q_state_action_value since rewards are negative
    Qsa_value = np.random.rand(height,width,MAX_SPEED+1,MAX_SPEED+1,nr_actions)-20
    Csa_count = np.zeros((height,width,MAX_SPEED+1,MAX_SPEED+1,nr_actions))
    state_visit_count = np.zeros((height,width,MAX_SPEED+1,MAX_SPEED+1))
    pi_policy = np.argmax(Qsa_value, axis = 4)
    
    episode_count = 0
    while episode_count < episodes:
        # ES exploring starts
        if expl_start:
            position, speed = exploring_start(racetrack, height, width)
        else:
            position, speed = start_position,[0,0]
        episode = drive(position, speed, racetrack, height, width, start_line, finish_line, noise)
        G = 0
        W = 1
        for step in range(len(episode), 0, -1):
            episode_step = episode[step-1]
            position = episode_step[0]
            speed = episode_step[1]
            action = episode_step[2]
            reward = episode_step[3]
            G = gamma * G + reward
            #print(Csa_count[position[0], position[1], speed[0], speed[1], action])
            Csa_count[position[0], position[1], speed[0], speed[1], action] = Csa_count[position[0], position[1], speed[0], speed[1], action] + W
            Qsa_value[position[0], position[1], speed[0], speed[1], action] = round(Qsa_value[position[0], position[1], speed[0], speed[1], action] + W / Csa_count[position[0], position[1], speed[0], speed[1], action] * (G-Qsa_value[position[0], position[1], speed[0], speed[1], action])-.000005,5)
            state_visit_count[position[0], position[1], speed[0], speed[1]] += 1
            pi_policy[position[0], position[1], speed[0], speed[1]] = np.argmax(Qsa_value[position[0], position[1], speed[0], speed[1], :])
            #instead of checking whether acion = argmax(target policy), because of ties, the return of the actions is checkec instead (prevents ever changing results)
            if Qsa_value[position[0], position[1], speed[0], speed[1], action] != np.ndarray.max(Qsa_value[position[0], position[1], speed[0], speed[1],:]):
                break
            W = W / (1/nr_actions)
        episode_count += 1
        #print(episode_count)
        if episode_count % 1000 == 0: print(episode_count,"/",episodes," episodes")  
    return Qsa_value, pi_policy
        

def monte_carlo_on_policy(racetrack, height, width, start_line, finish_line, gamma,
                          epsilon, episodes, expl_start, start_position = None, noise=False):
    # racetrack 32 x 17, speed 5x5, actions 9
    # -20 in Q_state_action_value since rewards are negative
    Qsa_value = np.random.rand(height,width,MAX_SPEED+1,MAX_SPEED+1,nr_actions)-20
    Countssa = np.ones((height,width,MAX_SPEED+1,MAX_SPEED+1,nr_actions))
    pi_policy = np.random.rand(height,width,MAX_SPEED+1,MAX_SPEED+1,nr_actions)
    pi_policy = pi_policy / np.repeat(np.sum(pi_policy, axis = 4)[:, :, :, :, np.newaxis], nr_actions, axis = 4) # make probabilities add to 1
    
    episode_count = 0
    while episode_count < episodes:
        if expl_start:
            position, speed = exploring_start(racetrack, height, width)
        else:
            position, speed = start_position,[0,0]
        episode = drive(position, speed, racetrack, height, width, start_line, finish_line, noise, pi_policy)
        G = 0
        visited = []
        for step in range(len(episode), 0, -1):
            episode_step = episode[step-1]
            position = episode_step[0]
            speed = episode_step[1]
            action = episode_step[2]
            reward = episode_step[3]
            G = gamma * G + reward
            # first visit only
            if not [position, speed, action] in visited:
                #print(Countssa[position[0], position[1], speed[0], speed[1], action])
                Qsa_value[position[0], position[1], speed[0], speed[1], action] = round(Qsa_value[position[0], position[1], speed[0], speed[1], action] + 1 / Countssa[position[0], position[1], speed[0], speed[1], action] * (G-Qsa_value[position[0], position[1], speed[0], speed[1], action])-.000005,5)
                Countssa[position[0], position[1], speed[0], speed[1]] += 1
                A = np.argmax(Qsa_value[position[0], position[1], speed[0], speed[1], :])
                pi_policy[position[0], position[1], speed[0], speed[1], :] = epsilon / nr_actions
                pi_policy[position[0], position[1], speed[0], speed[1], A] = 1 - epsilon + epsilon/nr_actions
        visited.append([position, speed, action])
        episode_count += 1
        if episode_count % 1000 == 0: print(episode_count,"/",episodes," episodes")    
    # make policy deterministic
    pi_policy = np.argmax(Qsa_value, axis = 4)
    return Qsa_value, pi_policy
        

        
def start_simulation(racetrack_nr, gamma, on_policy, epsilon, expl_start, start_position, episodes, filename, noise):
    racetrack, height, width, start_line, finish_line = read_racetrack(racetrack_nr)
    if on_policy:
        Qsa_value, pi_policy = monte_carlo_on_policy(racetrack, height, width, start_line, finish_line, gamma,
                                                     epsilon, episodes, expl_start, start_position, noise)
    else:
        Qsa_value, pi_policy = monte_carlo_off_policy(racetrack, height, width, start_line, finish_line, gamma,
                                                      episodes, expl_start, start_position, noise)
    with open(PATH+filename+"_Pi_policy.pkl", 'wb') as f:
        pickle.dump(pi_policy, f, pickle.HIGHEST_PROTOCOL)
    with open(PATH+filename+"_Qsa_value.pkl", 'wb') as f:
        pickle.dump(Qsa_value, f, pickle.HIGHEST_PROTOCOL) 
    print_solution(filename, racetrack, racetrack_nr, episodes, on_policy, expl_start, start_position, height, width, start_line, finish_line)  
        

def print_solution(filename, racetrack, racetrack_nr, episodes, on_policy, expl_start, start_position, height, width, start_line, finish_line):
    from matplotlib import pyplot as plt
    from copy import deepcopy    
    with open(PATH+filename+"_Pi_policy.pkl", 'rb') as f:
        pi_target_policy = pickle.load(f)
    if racetrack_nr == 1:
        fig, axs = plt.subplots(1, len(start_line), figsize=(7, 7))
    else:
        fig, axs = plt.subplots(3, int((len(start_line)+1.5)/3), figsize=(5, 5))
        axs[2, int((len(start_line)+1.5)/3)-1].matshow([[0,0],[0,0]])                 # to make sure the last plot is filled in case of odd number of starting positions
    print_on_off = ", off_policy, "
    if on_policy: print_on_off = ", on_policy, "
    if expl_start:
        print_start = "with exploring starts"
    else:
        print_start = "starting from: "+ str(start_position)
    title = str(episodes)+" episodes"+print_on_off+print_start
    _ = fig.suptitle(title, fontsize="x-large")
    counter = 0
    for start_pos in start_line:
        racetrack_plot = deepcopy(racetrack)
        episode = drive(start_pos,[0,0], racetrack, height, width, start_line, finish_line, False, pi_target_policy, True)
        for step in episode:
            position = step[0]
            if not(position[0]<0 or position[0]>(height-1) or position[1]<0 or position[1]> (width-1)):   # check whether not out of bound
                racetrack_plot[position[0],position[1]] = 4
        if racetrack_nr == 1:
            if episode == []:
                axs[counter].matshow([[0,0],[0,0]])
            else:
                axs[counter].matshow(racetrack_plot)
            axs[counter].set_title("Start: "+str(start_pos)) 
        else:
            index1 = int((counter) / int((len(start_line)+1.5)/3))
            index2 = counter - index1 * int((len(start_line)+1.5)/3)
            if episode == []:
                axs[index1, index2].matshow([[0,0],[0,0]])
            else:
                axs[index1, index2].matshow(racetrack_plot)
            axs[index1, index2].set_title("Start: "+ str(start_pos)) 
            plt.setp(axs, xticks=[], yticks=[])
        counter += 1
    outGraph = PATH+filename+".png"
    figure = plt.gcf() 
    figure.set_size_inches(20, 15)
    plt.savefig(outGraph, dpi = 200)
    plt.show()


def fig_1():
    FILE_NAME = "Ex_5_12_racetrack_fig_1"
    RACETRACK_NR = 1                 # 1 or 2
    GAMMA = 0.9
    ON_POLICY = False
    EPSILON = None                  # only relevant for ON_POLICY = True
    EXPL_START = True
    START_POSITION = None         # None if EXPL_START = True
    EPISODES = 20000
    NOISE = True                  # if True, probability 0.1 of no velocity increment
    start_simulation(RACETRACK_NR, GAMMA, ON_POLICY, EPSILON, EXPL_START, START_POSITION, EPISODES, FILE_NAME, NOISE)


def fig_2():
    FILE_NAME = "Ex_5_12_racetrack_fig_2"
    RACETRACK_NR = 1                 # 1 or 2
    GAMMA = 0.9
    ON_POLICY = False
    EPSILON = .03                  # only relevant for ON_POLICY = True
    EXPL_START = False
    START_POSITION = [0,3]         # None if EXPL_START = True
    EPISODES = 20000
    NOISE = False                  # if True, probability 0.1 of no velocity increment
    start_simulation(RACETRACK_NR, GAMMA, ON_POLICY, EPSILON, EXPL_START, START_POSITION, EPISODES, FILE_NAME, NOISE)


def fig_3():
    FILE_NAME = "Ex_5_12_racetrack_fig_3"
    RACETRACK_NR = 2                 # 1 or 2
    GAMMA = 0.9
    ON_POLICY = True
    EPSILON = .03                  # only relevant for ON_POLICY = True
    EXPL_START = True
    START_POSITION = None         # None if EXPL_START = True
    EPISODES = 20000
    NOISE = False                  # if True, probability 0.1 of no velocity increment
    start_simulation(RACETRACK_NR, GAMMA, ON_POLICY, EPSILON, EXPL_START, START_POSITION, EPISODES, FILE_NAME, NOISE)


def fig_4():    
    FILE_NAME = "Ex_5_12_racetrack_fig_4"
    RACETRACK_NR = 2                 # 1 or 2
    GAMMA = 0.9
    ON_POLICY = True
    EPSILON = .03                   # only relevant for ON_POLICY = True
    EXPL_START = False
    START_POSITION = [0,16]         # None if EXPL_START = True
    EPISODES = 20000
    NOISE = True                  # if True, probability 0.1 of no velocity increment
    start_simulation(RACETRACK_NR, GAMMA, ON_POLICY, EPSILON, EXPL_START, START_POSITION, EPISODES, FILE_NAME, NOISE)


if __name__ == '__main__':
    import numpy as np
    import pickle
    PATH = "C:/Users/Hans/workspace/Reinforcement Learning/Files/images/Chapter 5/"    
    
    fig_1()
    fig_2()
    fig_3()
    fig_4()



    
    