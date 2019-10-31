import random
from collections import defaultdict

import gym
import numpy as np
import time
from pprint import pprint
import sys
import copy
import os
SEP = ';'
PATH = '/home/anjak/Dokumente'


# Params
MIN_EPSILON = 0.1
N_STATES = 1000  # number of states determine with env.nS
ALPHA = 0.7  # learning_rate
GAMMA = 0.6  # discount_factor
EPSILON = 0.1  # default epsilon value (with this probability we take a random action)


def adaptive_epsilon_get(episode, min_epsilon=MIN_EPSILON, number_of_states=N_STATES):
    # return max(min_epsilon, min(1, 1.0 - np.math.log10((episode + 1) / number_of_states)))
    if (episode < 1000):
        return 1.0
    if (episode > 5000):
        return 0.0
    return EPSILON


def qlearning_alg(env, total_episodes=10000, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, render=False):
    '''

    :param env:environment we act in
    :param total_episodes: total number of episodes we go through
    :param alpha:learning rate from qlearning alg
    :param gamma: discountfactor from qlearningalg
    :return:qtable, convergence_rounds
    '''
    # initialize qtable
    convergence_rounds = 0
    #Q_table = np.zeros([env.observation_space.n, env.action_space.n])
    #Q_table= np.full((env.observation_space.n, env.action_space.n), 1.0)
   # Q_table= np.full((env.observation_space.n, env.action_space.n),-1.1)
    Q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    #Q_table = np.zeros(self.env.action_space.n)
    # print(Q_table)
    total_reward = 0
    nA = env.action_space.n
    nS = env.observation_space #returns Tuple(Discrete(32), Discrete(11), Discrete(2))
    print('nS',nS)
    policy = np.ones(nS)
    #  policy = np.ones(nS, dtype=int) #TODO: fails here with dytpe int
    old_policies = []
    num_old_policies = 100
    for episode in range(total_episodes):
        # sys.stdout.write("\rEpisode %d -> delta is %.2f and total reward is %.2f" % (episode, delta, total_reward))
        # sys.stdout.write('\r{}'.format( Q_table))
        # sys.stdout.write('\r{}'.format(policy))
        sys.stdout.flush()
        state = env.reset()
        done = False
        epsilon = adaptive_epsilon_get(episode)
        # TODO: comment above row epsilon=adaptive... out when you dont want an adaptive but a fixed epsilon
        delta = 0
        while not done:
            if (random.uniform(0, 1) > epsilon):
                action = np.argmax(Q_table[state])
            else:
                action = env.action_space.sample()
            # env.render() # to see which steps it takes
            # with probability epsilon take a random value (explore)
            new_state, reward, done, info = env.step(action)
            # this is to overcome the bad initialization for known states
            if done:
                Q_table[new_state] = reward
            if render:
                env.render() #this renders the env if the variable render is set to true

            Q_update = alpha * (reward + gamma * np.max(Q_table[new_state]) - Q_table[state, action])
            Q_table[state, action] = Q_table[state, action] + Q_update

            delta = max(delta, abs(Q_update))
            # update qtable according to book page 107
            # take a step (action) according to policy derived from Q
            # print('reward in round {} is {}'.format(episode, reward))
            # print('Value of boolean done is {}'.format(done))
            # print ('QTable for state {} is {}'.format(state, Q_table[state]))
            state = new_state
            convergence_rounds += 1
            total_reward += reward
            policy[state] = np.argmax(Q_table[state])
            #print('policy is {}'.format(policy))

            if done:
               # old_policies.append(policy[:])
                #copypolicy= copy.deepcopy(policy)
                #old_policies.append(copypolicy)
                old_policies.append(copy.deepcopy(policy))
                break

        # remove very oldest policy
        if len(old_policies) > num_old_policies:
            old_policies.remove(old_policies[0])

        # policy[state] = np.argmax(Q_table[state])
        # print('policy in episode {} is {}'.format(episode, policy))
        allEqual = len(old_policies) == num_old_policies
        for i in range(len(old_policies)):
            allEqual = allEqual and np.all(old_policies[i] == policy)

        if allEqual:
            break
       # print('policy is {}'.format(policy))
        # when policy does not change anymore over 10 episodes then stop (say it converges)

    # policy= np.argmax(Q_table[state])
    #print('policy for gamma {} is {}'.format(gamma, policy))
    print('\n')
    print('Qlearning converges after {} episodes and {} rounds'.format(episode, convergence_rounds))
    # TODO include stopping criteria when policy is goodenough
    print('Q-table is {}'.format(Q_table))
    policy[state] = np.argmax(Q_table[state])  # keep track of policy over different episodes and compare it
    # if episode % 100 == 0:
    #   print('Episode {} Total Reward {} convergence rounds: {}'.format(episode, total_reward, convergence_rounds))
    tempdebug= Q_table

    return convergence_rounds, total_reward, episode, policy,Q_table

def write_csv(matrix, file_name):
    #file_name= 'qlearningmatrixgridproblems.csv'
    outpath = os.path.join(PATH, file_name)
    np.savetxt(outpath, matrix,delimiter=SEP)


def read_csv(inpath):
    #inpath = '/home/anjak/Dokumente/qlearningmatrix.csv'
    matrix = np.loadtxt(open(inpath, "rb"), delimiter=SEP, skiprows=0)
    return matrix


def test_funtion(env_name, gamma, epsilon=EPSILON):
    '''
    :param env: environment we test (different envs like taxi-v2, FrozenLake-v0, ..
    :param gamma: discountfactor
    :param epsilon : probability at which we choose an action at random
    :return: convergence_rounds, mean_score, total_time
    I iterate 10 times (and take the average) over my whole test_function with different gammas each time as I want to see the behavior
    on average and make sure that the values were not just a single coincidence.
    '''
    print('Running %s with gamma = %s' % (env_name, gamma))
    print('Initial state: ')
    env = gym.make(env_name)
    env.reset()
    start_time = time.time()
    convergence_rounds, total_reward, episode, policy, qtable = qlearning_alg(env, gamma=gamma)
    write_csv(qtable,file_name= 'qlearningmatrixBlackjack.csv')
    write_csv(policy, file_name='policyqlearningBlackjack.csv')
    #to debug only
    temptodebug= policy
    #write_policy_to_file(policy) #this writes the policy in a separate odt file
    total_time = time.time() - start_time
    print('Convergence after {} rounds'.format(convergence_rounds))
    print('Runtime of the alg {}'.format(total_time))
    print('Total rewards {}'.format(total_reward))
    print('Final state:')
    env.render()  # this prints out the final state
    # TODO print convergence_rounds meanscore as well
    return total_time, convergence_rounds, total_reward, episode, policy


if __name__ == '__main__':
    '''
    mainfunction, calls the test function with different parameters 

    '''
    # iterate 10 times over the test and take the mean of it
    i = 0  # iterator
    convergence_rounds_over_10_it = 0
    convergence = []
    episodes= []
    for i in range(9):
        env_name = 'Blackjack-v0'
        # gammas= [0.1]
        gammas= [ 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.99]
        #gammas = [0.6]
        statistics_dict = {}
        for gamma in gammas:
            total_time, convergence_rounds, total_reward, episode, policy = test_funtion(env_name=env_name, gamma=gamma)
            #  total_time, convergence_rounds, total_reward = test_function(env_name=env_name, gamma=gamma)
            # time_over_10_it += total_time
            # convergence_rounds_over_10_it += convergence_rounds
            # reward_over_10_it += total_reward
            convergence.append(convergence_rounds)
            convergence_rounds_over_10_it = np.mean(convergence)
            episodes.append(episode)
            episodes_over_10_it= np.mean(episode)

            i = i + 1
            # print(time_over_10_it, reward_over_10_it, convergence_rounds_over_10_it)
            statistics_dict[gamma] = ['converges after {} episodes: '.format(episodes_over_10_it),
                                      'Convergence rounds : {}'.format(convergence_rounds_over_10_it),
                                      'Total Reward : {}'.format(total_reward),
                                      'Total time:{} '.format(total_time),
                                      'Policy: {}'.format(policy)]

    print('Gamma results: ')
    pprint(statistics_dict)