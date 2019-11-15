# Codesource: https://github.com/ml874/Blackjack--Reinforcement-Learning/blob/master/Blackjack-%20DQN%20(Only%20Hit%20or%20Stand).ipynb
import random
import gym
import numpy as np
import pandas as pd
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from collections import deque
import time

start_time = time.time()

num_rounds = 100  # Payout calculated over num_rounds
# num_rounds: define how long a game goes, num_rounds simulated over num_samples
num_samples = 50  # num_rounds simulated over num_samples

random.seed(0)  # TODO: necessary? if no -> remove


class DQNAgent:
    def __init__(self, env, epsilon=1.0, alpha=0.5, gamma=0.9, time=30000):
        self.env = env
        self.action_size = self.env.action_space.n  # size of action space (here 2)
        self.state_size = env.observation_space  # size of state space (here Discrete(2,...,)
        self.memory = deque(maxlen=2000)  # Record past experiences- [(state, action, reward, next_state, done)...]
        self.epsilon = epsilon  # Random exploration factor
        self.alpha = alpha  # Learning factor
        self.gamma = gamma  # Discount factor- closer to 1 learns well into distant future
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning = True
        self.model = self._build_model()

        self.time = time
        self.time_left = time  # Epsilon Decay
        self.small_decrement = (0.4 * epsilon) / (0.3 * self.time_left)  # reduce epsilon

    # Build Neural Net
    ''' neural net is sequential with 1 hidden layers --> input layer 32 input nodes (as we have 32 possible values), 
    kernel initializer random uniformand activation function reLu,
    hidden layer has 16 inputs and activation function reLu, and output layer has
    number of actions as input and activation function Softmax.
    output computed with loss function binary cross-entropy and Adam optimizer
    Important: Number of output units and input units of 2 consecutive layers must match
    number of units in output layer should be equal to the number of unique class labels
    This data-set has 32 input values and self.action_size output values, so the input and output
    layer are of dimension 32 resp self.action_size
    Dense: fully-connected layer, arguments are output dimensions (input shape which is 2 for first case)
    no need to specify input dim as for sequential is same as output of last layer'''

    def _build_model(self):
        # print(type(self.state_size))
        model = Sequential()  # initialize FeedForward Neural Network (output of each layer is input to next layer)
        model.add(Dense(32, input_shape=(2,), kernel_initializer='random_uniform', activation='relu'))  # input layer
        # for input layer we have to make sure that input_dim attribute matches number of features in training set
        model.add(Dense(16, activation='relu'))  # hidden layer
        model.add(Dense(self.action_size, activation='softmax'))  # output layer
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.alpha))  # compile with Adam optimizer and
        # binary cross-entropy for loss function (cost function in logistic regression)
        return model

    #     Remember function that stores states, actions, rewards, and done to memory
    #     def remember(self, state, action, reward, next_state, done):
    #         self.memory.append([state, action, reward, next_state, done])

    def choose_action(self, state):
        """
        Choose which action to take, based on the observation.
        Uses greedy epsilon for exploration/exploitation.
        if random number > epsilon, act 'rationally', otherwise choose random action
        """

        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            # print('random: ' + str(action))

        else:
            action_value = self.model.predict(state)
            action = np.argmax(action_value[0])

        self.update_parameters()
        return action

    def update_parameters(self):
        """
        Update epsilon and alpha after each action
        Set them to 0 if not learning
        """
        #         print(self.time_left)
        if self.time_left > 0.9 * self.time:
            self.epsilon -= self.small_decrement
        elif self.time_left > 0.7 * self.time:
            self.epsilon -= self.small_decrement
        elif self.time_left > 0.5 * self.time:
            self.epsilon -= self.small_decrement
        #             print('0.5')
        elif self.time_left > 0.3 * self.time:
            #             print('0.2')
            self.epsilon -= self.small_decrement
        elif self.time_left > 0.1 * self.time:
            self.epsilon -= self.small_decrement
        #         elif self.time_left < 0.05 * self.time:
        #             self.epsilon = 0.000
        #             self.learning = False

        #         print(self.time_left)
        #         print(self.time)
        self.time_left -= 1

    def learn(self, state, action, reward, next_state, done):

        # minibatch = random.sample(self.memory, batch_size)
        # print(minibatch)

        target = reward

        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

        target_f = self.model.predict(state)

        target_f[0][action] = target

        self.model.fit(state, target_f, epochs=1, verbose=0)
        # train model by callit fit method over 1 epoch

    def get_optimal_strategy(self):
        index = []
        for x in range(0, 21):
            for y in range(1, 11):
                index.append((x, y))

        df = pd.DataFrame(index=index, columns=['Stand', 'Hit'])

        for ind in index:
            outcome = self.model.predict([np.array([ind])], batch_size=1)
            df.loc[ind, 'Stand'] = outcome[0][0]
            df.loc[ind, 'Hit'] = outcome[0][1]

        df['Optimal'] = df.apply(lambda x: 'Hit' if x['Hit'] >= x['Stand'] else 'Stand', axis=1)
        df.to_csv('optimal_policy.csv')
        return df


if __name__ == "__main__":
    env = gym.make('Blackjack-v0')
    agent = DQNAgent(env=env, epsilon=1.0, alpha=0.001, gamma=0.1, time=7500)  # build a deep q neural net
    average_payouts = []
    state = env.reset()
    state = np.reshape(state[0:2], [1, 2])
    for sample in range(num_samples):
        round = 1
        total_payout = 0  # store total payout per sample
        while round <= num_rounds:
            action = agent.choose_action(state)
            next_state, payout, done, _ = env.step(action)
            next_state = np.reshape(next_state[0:2], [1, 2])

            total_payout += payout
            agent.learn(state, action, payout, next_state, done)

            state = next_state
            state = np.reshape(state[0:2], [1, 2])

            if done:
                state = env.reset()  # Environment deals new cards to player and dealer
                state = np.reshape(state[0:2], [1, 2])
                round += 1

    average_payouts.append(total_payout)

    if sample % 10 == 0:
        print('Done with sample: ' + str(sample) + str("   --- %s seconds ---" % (time.time() - start_time)))
        print(agent.epsilon)

print(agent.get_optimal_strategy())

# Plot payout per 1000 episodes for each value of 'sample'

plt.plot(average_payouts)
plt.xlabel('num_samples')
plt.ylabel('payout after 1000 rounds')
plt.show(block=False)
plt.pause(3)
plt.close()

print("Average payout after {} rounds is {}".format(num_rounds, sum(average_payouts) / (num_samples)))

# Plot payout per 1000 episodes for each value of 'sample'

plt.plot(average_payouts)
plt.xlabel('num_samples')
plt.ylabel('payout after 1000 rounds')
plt.show(block=False)
plt.pause(3)
plt.close()

print("Average payout after {} rounds is {}".format(num_rounds, sum(average_payouts) / num_samples))
