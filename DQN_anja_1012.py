import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
import pandas as pd
import gym
import matplotlib.pyplot as plt
from collections import deque
import time

env = gym.make('Blackjack-v0')
env.seed(0)
start_time = time.time()


def reshape_state(state):
    state = np.reshape(state[0:2], [1, 2])
    return state


class DQNAgent():
    def __init__(self, env, epsilon=1.0, alpha=0.5, gamma=0.9, time=30000):
        self.env = env
        self.action_size = self.env.action_space.n
        self.state_size = env.observation_space
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
    def _build_model(self):
        #         print(type(self.state_size))
        model = Sequential()
        model.add(Dense(32, input_shape=(2,), kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.alpha))

        return model

    # build a bigger neural network (use either this one or the above one depending on how many rounds/samples taken)
    def _build_big_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=(2,), kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(16, activation='relu'))
        # TODO: here add another layer --> TODO try to paint this NN to better see input output sizes
        # TODO: this line below does not work add it correctly
        # model.add(Dense(8, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.alpha))

    def choose_action(self, state):
        """
        Choose which action to take, based on the observation.
        Uses greedy epsilon for exploration/exploitation.
        """
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
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
        if self.time_left > 0.9 * self.time:
            self.epsilon -= self.small_decrement
        elif self.time_left > 0.7 * self.time:
            self.epsilon -= self.small_decrement
        elif self.time_left > 0.5 * self.time:
            self.epsilon -= self.small_decrement
        elif self.time_left > 0.3 * self.time:
            self.epsilon -= self.small_decrement
        elif self.time_left > 0.1 * self.time:
            self.epsilon -= self.small_decrement
        self.time_left -= 1

    def learn(self, state, action, reward, next_state, done):
        target = reward

        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def get_optimal_strategy(self, alpha, gamma):
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
        df.to_csv('optimal_policy' + str(num_rounds) + 'nr' + str(num_samples) + 'ns' + str(alpha) + 'alpha' + str(
            gamma) + 'gamma' + '.csv')
        return df

    def play_optimal_strategy(self, optimal_strategy, reward):
        """ for every state we are in choose the corresponding action to take according to the optimal strategy
        that was computed and print out reward
        Reshape the state and for that state access the index Optimal column and read the value out, if hit then
        action 1 else zero """
        done = False
        state = env.reset()  # starts in some state like (12,10,False)
        if state[0] >= 21:
            done = True
        state = reshape_state(state)
        state = state[0][0], state[0][1]

        while not done:
            if optimal_strategy['Optimal'][state] == 'Hit' or optimal_strategy['Optimal'][state] == 'Stand':
                # check if there is hit or stand on pos index and change to while not done
                action = 1 if optimal_strategy['Optimal'][state] == 'Hit' else 0
                state, reward, done, _ = env.step(action);
                if state[0] >= 21:
                    break
                state = reshape_state(state)
                state = state[0][0], state[0][1]
            else:
                action = random.randint(0, 1)
                state, reward, done, _ = env.step(action);
                state = reshape_state(state)
                state = state[0][0], state[0][1]

        if done:
            state = env.reset()
        return

    def print_and_plot_outcome(self,payout_list):
        print(payout_list[-100:])
        payout_list_last = payout_list[-100:]
        winning = payout_list_last.count(1) / len(payout_list_last)
        drawing = payout_list_last.count(0) / len(payout_list_last)
        loosing = payout_list_last.count(-1) / len(payout_list_last)

        # save winning, drawing and loosing probability in a file with according name and the parameters
        index = (alpha, gamma)
        win_loose_draw_file = pd.DataFrame(index=index, columns=['Win', 'Draw', 'Loose'])
        for ind in index:
            win_loose_draw_file.loc[ind, 'Win'] = winning
            win_loose_draw_file.loc[ind, 'Draw'] = drawing
            win_loose_draw_file.loc[ind, 'Loose'] = loosing

        win_loose_draw_file.to_csv(
            'winning_loosing_drawing_probabilities' + str(num_rounds) + 'nr' + str(num_samples) + 'ns' + str(
                alpha) + 'alpha' + str(gamma) + 'gamma' + '.csv')

        print('num_rounds', num_rounds, 'num_samples', num_samples, 'alpha', alpha, 'gamma', gamma)
        print("winning", winning, "drawing", drawing, "loosing", loosing)

        plt.plot(average_payouts)
        plt.xlabel('num_samples' + str(num_samples))
        plt.ylabel('payout after' +str(num_rounds) + 'num_rounds')
        plt.savefig('DQN_blackjack' + str(num_rounds) + 'nr' + str(num_samples) + 'ns' + str(alpha) + 'alpha' + str(
            gamma) + 'gamma' + '.png')
        # plt.show()
        # plt.close()
        return

    def run(self, num_rounds, num_samples, alpha, gamma):
        average_payouts = []
        payout_list = []
        summed_rewards = 0
        state = env.reset()
        state = reshape_state(state)
        for sample in range(num_samples):
            round = 1
            total_payout = 0  # store total payout per sample
            while round <= num_rounds:
                action = agent.choose_action(state)
                next_state, payout, done, _ = env.step(action)
                summed_rewards += payout
                next_state = np.reshape(next_state[0:2], [1, 2])
                total_payout += payout
                #  learning phase
                agent.learn(state, action, payout, next_state, done)
                state = next_state
                state = reshape_state(state)

                if done:
                    payout_list.append(payout)
                    state = env.reset()  # Environment deals new cards to player and dealer
                    state = reshape_state(state)
                    round += 1

            average_payouts.append(total_payout)
            # get the optimal strategy and play blackjack using this strategy to see the performance
            optimal_strategy = agent.get_optimal_strategy(alpha, gamma);
            optimal_strategy.to_csv('optimal_strategy_test.csv')
            reward = 0  # init reward in order that reward has a value before we assign it in the play_optimal_strat
            # comment out below line if you want to play the computed optimal strategy on our agent
            # play = agent.play_optimal_strategy(optimal_strategy, reward)
        return average_payouts, payout_list


if __name__ == "__main__":
    num_rounds_array = [100]  # Payout calculated over num_rounds
    num_samples_array = [50]  # num_rounds simulated over num_samples

    #num_rounds_array = [100, 200, 500, 1000]
    #num_samples_array = [50, 100, 200, 500]
    alphas = [0.1]  # [0.001, 0.1, 0.5, 0.9, 0.99]
    gammas = [0.9]  # [0.001, 0.1, 0.5, 0.9, 0.99]

    for num_rounds in num_rounds_array:
        print('num_rounds', num_rounds)
        for num_samples in num_samples_array:
            print('num_samples', num_samples_array)
            for alpha in alphas:
                print('alpha', alpha)
                for gamma in gammas:
                    print('gamma', gamma)
                    agent = DQNAgent(env=env, epsilon=1.0, alpha=alpha, gamma=gamma, time=7500)
                    average_payouts, payout_list = agent.run(num_rounds, num_samples, alpha, gamma)
                    agent.print_and_plot_outcome(payout_list)

# save average_payouts in a csv
average_payouts_df = pd.DataFrame(average_payouts, columns=['Payouts'])
average_payouts_df.to_csv('average_payouts'+str(num_rounds) + 'nr' + str(num_samples) + 'ns' + str(alpha) + 'alpha' +
                          str(gamma) + 'gamma' + '.csv')
# this plots and saves outcomes in csv files and prints to console
agent.print_and_plot_outcome(payout_list)
