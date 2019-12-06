import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



#TODO: 1 from googledocs here
average_payouts = pd.read_csv('Qlearning_sebastian_a0.5_g0.8.csv')

line1 = plt.plot(average_payouts, label='Qlearning')
plt.legend()
plt.xlabel('number of samples')
plt.ylabel('average payout')
plt.title('Qlearning average payouts over number of samples')
# comment below in if you want to save the plot
# plt.savefig('Qlearning_average_payouts_over_numsamples.png')
plt.show()
plt.close()


# compare against random and basic, import random and basic csv here
# TODO: here change tha pahnames to the according ones from the random and basic csvs
# random_average_payouts = pd.read_csv('pathtorandomcsv_file')
# basic_average_payouts = pd.read_csv('pathtobasiccsv_file')
# line2 = plt.plot(random_average_payouts, label = 'Normal')
# line3 = plt.plot(basic_average_payouts, label = 'Basic')

# TODO: 2 from sheet plots number 2
# barcharts for win, loose, draw percentage for all three algorithms qlearning, sarsa, and DQL



# TODO 3: see chat --> table with 5 different alphas and gamma --> write winning percentage for each of them


# TODO 4:

N = 3
#TODO fill in the probabilities for the different algorithms
winning = np.array((20, 35, 30))
drawing = np.array((25, 32, 34))
loosing  = np.array((25, 32, 34))

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, winning, width)
p2 = plt.bar(ind, drawing, width,
             bottom=winning)
p3 = plt.bar(ind, loosing, width,
             bottom=winning+drawing)

plt.ylabel('Percentage')
plt.title('Probabilty of winning Qlearning, SARSA and DQN')
plt.xticks(ind, ('QLearning', 'SARSA', 'DQN'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0], p3[0]), ('Winning', 'Drawing', 'Loosing'))
# TODO: comment below out to save the figure
#plt.savefig('probability_of_winning_qlearning_sarsa_dqn.png')

plt.show()



