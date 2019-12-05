import gym
import matplotlib.pyplot as plt
import timeit
import pandas as pd

start = timeit.default_timer()
env = gym.make('Blackjack-v0')
env.seed(0)
env.reset()

num_rounds = 1000  # Payout calculated over num_rounds
# corresponds to number of episodes in the q learning algorithm for example (how many times we assign action and take
# a step)
num_samples = 1000  # num_rounds simulated over num_samples
# how many times we run the test (can be varied, optimally big enough for good result but not bigger than needed)

average_payouts = []
payout_list = []

for sample in range(num_samples):
    round = 1
    total_payout = 0  # to store total payout over 'num_rounds'

    while round <= num_rounds:
        action = env.action_space.sample()  # take random action
        #         print('ACTION: ' + str(action))

        obs, payout, is_done, info = env.step(action)
        #         print('OBS: ' + str(obs))
        #         print('PAYOUT: ' + str(payout))
        #         print('ID_DONE: ' + str(is_done))
        #         print('INFO: ' + str(info))

        total_payout += payout
        if is_done:
            payout_list.append(payout)
            env.reset()  # Environment deals new cards to player and dealer
            round += 1
    average_payouts.append(total_payout)
df = pd.DataFrame(average_payouts, columns= ['average_payouts'])
#print (df)
export_csv = df.to_csv (r'C:/Users/sebas/Desktop/DRL/randomstart_anja.csv', index = None, header=True)

payout_list_last = payout_list[-100000:]
winning = payout_list_last.count(1)/len(payout_list_last)
drawing = payout_list_last.count(0)/len(payout_list_last)
loosing = payout_list_last.count(-1)/len(payout_list_last)
natural = payout_list_last.count(1.5)/len(payout_list_last)
print("length",len(payout_list))
print("winnin",winning,"drawing",drawing,"loosing",loosing, "natural",natural)

"""print("Average payout after {} rounds is {}".format(num_rounds, sum(average_payouts) / num_samples))
end = timeit.default_timer()
print('Runtime ' + str(end - start))
plt.plot(average_payouts)
plt.xlabel('num_samples')
plt.ylabel('payout after 1000 rounds')
plt.show(block=False)
plt.close()
"""
