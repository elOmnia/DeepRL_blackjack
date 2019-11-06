import gym
import matplotlib.pyplot as plt
import timeit
num_rounds = 10  # Payout calculated over num_rounds
num_samples = 10  # num_rounds simulated over num_samples

average_payouts = []

start = timeit.timeit()
env = gym.make('Blackjack-v0')
env.seed(0)
env.reset()

for sample in range(num_samples):
    round = 1
    total_payout = 0  # to store total payout over 'num_rounds'

while round <= num_rounds:
    action = env.action_space.sample()  # take random action
    #         print('ACTION: ' + str(action))

    obs, payout, is_done, info = env.step(action)
    # print('OBS: ' + str(obs))
    # print('PAYOUT: ' + str(payout))
    # print('ID_DONE: ' + str(is_done))
    # print('INFO: ' + str(info))

    total_payout += payout
    if is_done:
        env.reset()  # Environment deals new cards to player and dealer
        round += 1
average_payouts.append(total_payout)


plt.plot(average_payouts)
plt.xlabel('num_samples')
plt.ylabel('payout after 1000 rounds')
plt.show()
print("Average payout after {} rounds is {}".format(num_rounds, sum(average_payouts) / num_samples))
end = timeit.timeit()
print(end - start)

