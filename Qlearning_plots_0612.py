import pandas as pd
import matplotlib.pyplot as plt



#TODO: 1 from googledocs here
average_payouts = pd.read_csv('Qlearning_sebastian_a0.5_g0.8.csv')

plt.plot(average_payouts)
plt.xlabel('number of samples')
plt.ylabel('average payout')
plt.title('Qlearning average payouts over number of samples')
# comment below in if you want to save the plot
# plt.savefig('Qlearning_average_payouts_over_numsamples.png')
plt.show()
plt.close()




