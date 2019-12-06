import pandas as pd
import matplotlib.pyplot as plt



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
#line2 = plt.plot(random_average_payouts, label = 'Normal')
#line3 = plt.plot(basic_average_payouts, label = 'Basic')




