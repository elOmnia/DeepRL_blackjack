# DeepRL_blackjack
Authors: Anja Koller, Omnia el Sadaany, Sebastian Gaeumann University Zurich

basic blackjack implementations with deep RL
In this file we explain what you can find in which file and the sources we used for the code. 
We modified the code from the website for better readability and analysis. 
We solved Blackjack using Qlearning, SARSA and Qlearning with Neural networks (DeepQlearning). 
These are all Reinforcement learning techniques. 
The main code sources where:

1) Q-learning: QLearning w Writer_Sebi


CodeSource
--> https://github.com/Pradhyo/blackjack/blob/master/blackjack.ipynb

2) SARSA: 

CodeSource
--> https://github.com/Pradhyo/blackjack/blob/master/blackjack.ipynb (adaptions from that to SARSA)

3) Deep Q-learning: 
 	DQN_blackjack_website_1511.py
  DQN_anja_1012 (same as above but with saving the dataframe of averagepayouts for the plots)
Codesource
-->  https://github.com/ml874/Blackjack--Reinforcement-Learning/blob/master/Blackjack-%20DQN%20(Only%20Hit%20or%20Stand).ipynb


Additionally for performance comparison of the algorithms we implemented the normal and the random strategy in the following files, and one for plotting to have nice comparisons

4) Normal Strategy:  	Basic Strategy_Sebi.py

5) Random Strategy: Blackjack_randomStrategy_Anja.py

6) Comparison plots (all Algorithms): Qlearning_plots_0612.py


