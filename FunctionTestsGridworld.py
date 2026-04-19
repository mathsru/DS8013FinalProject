#Alex Eliseev
#501093338
#Main script of the project, runs a series of RL algorithms on the 3 grid worlds and compares them in terms of their performance
#The objective is to show how each generation learns far more effectively for the same amount of training time measured in number of actions 
#taken in the Gridworld.
#Each RL algorithm that is tested is progressively more advanced.
#The first algiorthm is the first DQN for RL made by DeepSeek team in 2015
#Then the second is their Rainbow DQN from 2017 and lastly the Beyond The Rainbow DQN from the research paper with even more
#The last 2 algorithms that are going to be tested and compared against them are the next 2 proceeding generations of the DQN.
#The rainbow DQN which adds a total of 6 new features that substaintially imrpove the performance of the regular DQN and finally
#The main subject the research paper itself the BTR DQN which further builds up off of the Rainbow DQN adding a further 4 new features and upgrading 2 others.
#The goal of the script is to demonstrate the evolution of the DQN through the years ending with BTR DQN released in 2025 and that is capable of learning at almost 5x the rate of the regular rainbow DQN.

#imports
import numpy as np
#Importing the different functions from the other script files
from AlgorithmsOnGridworld.ValueIterationGridworld import ValueIterationSmallGridWorld, ValueIterationMediumGridWorld, ValueIterationLargeGridWorld
from AlgorithmsOnGridworld.DQNGridworld import DQNSmallGridWorld, DQNMediumGridWorld, DQNLargeGridWorld
from AlgorithmsOnGridworld.RainbowDQNGridworld import RainbowDQNSmallGridWorld, RainbowDQNMediumGridWorld, RainbowDQNLargeGridWorld
from AlgorithmsOnGridworld.BeyondTheRainbowDQNGridworld import BeyondTheRainbowDQNSmallGridWorld, BeyondTheRainbowDQNMediumGridWorld, BeyondTheRainbowDQNLargeGridWorld
#MatPlotLib for the plots and showing Number of actions worth of training vs reward
import matplotlib.pyplot as plt


"""Running DQN,RainbowDQN on each of the GridWorlds. Once again meassuring number of actions and the average reward after training for that many actions
First on small gridworld. Having each of them run, for 250 iterations.
print("Training DQN on small gridworld...")
NumActionsOverTimeDQNSmall, RewardAtNumberOfActionsSmall = DQNSmallGridWorld()
print("Training Rainbow DQN on small gridworld...")
NumActionsOverTimeRainbowDQNSmall, RewardAtNumberOfActionsSmallRainbowDQN = RainbowDQNSmallGridWorld()
print("Training Beyond The Rainbow DQN on small gridworld...")
NumActionsOverTimeBeyondTheRainbowDQNSmall, RewardAtNumberOfActionsSmallBeyondTheRainbowDQN = BeyondTheRainbowDQNSmallGridWorld()

Plotting actions vs reward after number of training actions for each model with the small Gridworld.
plt.plot(NumActionsOverTimeDQNSmall, RewardAtNumberOfActionsSmall,label="DQN")
plt.plot(NumActionsOverTimeRainbowDQNSmall, RewardAtNumberOfActionsSmallRainbowDQN,label="Rainbow DQN")
plt.plot(NumActionsOverTimeBeyondTheRainbowDQNSmall,RewardAtNumberOfActionsSmallBeyondTheRainbowDQN,label="Beyond The Rainbow DQN")
plt.xlabel("Total Actions/Steps Taken In The Gridworld")
plt.ylabel("Reward")
plt.title("Rl Algorithms Learning Effectiveness Over Time, Number Of Training Actions Vs Reward For Small Gridworld")
plt.legend()
plt.grid()
plt.show()

#Running DQN, RainbowDQN and Beyond the Rainbow DQN on the medium gridowlrd, each algorithm gets 500 episodes of training.
print("Training DQN on medium gridworld...")
NumActionsOverTimeDQNMedium, RewardAtNumberOfActionsMedium = DQNMediumGridWorld()
print("Training Rainbow DQN on medium gridworld...")
NumActionsOverTimeRainbowDQNMedium, RewardAtNumberOfActionsMediumRainbowDQN = RainbowDQNMediumGridWorld()
print("Training Beyond The Rainbow DQN on medium gridworld...")
NumActionsOverTimeBeyondTheRainbowDQNMedium, RewardAtNumberOfActionsMediumBeyondTheRainbowDQN = BeyondTheRainbowDQNMediumGridWorld()
plt.plot(NumActionsOverTimeDQNMedium, RewardAtNumberOfActionsMedium,label="DQN")
plt.plot(NumActionsOverTimeRainbowDQNMedium,RewardAtNumberOfActionsMediumRainbowDQN,label="Rainbow DQN")
plt.plot(NumActionsOverTimeBeyondTheRainbowDQNMedium,RewardAtNumberOfActionsMediumBeyondTheRainbowDQN,label="Beyond The Rainbow DQN")
plt.xlabel("Total Actions/Steps Taken In The Gridworld")
plt.ylabel("Reward")
plt.title("Rl Algorithms Learning Effectiveness Over Time, Number Of Training Actions Vs Reward For Medium Gridworld")
plt.legend()
plt.grid()
plt.show()"""

#Running DQN, RainbowDQN and Beyond the Rainbow DQN on the large gridworld, each algorithm gets 1500 episodes of training.
print("Training DQN on large gridworld...") 
NumActionsOverTimeDQNLarge, RewardAtNumberOfActionsLarge = DQNLargeGridWorld()
print("Training Rainbow DQN on large gridworld...")
NumActionsOverTimeRainbowDQNLarge, RewardAtNumberOfActionsLargeRainbowDQN = RainbowDQNLargeGridWorld()
print("Training Beyond The Rainbow DQN on large gridworld...")
NumActionsOverTimeBeyondTheRainbowDQNLarge, RewardAtNumberOfActionsLargeBeyondTheRainbowDQN = BeyondTheRainbowDQNLargeGridWorld()
plt.plot(NumActionsOverTimeDQNLarge, RewardAtNumberOfActionsLarge,label="DQN")
plt.plot(NumActionsOverTimeRainbowDQNLarge,RewardAtNumberOfActionsLargeRainbowDQN,label="Rainbow DQN")
plt.plot(NumActionsOverTimeBeyondTheRainbowDQNLarge,RewardAtNumberOfActionsLargeBeyondTheRainbowDQN,label="Beyond The Rainbow DQN")
plt.plot(NumActionsOverTimeDQNLarge, RewardAtNumberOfActionsLarge,label="DQN")
plt.plot(NumActionsOverTimeRainbowDQNLarge,RewardAtNumberOfActionsLargeRainbowDQN,label="Rainbow DQN")
plt.xlabel("Total Actions/Steps Taken In The Gridworld")
plt.ylabel("Reward")
plt.title("Rl Algorithms Learning Effectiveness Over Time, Number Of Training Actions Vs Reward For Large Gridworld")
plt.legend()
plt.grid()
plt.show()
