# DS8013FinalProject
Testing various types of reinforced learning algorithms to see how their performance varies on learning how to traverse through progressively larger grid worlds. Replication project is based on the research paper: https://arxiv.org/pdf/2411.03820
With the main interest being to see how effectively and effeciently DQN, Rainbow DQN and lastly Beyond The Rainbow DQN became with each new generation.

-> Only requirements are Python and PyTorch

-> GitHub Link To Project

-> FunctionTestsGridworld.py
The main experiment of the project is found in main.py. Where the goal is to run and test the generations of DQN and compare how much more effective each is from the last, by assessing their performance in navigating through GridWorlds. File simply imports the functions from AlgorithmsOnGridWorld and just plots the results. Displaying the rewards earned per run vs number of training done total.

-> GridWorlds -> LargeGridworld.py, MediumGridWorld.py, SmallGridWorld.py
This folder contains variuos sized grid worlds that are called upon in the main to have the RL algorithms learn to navigate through them.They have custom reward() and nextstate() functions as they have different features. Like the large gridworld having a teleporter across the world. The main evaluation with the grid worlds is which RL algorithm can have the maximum amount of reward when reaching the exit of the grid world in the top right of the grid world. All agents begin in the bottom left corner of the world. Learning performance is assesed by the number of actions each RL algorithm under took in training to reach the average reward it has per run at the end. A core aspect of the research paper was that BTR DQN was capable of achieving 5x greater average reward per run after the same volume of training as its predecessor Rainbow DQN. For example after learning for 150 million frames of playing Super Mario Galaxy: Final Level it would average 3x the amount of reward per run of the final level. Reward being how fast it can beat the level.

-> AlgorithmsOnGridWorld -> BeyondTheRainbowDQNGridworld.py, DQNGRidworld.py, RainbowDQNGridworld.py
This folder contains the three generations of DQN implemented based on specifications from the research paper. Each is an upgrade of the previous and they each have 3 implementations each configured to work with the custom grid worlds and their custom reward functions.
