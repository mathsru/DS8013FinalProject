#Alex Eliseev
#DS8013
#Script contains methods that run Beyond The Rainbow DQN from the research paper on the small, medium and large gridworlds (Excluding IQN and Max Pooling Not applicable)
#Each world has a dedicated script but, it is 
#The Beyond The Rainbow DQN also learns episodically, which each episode being it essentially doing a run through the Gridworld.
#At the end each episode the number of total actions (including those in previous episodes) 
#and reward for that episode are recorded so that total actions taken vs average reward can be plotted

#Beyond The Rainbow DQN function for the small grid world. Is a modification of the Rainbow DQN with new features from the research paper
#Such as Impala Architecture, Spectral Normalization, Munchaussen and Vectorized Environments, Adapative Max Polling does not apply to GridWorlds and Implict Quantile Network that would replace C51 is simply overkil
#Features kept from regular rainbow are N-Step, Prioritized Replay, Dueling, Noisy Networks. Features not added are C51/IQN and double DQN replaced by maunchaussen
#End of each episode is the equivalent of doing a run across the gridworld, hence at the end of each run regardless of end ending
#due to running out of steps or reaching the exit, the total number of actions across history is recorded along with the reward for that episode/run
def BeyondTheRainbowDQNSmallGridWorld():
    #Imports
    import numpy as np
    from GridWorlds.SmallGridworld import SmallGridworld, SmallGridworldReward,GetNextStateSmall
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import deque #This is needed to make the prioritized replay.
    import random
    import copy

    #Creating Rainbow DQN model and Target DQN that will be be the double DQN feature and provie stability
    SmallWorld = SmallGridworld()
    ActionSize = 4
    StateSize = 2
    Rows = len(SmallWorld)
    Columns = len(SmallWorld[0])

    #Noisy linear, does gaussian factorization to manage noise, kept from rainbow, untouched
    class NoisyLinear(nn.Module):
        def __init__(self, in_f, out_f, std_init=0.5):
            super().__init__()
            self.in_f = in_f
            self.out_f = out_f

            self.weight_mu = nn.Parameter(torch.empty(out_f, in_f))
            self.weight_sigma = nn.Parameter(torch.empty(out_f, in_f))
            self.register_buffer("weight_epsilon", torch.empty(out_f, in_f))

            self.bias_mu = nn.Parameter(torch.empty(out_f))
            self.bias_sigma = nn.Parameter(torch.empty(out_f))
            self.register_buffer("bias_epsilon", torch.empty(out_f))

            self.reset_parameters(std_init)
            self.reset_noise()
        #Resetting params
        def reset_parameters(self, std_init):
            mu_range = 1 / self.in_f ** 0.5
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.bias_mu.data.uniform_(-mu_range, mu_range)

            self.weight_sigma.data.fill_(std_init / self.in_f ** 0.5)
            self.bias_sigma.data.fill_(std_init / self.out_f ** 0.5)
        #Reseting noise weights
        def reset_noise(self):
            eps_in = torch.randn(self.in_f)
            eps_out = torch.randn(self.out_f)

            self.weight_epsilon.copy_(eps_out.outer(eps_in))
            self.bias_epsilon.copy_(eps_out)
            #Updating wheights
        def forward(self, x):
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return nn.functional.linear(x, w, b)
    
    #Dueling rainbow class from Rainbow DQN, the new addition is the IMPALA-Style Architecture
    #The main idea is having a shared encoder with normalization layers for the main neural network.
    class BeyondTheRainbowDQN(nn.Module):
        def __init__(self):
            super().__init__()
            #Learns larger scale patterns and features of related to state
            self.Feature = nn.Sequential(
                NoisyLinear(StateSize,128),
                nn.LayerNorm(128), #Normalization layers have been added
                nn.ReLU(),
                NoisyLinear(128,128),
                nn.LayerNorm(128),
                nn.ReLU()
            )
            #Learns Q-value patterns/trends
            self.Value = nn.Sequential(
                NoisyLinear(128,64),
                nn.ReLU(),
                NoisyLinear(64,1)
            )
            #Learns local actions patterns
            self.Advantage = nn.Sequential(
                NoisyLinear(128,64),
                nn.ReLU(),
                NoisyLinear(64,ActionSize)
            )
        #Updates network weights
        def forward(self,X):
            features = self.Feature(X)
            Value = self.Value(features)
            Advantage = self.Advantage(features)
            return Value + (Advantage - Advantage.mean(dim=1,keepdim=True)) 
        #Resetting noise
        def reset_noise(self):
            for module in self.children():
                if hasattr(module, "reset_noise") and module is not self:
                    module.reset_noise()

    #Setting up the Rainbow DQN and the Target DQN which is a second parallel neural network which is the double DQN feature 
    #The target network will help q-values from exploding or going to 0 and make them more stable
    BeyondRainbowDQN = BeyondTheRainbowDQN()
    #Target DQN gone, no more double DQN replaced by muncahussen
    TargetUpdateFrequency = 20
    Epsilon = 1.0
    #Global variables, these are gonna be returned to the main
    NumActions = []
    AverageRewardAtNumOfActions = []
    TotalActions = 0

    #Next setting up the prioritized replay buffer with N-step = 3, paper used 3, no changes with BTR
    class PrioritizedReplay:
        def __init__(self,Capacity=10000,Alpha=0.2): #Paper used 0.2 alpha and Capacity of 10 million, my PC will explode and my grid worlds are small so oding 10000 steps
            self.Capacity = Capacity
            self.Alpha = Alpha
            self.Buffer = []
            self.Priorities = np.zeros(Capacity,dtype=np.float32)
            self.Pos = 0
        
        def Add(self,transition):
            MaxP = self.Priorities.max() if self.Buffer else 1.0
            if len(self.Buffer) < self.Capacity:
                self.Buffer.append(transition) #Appending transition if there is room in the buffer
            else:
                self.Buffer[self.Pos] = transition
            self.Priorities[self.Pos] = MaxP
            self.Pos = (self.Pos + 1) % self.Capacity
        
        #Sampling the buffer
        def Sample(self,BatchSize,beta=0.4):
            Priorities = self.Priorities[:len(self.Buffer)]
            Probabilities = Priorities ** self.Alpha
            Probabilities /= Probabilities.sum()

            IdxStates = np.random.choice(len(self.Buffer),BatchSize,p=Probabilities)
            Samples = [self.Buffer[i] for i in IdxStates]

            #Updating weights using the samples
            Weights = (len(self.Buffer) * Probabilities[IdxStates]) ** (-beta)
            Weights /= Weights.max()

            return Samples,IdxStates,torch.tensor(Weights,dtype=torch.float32)
        
        #Updating priorities
        def Update(self,IdxStates,Errors):
            for i, error in zip(IdxStates,Errors):
                self.Priorities[i] = abs(error) + 1e-5

    #Helper functions, that will be called by the main training loop below, first one encodes the state as a tensor/vector
    def EncodeState(State):
        return torch.tensor([State[0]/Rows,State[1]/Columns],dtype=torch.float32)
    
    
    #Step function to perform single step across grid
    def Step(State,Action,Grid):
        NextState = GetNextStateSmall(State,Action)
        Reward = Grid[NextState[0]][NextState[1]]
        ReachedExit = False
        if(NextState[0] == 0 and NextState[1] == (len(Grid[0])-1)): #Next state is terminal state, reached the end
            ReachedExit = True
        return NextState,Reward,ReachedExit
    
    #Step function for the beyond the rainbow, it batches and does 10 steps at a time.
    #Calls regular Step function to do a single step
    def StepBatched(States,Actions,Grid):
        NextStates = []
        Rewards = []
        Dones = []
        for state,action in zip(States,Actions):
            nextstate,reward,done = Step(state,action,Grid)
            NextStates.append(nextstate)
            Rewards.append(reward)
            Dones.append(done)
        return NextStates,Rewards,Dones
    
    #Now running the main training of the DQN itself on traversing the gridworld
    #Variables
    Episodes = 250 #DQN should hopefully be able to learn the small gridworld in 100 episodes.
    DiscountRate = 0.997 #Discount rate from research paper
    LearningRate = 1e-4 #Learning rate used in the research paper
    Actions = [0,1,2,3]
        #Other variables declaration
    ReplayBuffer =  PrioritizedReplay()
    BatchSize=256 #Otherwise my PC will explode
    Optimizer = optim.Adam(BeyondRainbowDQN.parameters(),lr=LearningRate) #Adam optimizer was used in the research paper
    #N-Step

    #Running 1000 episodes of training
    for i in range(Episodes):
        #New starter, vecotrizing the environment
        NumEnv = 10
        States = [[Rows-1,0] for _ in range(NumEnv)]
        StatesTensor = torch.stack([EncodeState(s) for s in States])
        MaxActionsPerEpisode = (Rows + Columns) * 5 #Ideal solution requires less steps than even this.
        ActionsInEpisode = 0
        EpisodeRewardTotal = 0
        Dones = [False] * NumEnv
        BeyondRainbowDQN.reset_noise() #Resetting noise filters

        
        #Agent explores the grid world until they reach the exit, this constitutes 1 episode.
        while not all(Dones) and (ActionsInEpisode < MaxActionsPerEpisode):
            ActionsInEpisode += 1
            StatesTensor = torch.stack([EncodeState(s) for s in States])
            #Epsilon greedy, in the research paper there was optional, they tested with and without
            #Using new vectorized environment and performing action selection on each
            #Still counting a single action worth of training
            if np.random.rand() <Epsilon:
                ActionsBatch = np.random.choice(Actions,size=NumEnv)
            else:
                with torch.no_grad():
                    ActionsBatch = torch.argmax(BeyondRainbowDQN(StatesTensor),dim=1).numpy()
            
            #Getting NextState, Reward and if we reached the end based on current state and chosen action
            NextStates, Rewards, Dones = StepBatched(States, ActionsBatch, SmallWorld)
            #Adding prioritized replay
            for state, action, reward, nextstate, done in zip(States, ActionsBatch, Rewards, NextStates, Dones):
                ReplayBuffer.Add((state, action, reward, nextstate, done))

            #Learning using the buffer and N-step
            if len(ReplayBuffer.Buffer) >= BatchSize: #Time to reset the buffer, hence learning from it before reseting
                Batch, IdxStates, Weights = ReplayBuffer.Sample(BatchSize)
                BatchStates,BatchActions,BatchRewards,BatchNextStates,BatchDones = zip(*Batch)

                #Turning batch actions, rewards, states, dones, next states into tensors
                BatchActions = torch.tensor(BatchActions,dtype=torch.long)
                StatesTensor = torch.stack([EncodeState(s) for s in BatchStates])
                NextStatesTensor = torch.stack([EncodeState(s) for s in BatchNextStates])
                RewardsTensor = torch.tensor(BatchRewards, dtype=torch.float32)
                DonesTensor = torch.tensor(BatchDones, dtype=torch.float32)

                #Attaining Q values from the state and action tensors
                Qval = BeyondRainbowDQN(StatesTensor)
                QSA = Qval.gather(1,BatchActions.unsqueeze(1)).squeeze()

                #Rather than using the Target neural network to do the Q-learning stabilization, BTR uses
                #Munchaussen Q target, adding a log policy term into reward shaping.
                Tau = 0.03 #Used temperature Tau in the research paper
                Alpha = 0.9 #Also value taken from the research paper
                with torch.no_grad():
                    Qnext = BeyondRainbowDQN(NextStatesTensor)
                    LogPi = torch.log_softmax(Qnext/Tau,dim=1)
                    Pi = torch.softmax(Qnext/Tau,dim=1)
                    MunchausenTerm =Alpha * Tau * LogPi.gather(1,BatchActions.unsqueeze(1)).squeeze()
                    NextQ = (Pi * Qnext).sum(dim=1)
                    Targets = RewardsTensor + MunchausenTerm + DiscountRate * NextQ * (1-DonesTensor)
                
                #Calculating errors
                TDErrors = (QSA - Targets).detach().cpu().numpy()
                Loss = (Weights *(QSA - Targets).pow(2)).mean()
                Optimizer.zero_grad()
                Loss.backward()
                torch.nn.utils.clip_grad_norm_(BeyondRainbowDQN.parameters(), 1.0)
                Optimizer.step()
                ReplayBuffer.Update(IdxStates,TDErrors)

            #Updating variables
            States = NextStates
            EpisodeRewardTotal += sum(Rewards)
            TotalActions += 1 #1 action taken
        
        #End of episode operations
        Epsilon = max(0.01,Epsilon * 0.995) #E-greedy end in paper used was 0.01, decay every 8M frames, i can decay every episode

        #End of episode, recording total number of training actions up to this point + reward of this episode
        print("Episode:", i, EpisodeRewardTotal) #Tracking progress
        NumActions.append(TotalActions)
        AverageRewardAtNumOfActions.append(EpisodeRewardTotal)
    
    #Returning the final results of training for the gridworld
    return NumActions,AverageRewardAtNumOfActions

#Exact same script of beyond the rainbow but for the medium grid world
def BeyondTheRainbowDQNMediumGridWorld():
    #Imports
    import numpy as np
    from GridWorlds.MediumGridworld import MediumGridworld, MediumGridworldReward,GetNextStateMedium
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import deque #This is needed to make the prioritized replay.
    import random
    import copy

    #Creating Rainbow DQN model and Target DQN that will be be the double DQN feature and provie stability
    MediumWorld = MediumGridworld()
    ActionSize = 4
    StateSize = 2
    Rows = len(MediumWorld)
    Columns = len(MediumWorld[0])

    #Noisy linear, does gaussian factorization to manage noise, kept from rainbow, untouched
    class NoisyLinear(nn.Module):
        def __init__(self, in_f, out_f, std_init=0.5):
            super().__init__()
            self.in_f = in_f
            self.out_f = out_f

            self.weight_mu = nn.Parameter(torch.empty(out_f, in_f))
            self.weight_sigma = nn.Parameter(torch.empty(out_f, in_f))
            self.register_buffer("weight_epsilon", torch.empty(out_f, in_f))

            self.bias_mu = nn.Parameter(torch.empty(out_f))
            self.bias_sigma = nn.Parameter(torch.empty(out_f))
            self.register_buffer("bias_epsilon", torch.empty(out_f))

            self.reset_parameters(std_init)
            self.reset_noise()
        #Resetting params
        def reset_parameters(self, std_init):
            mu_range = 1 / self.in_f ** 0.5
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.bias_mu.data.uniform_(-mu_range, mu_range)

            self.weight_sigma.data.fill_(std_init / self.in_f ** 0.5)
            self.bias_sigma.data.fill_(std_init / self.out_f ** 0.5)
        #Reseting noise weights
        def reset_noise(self):
            eps_in = torch.randn(self.in_f)
            eps_out = torch.randn(self.out_f)

            self.weight_epsilon.copy_(eps_out.outer(eps_in))
            self.bias_epsilon.copy_(eps_out)
            #Updating wheights
        def forward(self, x):
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return nn.functional.linear(x, w, b)
    
    #Dueling rainbow class from Rainbow DQN, the new addition is the IMPALA-Style Architecture
    #The main idea is having a shared encoder with normalization layers for the main neural network.
    class BeyondTheRainbowDQN(nn.Module):
        def __init__(self):
            super().__init__()
            #Learns larger scale patterns and features of related to state
            self.Feature = nn.Sequential(
                NoisyLinear(StateSize,128),
                nn.LayerNorm(128), #Normalization layers have been added
                nn.ReLU(),
                NoisyLinear(128,128),
                nn.LayerNorm(128),
                nn.ReLU()
            )
            #Learns Q-value patterns/trends
            self.Value = nn.Sequential(
                NoisyLinear(128,64),
                nn.ReLU(),
                NoisyLinear(64,1)
            )
            #Learns local actions patterns
            self.Advantage = nn.Sequential(
                NoisyLinear(128,64),
                nn.ReLU(),
                NoisyLinear(64,ActionSize)
            )
        #Updates network weights
        def forward(self,X):
            features = self.Feature(X)
            Value = self.Value(features)
            Advantage = self.Advantage(features)
            return Value + (Advantage - Advantage.mean(dim=1,keepdim=True)) 
        #Resetting noise
        def reset_noise(self):
            for module in self.children():
                if hasattr(module, "reset_noise") and module is not self:
                    module.reset_noise()

    #Setting up the Rainbow DQN and the Target DQN which is a second parallel neural network which is the double DQN feature 
    #The target network will help q-values from exploding or going to 0 and make them more stable
    BeyondRainbowDQN = BeyondTheRainbowDQN()
    #Target DQN gone, no more double DQN replaced by muncahussen
    TargetUpdateFrequency = 20
    Epsilon = 1.0
    #Global variables, these are gonna be returned to the main
    NumActions = []
    AverageRewardAtNumOfActions = []
    TotalActions = 0

    #Next setting up the prioritized replay buffer with N-step = 3, paper used 3, no changes with BTR
    class PrioritizedReplay:
        def __init__(self,Capacity=10000,Alpha=0.2): #Paper used 0.2 alpha and Capacity of 10 million, my PC will explode and my grid worlds are small so oding 10000 steps
            self.Capacity = Capacity
            self.Alpha = Alpha
            self.Buffer = []
            self.Priorities = np.zeros(Capacity,dtype=np.float32)
            self.Pos = 0
        
        def Add(self,transition):
            MaxP = self.Priorities.max() if self.Buffer else 1.0
            if len(self.Buffer) < self.Capacity:
                self.Buffer.append(transition) #Appending transition if there is room in the buffer
            else:
                self.Buffer[self.Pos] = transition
            self.Priorities[self.Pos] = MaxP
            self.Pos = (self.Pos + 1) % self.Capacity
        
        #Sampling the buffer
        def Sample(self,BatchSize,beta=0.4):
            Priorities = self.Priorities[:len(self.Buffer)]
            Probabilities = Priorities ** self.Alpha
            Probabilities /= Probabilities.sum()

            IdxStates = np.random.choice(len(self.Buffer),BatchSize,p=Probabilities)
            Samples = [self.Buffer[i] for i in IdxStates]

            #Updating weights using the samples
            Weights = (len(self.Buffer) * Probabilities[IdxStates]) ** (-beta)
            Weights /= Weights.max()

            return Samples,IdxStates,torch.tensor(Weights,dtype=torch.float32)
        
        #Updating priorities
        def Update(self,IdxStates,Errors):
            for i, error in zip(IdxStates,Errors):
                self.Priorities[i] = abs(error) + 1e-5

    #Helper functions, that will be called by the main training loop below, first one encodes the state as a tensor/vector
    def EncodeState(State):
        return torch.tensor([State[0]/Rows,State[1]/Columns],dtype=torch.float32)
    
    
    #Step function to perform single step across grid
    def Step(State,Action,Grid):
        NextState = GetNextStateMedium(State,Action)
        Reward = Grid[NextState[0]][NextState[1]]
        ReachedExit = False
        if(NextState[0] == 0 and NextState[1] == (len(Grid[0])-1)): #Next state is terminal state, reached the end
            ReachedExit = True
        return NextState,Reward,ReachedExit
    
    #Step function for the beyond the rainbow, it batches and does 10 steps at a time.
    #Calls regular Step function to do a single step
    def StepBatched(States,Actions,Grid):
        NextStates = []
        Rewards = []
        Dones = []
        for state,action in zip(States,Actions):
            nextstate,reward,done = Step(state,action,Grid)
            NextStates.append(nextstate)
            Rewards.append(reward)
            Dones.append(done)
        return NextStates,Rewards,Dones
    
    #Now running the main training of the DQN itself on traversing the gridworld
    #Variables
    Episodes = 500 #DQN should hopefully be able to learn the small gridworld in 100 episodes.
    DiscountRate = 0.997 #Discount rate from research paper
    LearningRate = 1e-4 #Learning rate used in the research paper
    Actions = [0,1,2,3]
        #Other variables declaration
    ReplayBuffer =  PrioritizedReplay()
    BatchSize=256 #Otherwise my PC will explode
    Optimizer = optim.Adam(BeyondRainbowDQN.parameters(),lr=LearningRate) #Adam optimizer was used in the research paper
    #N-Step

    #Running 1000 episodes of training
    for i in range(Episodes):
        #New starter, vecotrizing the environment
        NumEnv = 10
        States = [[Rows-1,0] for _ in range(NumEnv)]
        StatesTensor = torch.stack([EncodeState(s) for s in States])
        MaxActionsPerEpisode = (Rows + Columns) * 5 #Ideal solution requires less steps than even this.
        ActionsInEpisode = 0
        EpisodeRewardTotal = 0
        Dones = [False] * NumEnv
        BeyondRainbowDQN.reset_noise() #Resetting noise filters

        
        #Agent explores the grid world until they reach the exit, this constitutes 1 episode.
        while not all(Dones) and (ActionsInEpisode < MaxActionsPerEpisode):
            ActionsInEpisode += 1
            StatesTensor = torch.stack([EncodeState(s) for s in States])
            #Epsilon greedy, in the research paper there was optional, they tested with and without
            #Using new vectorized environment and performing action selection on each
            #Still counting a single action worth of training
            if np.random.rand() <Epsilon:
                ActionsBatch = np.random.choice(Actions,size=NumEnv)
            else:
                with torch.no_grad():
                    ActionsBatch = torch.argmax(BeyondRainbowDQN(StatesTensor),dim=1).numpy()
            
            #Getting NextState, Reward and if we reached the end based on current state and chosen action
            NextStates, Rewards, Dones = StepBatched(States, ActionsBatch, MediumWorld)
            #Adding prioritized replay
            for state, action, reward, nextstate, done in zip(States, ActionsBatch, Rewards, NextStates, Dones):
                ReplayBuffer.Add((state, action, reward, nextstate, done))

            #Learning using the buffer and N-step
            if len(ReplayBuffer.Buffer) >= BatchSize: #Time to reset the buffer, hence learning from it before reseting
                Batch, IdxStates, Weights = ReplayBuffer.Sample(BatchSize)
                BatchStates,BatchActions,BatchRewards,BatchNextStates,BatchDones = zip(*Batch)

                #Turning batch actions, rewards, states, dones, next states into tensors
                BatchActions = torch.tensor(BatchActions,dtype=torch.long)
                StatesTensor = torch.stack([EncodeState(s) for s in BatchStates])
                NextStatesTensor = torch.stack([EncodeState(s) for s in BatchNextStates])
                RewardsTensor = torch.tensor(BatchRewards, dtype=torch.float32)
                DonesTensor = torch.tensor(BatchDones, dtype=torch.float32)

                #Attaining Q values from the state and action tensors
                Qval = BeyondRainbowDQN(StatesTensor)
                QSA = Qval.gather(1,BatchActions.unsqueeze(1)).squeeze()

                #Rather than using the Target neural network to do the Q-learning stabilization, BTR uses
                #Munchaussen Q target, adding a log policy term into reward shaping.
                Tau = 0.03 #Used temperature Tau in the research paper
                Alpha = 0.9 #Also value taken from the research paper
                with torch.no_grad():
                    Qnext = BeyondRainbowDQN(NextStatesTensor)
                    LogPi = torch.log_softmax(Qnext/Tau,dim=1)
                    Pi = torch.softmax(Qnext/Tau,dim=1)
                    MunchausenTerm =Alpha * Tau * LogPi.gather(1,BatchActions.unsqueeze(1)).squeeze()
                    NextQ = (Pi * Qnext).sum(dim=1)
                    Targets = RewardsTensor + MunchausenTerm + DiscountRate * NextQ * (1-DonesTensor)
                
                #Calculating errors
                TDErrors = (QSA - Targets).detach().cpu().numpy()
                Loss = (Weights *(QSA - Targets).pow(2)).mean()
                Optimizer.zero_grad()
                Loss.backward()
                torch.nn.utils.clip_grad_norm_(BeyondRainbowDQN.parameters(), 1.0)
                Optimizer.step()
                ReplayBuffer.Update(IdxStates,TDErrors)

            #Updating variables
            States = NextStates
            EpisodeRewardTotal += sum(Rewards)
            TotalActions += 1 #1 action taken
        
        #End of episode operations
        Epsilon = max(0.01,Epsilon * 0.995) #E-greedy end in paper used was 0.01, decay every 8M frames, i can decay every episode

        #End of episode, recording total number of training actions up to this point + reward of this episode
        print("Episode:", i, EpisodeRewardTotal) #Tracking progress
        NumActions.append(TotalActions)
        AverageRewardAtNumOfActions.append(EpisodeRewardTotal)
    
    #Returning the final results of training for the gridworld
    return NumActions,AverageRewardAtNumOfActions

#Same beyond the rainbow setup but with the large grid world
def BeyondTheRainbowDQNLargeGridWorld():
    #Imports
    import numpy as np
    from GridWorlds.LargeGridworld import LargeGridworld, LargeGridworldReward,GetNextStateLarge
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import deque #This is needed to make the prioritized replay.
    import random
    import copy

    #Creating Rainbow DQN model and Target DQN that will be be the double DQN feature and provie stability
    LargeWorld = LargeGridworld()
    ActionSize = 4
    StateSize = 2
    Rows = len(LargeWorld)
    Columns = len(LargeWorld[0])

    #Noisy linear, does gaussian factorization to manage noise, kept from rainbow, untouched
    class NoisyLinear(nn.Module):
        def __init__(self, in_f, out_f, std_init=0.5):
            super().__init__()
            self.in_f = in_f
            self.out_f = out_f

            self.weight_mu = nn.Parameter(torch.empty(out_f, in_f))
            self.weight_sigma = nn.Parameter(torch.empty(out_f, in_f))
            self.register_buffer("weight_epsilon", torch.empty(out_f, in_f))

            self.bias_mu = nn.Parameter(torch.empty(out_f))
            self.bias_sigma = nn.Parameter(torch.empty(out_f))
            self.register_buffer("bias_epsilon", torch.empty(out_f))

            self.reset_parameters(std_init)
            self.reset_noise()
        #Resetting params
        def reset_parameters(self, std_init):
            mu_range = 1 / self.in_f ** 0.5
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.bias_mu.data.uniform_(-mu_range, mu_range)

            self.weight_sigma.data.fill_(std_init / self.in_f ** 0.5)
            self.bias_sigma.data.fill_(std_init / self.out_f ** 0.5)
        #Reseting noise weights
        def reset_noise(self):
            eps_in = torch.randn(self.in_f)
            eps_out = torch.randn(self.out_f)

            self.weight_epsilon.copy_(eps_out.outer(eps_in))
            self.bias_epsilon.copy_(eps_out)
            #Updating wheights
        def forward(self, x):
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return nn.functional.linear(x, w, b)
    
    #Dueling rainbow class from Rainbow DQN, the new addition is the IMPALA-Style Architecture
    #The main idea is having a shared encoder with normalization layers for the main neural network.
    class BeyondTheRainbowDQN(nn.Module):
        def __init__(self):
            super().__init__()
            #Learns larger scale patterns and features of related to state
            self.Feature = nn.Sequential(
                NoisyLinear(StateSize,128),
                nn.LayerNorm(128), #Normalization layers have been added
                nn.ReLU(),
                NoisyLinear(128,128),
                nn.LayerNorm(128),
                nn.ReLU()
            )
            #Learns Q-value patterns/trends
            self.Value = nn.Sequential(
                NoisyLinear(128,64),
                nn.ReLU(),
                NoisyLinear(64,1)
            )
            #Learns local actions patterns
            self.Advantage = nn.Sequential(
                NoisyLinear(128,64),
                nn.ReLU(),
                NoisyLinear(64,ActionSize)
            )
        #Updates network weights
        def forward(self,X):
            features = self.Feature(X)
            Value = self.Value(features)
            Advantage = self.Advantage(features)
            return Value + (Advantage - Advantage.mean(dim=1,keepdim=True)) 
        #Resetting noise
        def reset_noise(self):
            for module in self.children():
                if hasattr(module, "reset_noise") and module is not self:
                    module.reset_noise()

    #Setting up the Rainbow DQN and the Target DQN which is a second parallel neural network which is the double DQN feature 
    #The target network will help q-values from exploding or going to 0 and make them more stable
    BeyondRainbowDQN = BeyondTheRainbowDQN()
    #Target DQN gone, no more double DQN replaced by muncahussen
    TargetUpdateFrequency = 20
    Epsilon = 1.0
    #Global variables, these are gonna be returned to the main
    NumActions = []
    AverageRewardAtNumOfActions = []
    TotalActions = 0

    #Next setting up the prioritized replay buffer with N-step = 3, paper used 3, no changes with BTR
    class PrioritizedReplay:
        def __init__(self,Capacity=10000,Alpha=0.2): #Paper used 0.2 alpha and Capacity of 10 million, my PC will explode and my grid worlds are small so oding 10000 steps
            self.Capacity = Capacity
            self.Alpha = Alpha
            self.Buffer = []
            self.Priorities = np.zeros(Capacity,dtype=np.float32)
            self.Pos = 0
        
        def Add(self,transition):
            MaxP = self.Priorities.max() if self.Buffer else 1.0
            if len(self.Buffer) < self.Capacity:
                self.Buffer.append(transition) #Appending transition if there is room in the buffer
            else:
                self.Buffer[self.Pos] = transition
            self.Priorities[self.Pos] = MaxP
            self.Pos = (self.Pos + 1) % self.Capacity
        
        #Sampling the buffer
        def Sample(self,BatchSize,beta=0.4):
            Priorities = self.Priorities[:len(self.Buffer)]
            Probabilities = Priorities ** self.Alpha
            Probabilities /= Probabilities.sum()

            IdxStates = np.random.choice(len(self.Buffer),BatchSize,p=Probabilities)
            Samples = [self.Buffer[i] for i in IdxStates]

            #Updating weights using the samples
            Weights = (len(self.Buffer) * Probabilities[IdxStates]) ** (-beta)
            Weights /= Weights.max()

            return Samples,IdxStates,torch.tensor(Weights,dtype=torch.float32)
        
        #Updating priorities
        def Update(self,IdxStates,Errors):
            for i, error in zip(IdxStates,Errors):
                self.Priorities[i] = abs(error) + 1e-5

    #Helper functions, that will be called by the main training loop below, first one encodes the state as a tensor/vector
    def EncodeState(State):
        return torch.tensor([State[0]/Rows,State[1]/Columns],dtype=torch.float32)
    
    
    #Step function to perform single step across grid
    def Step(State,Action,Grid):
        NextState = GetNextStateLarge(State,Action)
        Reward = Grid[NextState[0]][NextState[1]]
        ReachedExit = False
        if(NextState[0] == 0 and NextState[1] == (len(Grid[0])-1)): #Next state is terminal state, reached the end
            ReachedExit = True
        return NextState,Reward,ReachedExit
    
    #Step function for the beyond the rainbow, it batches and does 10 steps at a time.
    #Calls regular Step function to do a single step
    def StepBatched(States,Actions,Grid):
        NextStates = []
        Rewards = []
        Dones = []
        for state,action in zip(States,Actions):
            nextstate,reward,done = Step(state,action,Grid)
            NextStates.append(nextstate)
            Rewards.append(reward)
            Dones.append(done)
        return NextStates,Rewards,Dones
    
    #Now running the main training of the DQN itself on traversing the gridworld
    #Variables
    Episodes = 500 #DQN should hopefully be able to learn the small gridworld in 100 episodes.
    DiscountRate = 0.997 #Discount rate from research paper
    LearningRate = 1e-4 #Learning rate used in the research paper
    ACTIONS = [0,1,2,3]
        #Other variables declaration
    ReplayBuffer =  PrioritizedReplay()
    BatchSize=256 #Otherwise my PC will explode
    Optimizer = optim.Adam(BeyondRainbowDQN.parameters(),lr=LearningRate) #Adam optimizer was used in the research paper
    #N-Step

    #Running 1000 episodes of training
    for i in range(Episodes):
        #New starter, vecotrizing the environment
        NumEnv = 10
        States = [[Rows-1,0] for _ in range(NumEnv)]
        StatesTensor = torch.stack([EncodeState(s) for s in States])
        MaxActionsPerEpisode = (Rows + Columns) * 5 #Ideal solution requires less steps than even this.
        ActionsInEpisode = 0
        EpisodeRewardTotal = 0
        Dones = [False] * NumEnv
        BeyondRainbowDQN.reset_noise() #Resetting noise filters

        
        #Agent explores the grid world until they reach the exit, this constitutes 1 episode.
        while not all(Dones) and (ActionsInEpisode < MaxActionsPerEpisode):
            ActionsInEpisode += 1
            StatesTensor = torch.stack([EncodeState(s) for s in States])
            #Epsilon greedy, in the research paper there was optional, they tested with and without
            #Using new vectorized environment and performing action selection on each
            #Still counting a single action worth of training
            if np.random.rand() <Epsilon:
                ActionsBatch = np.random.choice(ACTIONS,size=NumEnv)
            else:
                with torch.no_grad():
                    ActionsBatch = torch.argmax(BeyondRainbowDQN(StatesTensor),dim=1).numpy()
            
            #Getting NextState, Reward and if we reached the end based on current state and chosen action
            NextStates, Rewards, Dones = StepBatched(States, ActionsBatch, LargeWorld)
            #Adding prioritized replay
            for state, action, reward, nextstate, done in zip(States, ActionsBatch, Rewards, NextStates, Dones):
                ReplayBuffer.Add((state, action, reward, nextstate, done))

            #Learning using the buffer and N-step
            if len(ReplayBuffer.Buffer) >= BatchSize: #Time to reset the buffer, hence learning from it before reseting
                Batch, IdxStates, Weights = ReplayBuffer.Sample(BatchSize)
                BatchStates,BatchActions,BatchRewards,BatchNextStates,BatchDones = zip(*Batch)

                #Turning batch actions, rewards, states, dones, next states into tensors
                BatchActions = torch.tensor(BatchActions,dtype=torch.long)
                StatesTensor = torch.stack([EncodeState(s) for s in BatchStates])
                NextStatesTensor = torch.stack([EncodeState(s) for s in BatchNextStates])
                RewardsTensor = torch.tensor(BatchRewards, dtype=torch.float32)
                DonesTensor = torch.tensor(BatchDones, dtype=torch.float32)

                #Attaining Q values from the state and action tensors
                Qval = BeyondRainbowDQN(StatesTensor)
                QSA = Qval.gather(1,BatchActions.unsqueeze(1)).squeeze()

                #Rather than using the Target neural network to do the Q-learning stabilization, BTR uses
                #Munchaussen Q target, adding a log policy term into reward shaping.
                Tau = 0.03 #Used temperature Tau in the research paper
                Alpha = 0.9 #Also value taken from the research paper
                with torch.no_grad():
                    Qnext = BeyondRainbowDQN(NextStatesTensor)
                    LogPi = torch.log_softmax(Qnext/Tau,dim=1)
                    Pi = torch.softmax(Qnext/Tau,dim=1)
                    MunchausenTerm =Alpha * Tau * LogPi.gather(1,BatchActions.unsqueeze(1)).squeeze()
                    NextQ = (Pi * Qnext).sum(dim=1)
                    Targets = RewardsTensor + MunchausenTerm + DiscountRate * NextQ * (1-DonesTensor)
                
                #Calculating errors
                TDErrors = (QSA - Targets).detach().cpu().numpy()
                Loss = (Weights *(QSA - Targets).pow(2)).mean()
                Optimizer.zero_grad()
                Loss.backward()
                torch.nn.utils.clip_grad_norm_(BeyondRainbowDQN.parameters(), 1.0)
                Optimizer.step()
                ReplayBuffer.Update(IdxStates,TDErrors)

            #Updating variables
            States = NextStates
            EpisodeRewardTotal += sum(Rewards)
            TotalActions += 1 #1 action taken
        
        #End of episode operations
        Epsilon = max(0.01,Epsilon * 0.995) #E-greedy end in paper used was 0.01, decay every 8M frames, i can decay every episode

        #End of episode, recording total number of training actions up to this point + reward of this episode
        print("Episode:", i, EpisodeRewardTotal) #Tracking progress
        NumActions.append(TotalActions)
        AverageRewardAtNumOfActions.append(EpisodeRewardTotal)
    
    #Returning the final results of training for the gridworld
    return NumActions,AverageRewardAtNumOfActions



    


