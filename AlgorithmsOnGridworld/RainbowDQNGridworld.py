#Alex Eliseev
#DS8013
#Script contains methods that run Rainbow DQN on the small, medium and large gridworlds
#Each world has a dedicated script but, it is 
#The Rainbow DQN also learns episodically, which each episode being it essentially doing a run through the Gridworld.
#At the end each episode the number of total actions (including those in previous episodes) 
#and reward for that episode are recorded so that total actions taken vs average reward can be plotted

#Rainbow DQN function for the small grid world. Runs a variant of the standard DQN. Rainbow DQN was also made by Deepmind lab, published in 2017
#The new additions over the regular DQN was the addition of the Double DQN, Dueling DQN, Noisy Networks, Distributional RL, N-Step and Prioritized Replay features
#End of each episode is the equivalent of doing a run across the gridworld, hence at the end of each run regardless of end ending
#due to running out of steps or reaching the exit, the total number of actions across history is recorded along with the reward for that episode/run
def RainbowDQNSmallGridWorld():
    #Imports
    import numpy as np
    from GridWorlds.SmallGridworld import SmallGridworld, SmallGridworldReward,GetNextStateSmall
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import deque #This is needed to make the prioritized replay.
    import random
    import copy
    loss_fn = nn.MSELoss()

    #Creating Rainbow DQN model and Target DQN that will be be the double DQN feature and provie stability
    SmallWorld = SmallGridworld()
    ActionSize = 4
    StateSize = 2
    Rows = len(SmallWorld)
    Columns = len(SmallWorld[0])

    #Noisy linear, does gaussian factorization to manage noise
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
    
    #Second class for creating the dueling feature of Rainbow DQN, essentially has 3 seperate neural networks inside of it
    #They come from essentially spliting up the original single DQN into 3 mini ones that each can learn a certain aspect of getting across the gridworld
    #Also each has a NoisyLinear convolution layer as opposed to just linear like for the regular DQN to better manage noise
    #It uses gaussian distrbution the manual implementation was very diff but, PyTorch has a library available for the layer.
    class RainbowDQN(nn.Module):
        def __init__(self):
            super().__init__()
            #Learns larger scale patterns and features of related to state
            self.Feature = nn.Sequential(
                NoisyLinear(StateSize,128),
                nn.ReLU(),
                NoisyLinear(128,64),
                nn.ReLU()
            )
            #Learns Q-value patterns/trends
            self.Value = nn.Sequential(
                NoisyLinear(64,32),
                nn.ReLU(),
                NoisyLinear(32,1)
            )
            #Learns local actions patterns
            self.Advantage = nn.Sequential(
                NoisyLinear(64,32),
                nn.ReLU(),
                NoisyLinear(32,ActionSize)
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
    NoisyRainbowDQN = RainbowDQN()
    TargetDQN = copy.deepcopy(NoisyRainbowDQN)
    TargetDQN.eval()
    TargetUpdateFrequency = 20
    Epsilon = 1.0

    #Global variables, these are gonna be returned to the main
    NumActions = []
    AverageRewardAtNumOfActions = []
    TotalActions = 0

    #Next setting up the prioritized replay buffer with N-step = 3, paper used 3
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
    def EncodeState(State,Rows,Columns):
        return torch.tensor([State[0]/Rows,State[1]/Columns],dtype=torch.float32)
    
    #Environment step of the agent adapted to the DQN
    def Step(State,Action,Grid):
        NextState = GetNextStateSmall(State,Action)
        Reward = Grid[NextState[0]][NextState[1]]
        ReachedExit = False
        if(NextState[0] == 0 and NextState[1] == (len(Grid[0])-1)): #Next state is terminal state, reached the end
            ReachedExit = True
        return NextState,Reward,ReachedExit
    
    #Now running the main training of the DQN itself on traversing the gridworld
    #Variables
    Episodes = 250 #DQN should hopefully be able to learn the small gridworld in 100 episodes.
    DiscountRate = 0.997 #Discount rate from research paper
    LearningRate = 1e-4 #Learning rate used in the research paper
    ACTIONS = [0,1,2,3]
        #Other variables declaration
    ReplayBuffer =  PrioritizedReplay()
    BatchSize=256 #Otherwise my PC will explode
    Optimizer = optim.Adam(NoisyRainbowDQN.parameters(),lr=LearningRate) #Adam optimizer was used in the research paper
    #N-Step
    n = 3
    NBuffer = deque(maxlen=n)

    #Running 1000 episodes of training
    for i in range(Episodes):
        State = [len(SmallWorld)-1,0] #Starting in bottom left
        MaxActionsPerEpisode = (Rows + Columns) * 5 #Ideal solution requires less steps than even this.
        ActionsInEpisode = 0
        EpisodeRewardTotal = 0
        Done = False
        NoisyRainbowDQN.reset_noise() #Resetting noise filters
        TargetDQN.reset_noise()
        
        #Agent explores the grid world until they reach the exit, this constitutes 1 episode.
        while not Done and (ActionsInEpisode < MaxActionsPerEpisode):
            ActionsInEpisode += 1
            state = EncodeState(State,Rows,Columns).unsqueeze(0)

            #Epsilon greedy, in the research paper there was optional, they tested with and without
            if np.random.rand() <Epsilon:
                Action = np.random.choice(ACTIONS) #Choosing random action to perform/exploring
            else:
                with torch.no_grad():
                    Action = torch.argmax(NoisyRainbowDQN(state)).item() #Otherwise doing previously best discovered action
            
            #Getting NextState, Reward and if we reached the end based on current state and chosen action
            NextState,Reward,Done = Step(State,Action,SmallWorld)
            #Adding prioritized replay
            NBuffer.append((State,Action,Reward,NextState,Done))
            if len(NBuffer) == n: #Reached length 3
                Reward = sum([NBuffer[y][2] * (DiscountRate ** y) for y in range(n)])
                state,action,_,_,_ = NBuffer[0]
                _,_,_,nextstate,d = NBuffer[-1]
                ReplayBuffer.Add((state,action,Reward,nextstate,d))

            nextstate = EncodeState(NextState,Rows,Columns)

            #Learning using the buffer and N-step
            if len(ReplayBuffer.Buffer) >= BatchSize: #Time to reset the buffer, hence learning from it before reseting
                Batch, IdxStates, Weights = ReplayBuffer.Sample(BatchSize)
                States,Actions,Rewards,NextStates,Dones = zip(*Batch)

                #Converting all results into tensors
                States = torch.stack([EncodeState(s,Rows,Columns) for s in States])
                NextStates = torch.stack([EncodeState(s,Rows,Columns) for s in NextStates])
                Actions = torch.tensor(Actions)
                Rewards = torch.tensor(Rewards,dtype=torch.float32)
                Dones = torch.tensor(Dones,dtype=torch.float32)

                #Attaining Q values from the state and action tensors
                Qval = NoisyRainbowDQN(States)
                QSA = Qval.gather(1,Actions.unsqueeze(1)).squeeze()

                #Now doing the learning from the Q-Vals, but using the entire batch
                with torch.no_grad():
                    NextActions = NoisyRainbowDQN(NextStates).argmax(dim=1,keepdim=True)
                    NextQ = TargetDQN(NextStates).gather(1,NextActions).squeeze()
                    Targets = Rewards + DiscountRate * NextQ * (1-Dones)
                
                #Calculating errors
                TDErrors = (QSA - Targets).detach().cpu().numpy()
                Loss = (Weights *(QSA - Targets).pow(2)).mean()
                Optimizer.zero_grad()
                Loss.backward()
                torch.nn.utils.clip_grad_norm_(NoisyRainbowDQN.parameters(), 1.0)
                Optimizer.step()
                ReplayBuffer.Update(IdxStates,TDErrors)

            #Updating variables
            State = NextState
            EpisodeRewardTotal += Reward
            TotalActions += 1 #1 action taken
        
        #End of episode operations
        Epsilon = max(0.01,Epsilon * 0.995) #E-greedy end in paper used was 0.01, decay every 8M frames, i can decay every episode

        #Updating target network
        if (i % TargetUpdateFrequency) == 0:
            TargetDQN.load_state_dict(NoisyRainbowDQN.state_dict())

        #End of episode, recording total number of training actions up to this point + reward of this episode
        print("Episode:", i, EpisodeRewardTotal) #Tracking progress
        NumActions.append(TotalActions)
        AverageRewardAtNumOfActions.append(EpisodeRewardTotal)
    
    #Returning the final results of training for the gridworld
    return NumActions,AverageRewardAtNumOfActions

#Repeating same function as above but for the medium gridworld
def RainbowDQNMediumGridWorld():
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

    #Noisy linear, does gaussian factorization to manage noise
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
    
    #Second class for creating the dueling feature of Rainbow DQN, essentially has 3 seperate neural networks inside of it
    #They come from essentially spliting up the original single DQN into 3 mini ones that each can learn a certain aspect of getting across the gridworld
    #Also each has a NoisyLinear convolution layer as opposed to just linear like for the regular DQN to better manage noise
    #It uses gaussian distrbution the manual implementation was very diff but, PyTorch has a library available for the layer.
    class RainbowDQN(nn.Module):
        def __init__(self):
            super().__init__()
            #Learns larger scale patterns and features of related to state
            self.Feature = nn.Sequential(
                NoisyLinear(StateSize,128),
                nn.ReLU(),
                NoisyLinear(128,64),
                nn.ReLU()
            )
            #Learns Q-value patterns/trends
            self.Value = nn.Sequential(
                NoisyLinear(64,32),
                nn.ReLU(),
                NoisyLinear(32,1)
            )
            #Learns local actions patterns
            self.Advantage = nn.Sequential(
                NoisyLinear(64,32),
                nn.ReLU(),
                NoisyLinear(32,ActionSize)
            )
        #Runs network
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
    NoisyRainbowDQN = RainbowDQN()
    TargetDQN = copy.deepcopy(NoisyRainbowDQN)
    TargetDQN.eval()
    TargetUpdateFrequency = 20

    #Global variables, these are gonna be returned to the main
    NumActions = []
    AverageRewardAtNumOfActions = []
    TotalActions = 0

    #Next setting up the prioritized replay buffer with N-step = 3, paper used 3
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
    def EncodeState(State,Rows,Columns):
        return torch.tensor([State[0]/Rows,State[1]/Columns],dtype=torch.float32)
    
    #Environment step of the agent adapted to the DQN
    def Step(State,Action,Grid):
        NextState = GetNextStateMedium(State,Action)
        Reward = Grid[NextState[0]][NextState[1]]
        ReachedExit = False
        if(NextState[0] == 0 and NextState[1] == (len(Grid[0])-1)): #Next state is terminal state, reached the end
            ReachedExit = True
        return NextState,Reward,ReachedExit
    
    #Now running the main training of the DQN itself on traversing the gridworld
    #Variables
    Episodes = 500 #DQN should hopefully be able to learn the small gridworld in 100 episodes.
    DiscountRate = 0.997 #Discount rate from research paper
    LearningRate = 1e-4 #Learning rate used in the research paper
    ACTIONS = [0,1,2,3]
        #Other variables declaration
    ReplayBuffer =  PrioritizedReplay()
    BatchSize=256 #Otherwise my PC will explode
    Optimizer = optim.Adam(NoisyRainbowDQN.parameters(),lr=LearningRate) #Adam optimizer was used in the research paper
    #N-Step
    n = 25
    NBuffer = deque(maxlen=n)
    Epsilon = 1.0

    #Running 1000 episodes of training
    for i in range(Episodes):
        State = [len(MediumWorld)-1,0] #Starting in bottom left
        MaxActionsPerEpisode = (Rows + Columns) * 5 #Ideal solution requires less steps than even this.
        ActionsInEpisode = 0
        EpisodeRewardTotal = 0
        Done = False
        NoisyRainbowDQN.reset_noise() #Resetting noise filters
        TargetDQN.reset_noise()
        
        #Agent explores the grid world until they reach the exit, this constitutes 1 episode.
        while not Done and (ActionsInEpisode < MaxActionsPerEpisode):
            ActionsInEpisode += 1
            state = EncodeState(State,Rows,Columns).unsqueeze(0)

            #Epsilon greedy, in the research paper there was optional, they tested with and without
            if np.random.rand() <Epsilon:
                Action = np.random.choice(ACTIONS) #Choosing random action to perform/exploring
            else:
                with torch.no_grad():
                    Action = torch.argmax(NoisyRainbowDQN(state)).item() #Otherwise doing previously best discovered action
            
            #Getting NextState, Reward and if we reached the end based on current state and chosen action
            NextState,Reward,Done = Step(State,Action,MediumWorld)
            #Adding prioritized replay
            NBuffer.append((State,Action,Reward,NextState,Done))
            if len(NBuffer) == n: #Reached length 3
                Reward = sum([NBuffer[y][2] * (DiscountRate ** y) for y in range(n)])
                state,action,_,_,_ = NBuffer[0]
                _,_,_,nextstate,d = NBuffer[-1]
                ReplayBuffer.Add((state,action,Reward,nextstate,d))

            nextstate = EncodeState(NextState,Rows,Columns)

            #Learning using the buffer and N-step
            if len(ReplayBuffer.Buffer) >= BatchSize: #Time to reset the buffer, hence learning from it before reseting
                Batch, IdxStates, Weights = ReplayBuffer.Sample(BatchSize)
                States,Actions,Rewards,NextStates,Dones = zip(*Batch)

                #Converting all results into tensors
                States = torch.stack([EncodeState(s,Rows,Columns) for s in States])
                NextStates = torch.stack([EncodeState(s,Rows,Columns) for s in NextStates])
                Actions = torch.tensor(Actions)
                Rewards = torch.tensor(Rewards,dtype=torch.float32)
                Dones = torch.tensor(Dones,dtype=torch.float32)

                #Attaining Q values from the state and action tensors
                Qval = NoisyRainbowDQN(States)
                QSA = Qval.gather(1,Actions.unsqueeze(1)).squeeze()

                #Now doing the learning from the Q-Vals, but using the entire batch
                with torch.no_grad():
                    NextActions = NoisyRainbowDQN(NextStates).argmax(dim=1,keepdim=True)
                    NextQ = TargetDQN(NextStates).gather(1,NextActions).squeeze()
                    Targets = Rewards + DiscountRate * NextQ * (1-Dones)
                
                #Calculating errors
                TDErrors = (QSA - Targets).detach().cpu().numpy()
                Loss = (Weights *(QSA - Targets).pow(2)).mean()
                Optimizer.zero_grad()
                Loss.backward()
                torch.nn.utils.clip_grad_norm_(NoisyRainbowDQN.parameters(), 1.0)
                Optimizer.step()
                ReplayBuffer.Update(IdxStates,TDErrors)

            #Updating variables
            State = NextState
            EpisodeRewardTotal += Reward
            TotalActions += 1 #1 action taken
        
        #End of episode operations
        Epsilon = max(0.01,Epsilon * 0.995) #E-greedy end in paper used was 0.01, decay every 8M frames, i can decay every episode

        #Updating target network
        if (i % TargetUpdateFrequency) == 0:
            TargetDQN.load_state_dict(NoisyRainbowDQN.state_dict())

        #End of episode, recording total number of training actions up to this point + reward of this episode
        print("Episode:", i, EpisodeRewardTotal) #Tracking progress
        NumActions.append(TotalActions)
        AverageRewardAtNumOfActions.append(EpisodeRewardTotal)
    
    #Returning the final results of training for the gridworld
    return NumActions,AverageRewardAtNumOfActions

#Repeating same function as above but for the large gridworld
def RainbowDQNLargeGridWorld():
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

    #Noisy linear, does gaussian factorization to manage noise
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
    
    #Second class for creating the dueling feature of Rainbow DQN, essentially has 3 seperate neural networks inside of it
    #They come from essentially spliting up the original single DQN into 3 mini ones that each can learn a certain aspect of getting across the gridworld
    #Also each has a NoisyLinear convolution layer as opposed to just linear like for the regular DQN to better manage noise
    #It uses gaussian distrbution the manual implementation was very diff but, PyTorch has a library available for the layer.
    class RainbowDQN(nn.Module):
        def __init__(self):
            super().__init__()
            #Learns larger scale patterns and features of related to state
            self.Feature = nn.Sequential(
                NoisyLinear(StateSize,128),
                nn.ReLU(),
                NoisyLinear(128,64),
                nn.ReLU()
            )
            #Learns Q-value patterns/trends
            self.Value = nn.Sequential(
                NoisyLinear(64,32),
                nn.ReLU(),
                NoisyLinear(32,1)
            )
            #Learns local actions patterns
            self.Advantage = nn.Sequential(
                NoisyLinear(64,32),
                nn.ReLU(),
                NoisyLinear(32,ActionSize)
            )
        #Runs network
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
    NoisyRainbowDQN = RainbowDQN()
    TargetDQN = copy.deepcopy(NoisyRainbowDQN)
    TargetDQN.eval()
    TargetUpdateFrequency = 20

    #Global variables, these are gonna be returned to the main
    NumActions = []
    AverageRewardAtNumOfActions = []
    TotalActions = 0

    #Next setting up the prioritized replay buffer with N-step = 3, paper used 3
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
    def EncodeState(State,Rows,Columns):
        return torch.tensor([State[0]/Rows,State[1]/Columns],dtype=torch.float32)
    
    #Environment step of the agent adapted to the DQN
    def Step(State,Action,Grid):
        NextState = GetNextStateLarge(State,Action)
        Reward = Grid[NextState[0]][NextState[1]]
        ReachedExit = False
        if(NextState[0] == 0 and NextState[1] == (len(Grid[0])-1)): #Next state is terminal state, reached the end
            ReachedExit = True
        return NextState,Reward,ReachedExit
    
    #Now running the main training of the DQN itself on traversing the gridworld
    #Variables
    Episodes = 500 #DQN should hopefully be able to learn the small gridworld in 100 episodes.
    DiscountRate = 0.997 #Discount rate from research paper
    LearningRate = 1e-4 #Learning rate used in the research paper
    ACTIONS = [0,1,2,3]
        #Other variables declaration
    ReplayBuffer =  PrioritizedReplay()
    BatchSize=256 #Otherwise my PC will explode
    Optimizer = optim.Adam(NoisyRainbowDQN.parameters(),lr=LearningRate) #Adam optimizer was used in the research paper
    #N-Step
    n = 25
    NBuffer = deque(maxlen=n)
    Epsilon = 1.0

    #Running 1000 episodes of training
    for i in range(Episodes):
        State = [len(LargeWorld)-1,0] #Starting in bottom left
        MaxActionsPerEpisode = (Rows + Columns) * 5 #Ideal solution requires less steps than even this.
        ActionsInEpisode = 0
        EpisodeRewardTotal = 0
        Done = False
        NoisyRainbowDQN.reset_noise() #Resetting noise filters
        TargetDQN.reset_noise()
        
        #Agent explores the grid world until they reach the exit, this constitutes 1 episode.
        while not Done and (ActionsInEpisode < MaxActionsPerEpisode):
            ActionsInEpisode += 1
            state = EncodeState(State,Rows,Columns).unsqueeze(0)

            #Epsilon greedy, in the research paper there was optional, they tested with and without
            if np.random.rand() <Epsilon:
                Action = np.random.choice(ACTIONS) #Choosing random action to perform/exploring
            else:
                with torch.no_grad():
                    Action = torch.argmax(NoisyRainbowDQN(state)).item() #Otherwise doing previously best discovered action
            
            #Getting NextState, Reward and if we reached the end based on current state and chosen action
            NextState,Reward,Done = Step(State,Action,LargeWorld)
            #Adding prioritized replay
            NBuffer.append((State,Action,Reward,NextState,Done))
            if len(NBuffer) == n: #Reached length 3
                Reward = sum([NBuffer[y][2] * (DiscountRate ** y) for y in range(n)])
                Reward = Reward/100
                state,action,_,_,_ = NBuffer[0]
                _,_,_,nextstate,d = NBuffer[-1]
                ReplayBuffer.Add((state,action,Reward,nextstate,d))

            nextstate = EncodeState(NextState,Rows,Columns)

            #Learning using the buffer and N-step
            if len(ReplayBuffer.Buffer) >= BatchSize: #Time to reset the buffer, hence learning from it before reseting
                Batch, IdxStates, Weights = ReplayBuffer.Sample(BatchSize)
                States,Actions,Rewards,NextStates,Dones = zip(*Batch)

                #Converting all results into tensors
                States = torch.stack([EncodeState(s,Rows,Columns) for s in States])
                NextStates = torch.stack([EncodeState(s,Rows,Columns) for s in NextStates])
                Actions = torch.tensor(Actions)
                Rewards = torch.tensor(Rewards,dtype=torch.float32)
                Dones = torch.tensor(Dones,dtype=torch.float32)

                #Attaining Q values from the state and action tensors
                Qval = NoisyRainbowDQN(States)
                QSA = Qval.gather(1,Actions.unsqueeze(1)).squeeze()

                #Now doing the learning from the Q-Vals, but using the entire batch
                with torch.no_grad():
                    NextActions = NoisyRainbowDQN(NextStates).argmax(dim=1,keepdim=True)
                    NextQ = TargetDQN(NextStates).gather(1,NextActions).squeeze()
                    Targets = Rewards + DiscountRate * NextQ * (1-Dones)
                
                #Calculating errors
                TDErrors = (QSA - Targets).detach().cpu().numpy()
                Loss = (Weights *(QSA - Targets).pow(2)).mean()
                Optimizer.zero_grad()
                Loss.backward()
                torch.nn.utils.clip_grad_norm_(NoisyRainbowDQN.parameters(), 1.0)
                Optimizer.step()
                ReplayBuffer.Update(IdxStates,TDErrors)

            #Updating variables
            State = NextState
            EpisodeRewardTotal += Reward
            TotalActions += 1 #1 action taken
        
        #End of episode operations
        Epsilon = max(0.01,Epsilon * 0.995) #E-greedy end in paper used was 0.01, decay every 8M frames, i can decay every episode

        #Updating target network
        if (i % TargetUpdateFrequency) == 0:
            TargetDQN.load_state_dict(NoisyRainbowDQN.state_dict())

        #End of episode, recording total number of training actions up to this point + reward of this episode
        print("Episode:", i, EpisodeRewardTotal) #Tracking progress
        NumActions.append(TotalActions)
        AverageRewardAtNumOfActions.append(EpisodeRewardTotal)
    
    #Returning the final results of training for the gridworld
    return NumActions,AverageRewardAtNumOfActions


    


