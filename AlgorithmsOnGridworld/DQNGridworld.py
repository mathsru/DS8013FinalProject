#Alex Eliseev
#DS8013
#Script contains methods that run DQN on the small, medium and large gridworlds
#Each world has a dedicated script but, it is 
#The DQN learns episodically, so an episode of traversing through each given gridworld
#At the end each episode the number of total actions (including those in previous episodes) 
#and reward for that episode are recorded so that total actions taken vs average reward can be plotted

#DQN function for the small grid world. Runs a variant of the standard DQN. That was developed back in 2015 by the DeepMind lab.
#End of each episode is the equivalent of doing a run across the gridworld, hence at the end of each run regardless of end ending
#due to running out of steps or reaching the exit, the total number of actions across history is recorded along with the reward for that episode/run
def DQNSmallGridWorld():
    #Imports
    import numpy as np
    from GridWorlds.SmallGridworld import SmallGridworld, SmallGridworldReward,GetNextStateSmall
    import torch
    import torch.nn as nn
    import torch.optim as optim
    loss_fn = nn.MSELoss()

    #Creating DQN model
    SmallWorld = SmallGridworld()
    ActionSize = 4
    StateSize = 2
    Rows = len(SmallWorld)
    Columns = len(SmallWorld[0])
    #Setting up a basic 3 layer neural network with Relu activation from PyTorch.
    BasicDQN = nn.Sequential(
        nn.Linear(StateSize,128), #Batch sizes are downscaled, paper could afford to do 512. Too computationally heavy for me
        nn.ReLU(), #ReLU activation was used in paper
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,ActionSize)
    )

    #Global variables, these are gonna be returned to the main
    NumActions = []
    AverageRewardAtNumOfActions = []
    TotalActions = 0

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
    Episodes = 500 #DQN should hopefully be able to learn the small gridworld in 100 episodes.
    DiscountRate = 0.997 #Discount rate from research paper
    LearningRate = 1e-4 #Learning rate used in the research paper
    Epsilon = 1
    Actions = [0,1,2,3]
    Optimizer = optim.Adam(BasicDQN.parameters(),lr=LearningRate) #Adam optimizer was used in the research paper

    #Running 1000 episodes of training
    for i in range(Episodes):
        State = [len(SmallWorld)-1,0] #Starting in bottom left
        MaxActionsPerEpisode = (Rows + Columns) * 5 #Ideal solution requires less steps than even this.
        ActionsInEpisode = 0
        EpisodeRewardTotal = 0
        Done = False
        
        #Agent explores the grid world until they reach the exit, this constitutes 1 episode.
        while not Done and (ActionsInEpisode < MaxActionsPerEpisode):
            ActionsInEpisode += 1
            state = EncodeState(State,Rows,Columns)

            #Epsilon greedy, in the research paper there was optional, they tested with and without
            if np.random.rand() <Epsilon:
                Action = np.random.choice(Actions) #Choosing random action to perform/exploring
            else:
                with torch.no_grad():
                    Action = torch.argmax(BasicDQN(state)).item() #Otherwise doing previously best discovered action
            
            #Getting NextState, Reward and if we reached the end based on current state and chosen action
            NextState,Reward,Done = Step(State,Action,SmallWorld)
            nextstate = EncodeState(NextState,Rows,Columns)

            #Q-learning
            with torch.no_grad():
                NextQ = torch.max(BasicDQN(nextstate)).item()
                Target = Reward if Done else Reward + DiscountRate * NextQ
            
            
            Predicted = BasicDQN(state)[Action]
            TargetTensor = torch.tensor(Target, dtype=torch.float32)
            Loss = loss_fn(Predicted,TargetTensor)
            Optimizer.zero_grad()
            Loss.backward()
            Optimizer.step()

            #Updating variables
            State = NextState
            EpisodeRewardTotal += Reward
            TotalActions += 1 #1 action taken
        
        #Decaying exploration
        Epsilon = max(0.01,Epsilon * 0.999) #E-greedy end in paper used was 0.01, decay every 8M frames, i can decay every episode

        #End of episode, recording total number of training actions up to this point + reward of this episode
        print("Episode:", i) #Tracking progress
        NumActions.append(TotalActions)
        AverageRewardAtNumOfActions.append(EpisodeRewardTotal)
    
    #Returning the final results of training for the gridworld
    return NumActions,AverageRewardAtNumOfActions


#DQN for the medium gridworld, number of episodes increased for longer training as grid world now is size 25x25
def DQNMediumGridWorld():
    #Imports
    import numpy as np
    from GridWorlds.MediumGridworld import MediumGridworld, MediumGridworldReward,GetNextStateMedium #Proceed from here
    import torch
    import torch.nn as nn
    import torch.optim as optim
    loss_fn = nn.MSELoss()

    #Creating DQN model
    MediumWorld = MediumGridworld()
    ActionSize = 4
    StateSize = 2
    Rows = len(MediumWorld)
    Columns = len(MediumWorld[0])
    #Setting up a basic 3 layer neural network with Relu activation from PyTorch.
    BasicDQN = nn.Sequential(
        nn.Linear(StateSize,128), #Batch sizes are downscaled, paper could afford to do 512. Too computationally heavy for me
        nn.ReLU(), #ReLU activation was used in paper
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,ActionSize)
    )

    #Global variables, these are gonna be returned to the main
    NumActions = []
    AverageRewardAtNumOfActions = []
    TotalActions = 0

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
    Episodes = 500 #DQN should hopefully be able to learn the medium gridworld in 1750 episodes.
    DiscountRate = 0.997 #Discount rate from research paper
    LearningRate = 1e-4 #Learning rate used in the research paper
    Epsilon = 1
    Actions = [0,1,2,3]
    Optimizer = optim.Adam(BasicDQN.parameters(),lr=LearningRate) #Adam optimizer was used in the research paper

    #Running 250 episodes of training
    for i in range(Episodes):
        State = [len(MediumWorld)-1,0] #Starting in bottom left
        MaxActionsPerEpisode = (Rows + Columns) * 10 #Ideal solution requires less steps than even this.
        ActionsInEpisode = 0
        EpisodeRewardTotal = 0
        Done = False
        
        #Agent explores the grid world until they reach the exit, this constitutes 1 episode.
        while not Done and (ActionsInEpisode < MaxActionsPerEpisode):
            ActionsInEpisode += 1
            state = EncodeState(State,Rows,Columns)

            #Epsilon greedy, in the research paper there was optional, they tested with and without
            if np.random.rand() <Epsilon:
                Action = np.random.choice(Actions) #Choosing random action to perform/exploring
            else:
                with torch.no_grad():
                    Action = torch.argmax(BasicDQN(state)).item() #Otherwise doing previously best discovered action
            
            #Getting NextState, Reward and if we reached the end based on current state and chosen action
            NextState,Reward,Done = Step(State,Action,MediumWorld)
            nextstate = EncodeState(NextState,Rows,Columns)

            #Q-learning
            with torch.no_grad():
                NextQ = torch.max(BasicDQN(nextstate)).item()
                Target = Reward if Done else Reward + DiscountRate * NextQ
            
            
            Predicted = BasicDQN(state)[Action]
            TargetTensor = torch.tensor(Target, dtype=torch.float32)
            Loss = loss_fn(Predicted,TargetTensor)
            Optimizer.zero_grad()
            Loss.backward()
            Optimizer.step()

            #Updating variables
            State = NextState
            EpisodeRewardTotal += Reward
            TotalActions += 1 #1 action taken
        
        #Decaying exploration
        Epsilon = max(0.01,Epsilon * 0.99) #E-greedy end in paper used was 0.01, decay every 8M frames, i can decay every episode

        #End of episode, recording total number of training actions up to this point + reward of this episode
        print("Episode:", i) #Tracking progress
        NumActions.append(TotalActions)
        AverageRewardAtNumOfActions.append(EpisodeRewardTotal)
    
    #Returning the final results of training for the gridworld
    return NumActions,AverageRewardAtNumOfActions

#DQN for the large gridworld, number of episodes increased for longer training as grid world now is size 100x50
def DQNLargeGridWorld():
    #Imports
    import numpy as np
    from GridWorlds.LargeGridworld import LargeGridworld, LargeGridworldReward,GetNextStateLarge #Proceed from here
    import torch
    import torch.nn as nn
    import torch.optim as optim
    loss_fn = nn.MSELoss()

    #Creating DQN model
    LargeWorld = LargeGridworld()
    ActionSize = 4
    StateSize = 2
    Rows = len(LargeWorld)
    Columns = len(LargeWorld[0])
    #Setting up a basic 3 layer neural network with Relu activation from PyTorch.
    BasicDQN = nn.Sequential(
        nn.Linear(StateSize,128), #Batch sizes are downscaled, paper could afford to do 512. Too computationally heavy for me
        nn.ReLU(), #ReLU activation was used in paper
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,ActionSize)
    )

    #Global variables, these are gonna be returned to the main
    NumActions = []
    AverageRewardAtNumOfActions = []
    TotalActions = 0

    #Helper functions, that will be called by the main training loop below, first one encodes the state as a tensor/vector
    def EncodeState(State,Rows,Columns):
        return torch.tensor([State[0]/Rows,State[1]/Columns],dtype=torch.float32)
    
    #Environment step of the agent adapted to the DQN
    def Step(State,Action,Grid):
        NextState = GetNextStateLarge(State,Action)
        #print(NextState)
        Reward = Grid[NextState[0]][NextState[1]]
        ReachedExit = False
        if(NextState[0] == 0 and NextState[1] == (len(Grid[0])-1)): #Next state is terminal state, reached the end
            ReachedExit = True
        return NextState,Reward,ReachedExit
    
    #Now running the main training of the DQN itself on traversing the gridworld
    #Variables
    Episodes = 500 #DQN should hopefully be able to learn the large gridworld in 7500 episodes.
    DiscountRate = 0.997 #Discount rate from research paper
    LearningRate = 1e-4 #Learning rate used in the research paper
    Epsilon = 1
    Actions = [0,1,2,3]
    Optimizer = optim.Adam(BasicDQN.parameters(),lr=LearningRate) #Adam optimizer was used in the research paper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Running 250 episodes of training
    for i in range(Episodes):
        State = [len(LargeWorld)-1,0] #Starting in bottom left
        #print("Starting state:",State)
        #print(len(LargeWorld))
        MaxActionsPerEpisode = (Rows + Columns) * 5 #Ideal solution requires less steps than even this.
        ActionsInEpisode = 0
        EpisodeRewardTotal = 0
        Done = False
        
        #Agent explores the grid world until they reach the exit, this constitutes 1 episode.
        while not Done and (ActionsInEpisode < MaxActionsPerEpisode):
            ActionsInEpisode += 1
            state = EncodeState(State,Rows,Columns)

            #Epsilon greedy, in the research paper there was optional, they tested with and without
            if np.random.rand() <Epsilon:
                Action = np.random.choice(Actions) #Choosing random action to perform/exploring
            else:
                with torch.no_grad():
                    Action = torch.argmax(BasicDQN(state)).item() #Otherwise doing previously best discovered action
            
            #Getting NextState, Reward and if we reached the end based on current state and chosen action
            NextState,Reward,Done = Step(State,Action,LargeWorld)
            nextstate = EncodeState(NextState,Rows,Columns)

            #Q-learning
            with torch.no_grad():
                NextQ = torch.max(BasicDQN(nextstate)).item()
                Target = Reward if Done else Reward + DiscountRate * NextQ
            
            
            Predicted = BasicDQN(state)[Action]
            TargetTensor = torch.tensor(Target, dtype=torch.float32)
            Loss = loss_fn(Predicted,TargetTensor)
            Optimizer.zero_grad()
            Loss.backward()
            Optimizer.step()

            #Updating variables
            State = NextState
            EpisodeRewardTotal += Reward
            TotalActions += 1 #1 action taken
        
        #Decaying exploration
        Epsilon = max(0.01,Epsilon * 0.999) #E-greedy end in paper used was 0.01, decay every 8M frames, i can decay every episode

        #End of episode, recording total number of training actions up to this point + reward of this episode
        print("Episode:", i) #Tracking progress
        NumActions.append(TotalActions)
        AverageRewardAtNumOfActions.append(EpisodeRewardTotal)
    
    #Returning the final results of training for the gridworld
    return NumActions,AverageRewardAtNumOfActions


    


