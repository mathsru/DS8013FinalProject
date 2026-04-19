#Alex Eliseev
#501093338
#All 3 of the gridworlds used in the experiment and testing the various RL models are going to be pre-hardcoded.
#The main idea is they will have an obvious to an outside observer real solution for what the optimal pathing across the world is.
#Each one of the grid worlds will have certain quirks to them, like for example a portal that takes the agent across the map skipping most of it
#Swamp locations that have - rewards for travelling through but with treassure chests inside that can be claimed once but offset any losses
#for travelling through the swamp itself e.t.c. I will then compare what the ideal solution vs the solution found by the model is.
#The second gridworld is the middle sized grid world of size 25x25 and has more features than the small along with a unique rule function to it

#Function simply returns the gridworld itself.
def MediumGridworld():
    #Medium world of 25x25 size. Once again has a road going across the world that can be taken.
    #There is a swamp in the center left, with a large reward in it surounded by negatives. Going for the 150 is absolutely worth it.
    #There is a 50 that can be grabbed in the top left, but the detour would not be worth the time.
    #One more swamp in the bottom right with a large reward
    Medium = [
        [50,-5,-3,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25],
        [-5,-5,-3,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [-3,-3,-3,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [-1,-1,-1,-1,0,0,0,-2,-2,-2,-2,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,-2,50,-2,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,-2,-2,-2,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,-1,-1,-1,1,-1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,-1,-2,-2,-2,-1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,-1,-2,-3,-3,-2,-1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,-1,-2,150,-3,-2,-1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,-1,-2,-3,-3,-3,-2,-1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,-1,-2,-2,-2,-2,-2,-1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,-1,-1,-1,-1,-1,-1,-1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,30,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,-3,-3,-3],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-3,-5,-5],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-3,-5,150]
    ]
    #Returning the medium gridworld.
    return Medium

#Reward function from travelling across the small gridworld, reward function will vary between the worlds.
def MediumGridworldReward(State,Action,Grid):
    NextState = GetNextStateMedium(State,Action) #Getting next state from taking the action
    Reward = Grid[NextState[0]][NextState[1]]
    return Reward

#Next state function for small gridworld, returns the next expected state if an action is taken at the current state.
def GetNextStateMedium(State,Action):
    #First checking if the agent would leave the world in which case,restricting it and causing it to bump into the wall.
    if(State[1] == 0 and Action == 3):
        return State #At the left wall, cant go left
    elif(State[1] == 24 and Action == 1):
        return State #At the right wall, cant go right
    elif(State[0] == 0 and Action == 0):
        return State #At the top wall, cant go up
    elif(State[0] == 24 and Action == 2): 
        return State #At the bottom wall and cant go down
    else:
        #Otherwise legal move
        if(Action == 0):
            return [State[0] - 1,State[1]] #Moving 1 up
        elif(Action == 1):
            return [State[0],State[1] + 1] #Moving 1 right
        elif(Action == 2):
            return [State[0] + 1,State[1]] #Moving 1 down
        else:
            return[State[0],State[1] -1] #Moving 1 left
