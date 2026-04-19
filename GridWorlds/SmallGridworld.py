#Alex Eliseev
#501093338
#All 3 of the gridworlds used in the experiment and testing the various RL models are going to be pre-hardcoded with start in bottom left and exit in bottom right.
#The main idea is they will have an obvious to an outside observer real solution for what the optimal pathing across the world is.
#Each one of the grid worlds will have certain quirks to them, like for example a portal that takes the agent across the map skipping most of it
#Swamp locations that have - rewards for travelling through but with treassure chests inside that can be claimed once but offset any losses
#for travelling through the swamp itself e.t.c. I will then compare what the ideal solution vs the solution found by the model is.
#The first gridworld is a simple small 7x7 gridworld with just 49 tiles.
#Actions will be Up = 1, Right = 2, Down = 3, Left = 4,

#Function simply returns the gridworld itself.
def SmallGridworld():
    #Small girdworld has a massive treassure in bottom right corner
    #Ideal pathing is to rush bottom right corner and head to the exit, (75 + 25 - 1)/12 
    #Option to go through middle, (1+0+3+0-1+10+5+0+1+1+25)/12 = 45/12 =3.75
    #Option to take the road along the left and top side (37)/12 = 3.08
    #There is alternative option to go bottom right and then grab the 10 before heading to top this nets ~6 reward on average.
    Small = [
        [1,1,1,1,1,1,25],
        [1,0,0,0,0,0,0],
        [1,0,0,0,5,0,0],
        [1,0,-1,10,-1,0,0],
        [1,0,0,-1,0,0,0],
        [1,0,3,0,0,0,0],
        [0,-1,0,0,0,0,75]
    ]
    #Returning the small gridworld.
    return Small

#Reward function from travelling across the small gridworld, reward function will vary between the worlds.
def SmallGridworldReward(State,Action,SmallGrid):
    NextState = GetNextStateSmall(State,Action) #Getting next state from taking the action
    Reward = SmallGrid[NextState[0]][NextState[1]]
    return Reward

#Next state function for small gridworld, returns the next expected state if an action is taken at the current state.
def GetNextStateSmall(State,Action):
    #First checking if the agent would leave the world in which case,restricting it and causing it to bump into the wall.
    if(State[1] == 0 and Action == 3):
        return State #At the left wall, cant go left
    elif(State[1] == 6 and Action == 1):
        return State #At the right wall, cant go right
    elif(State[0] == 0 and Action == 0):
        return State #At the top wall, cant go up
    elif(State[0] == 6 and Action == 2): 
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
    