import math
import numpy as np

class RRT():
    '''
    RRT algorithm

    Args:
        - None -
        
    '''
    def __init__(self,):
        self.step_len = 0.5
        self.goal_prob = 0.05
        self.delta = 0.5
        
    def run(self, grid, start, goal, max_iter=100000):
        '''
        Running RRT algorithm

        Parameters:
            start (tuple(int, int)): Start position 
            goal (tuple(int, int)): Goal position
            max_iter (default: int=100 000): Maximum number of RRT runs
            
        Returns:
            path (list(tuple, ..., tuple)): Path found by RRT

        '''
        self.start = start
        self.goal = goal
        self.grid = grid
        self.H = len(grid)
        self.W = len(grid)
        self.OPEN=[start]
        self.PARENT={}
        self.PARENT[start] = start
        for i in range(max_iter):
            r_state = self.getRandomState()
            n_state = self.getNeighbor(r_state)
            state = self.getNewState(n_state, r_state)
            if state and not self.isCollision(n_state, state):
                self.OPEN.append(state)
                dist, _ = self.getDistance(state, goal)
                if dist <= self.step_len and not self.isCollision(state, goal):
                    return self.extractPath(state)
                
    def isCollision(self, start, goal):
        '''
        Checking if there is collision

        Parameters:
            start (tuple(int, int)): Start position 
            goal (tuple(int, int)): Goal position
            
        Returns:
            True: if collision occured

        '''
        if (self.grid[int(start[0]), int(start[1])]==0) or (self.grid[int(goal[0]), int(goal[1])]==0):
            return True
        
    def getRandomState(self, dx=1, eps=0.05):
        '''
        Get random point in 2d map

        Parameters:
            dx (default: float=1): Step size 
            eps (default: float=0.05): Probability with which goal position will be returned
            
        Returns:
            state (tuple(int, int)): Position

        '''
        if np.random.uniform() < eps:
            return self.goal
        else:
            return (np.random.uniform(0+dx, self.H-dx), np.random.uniform(0+dx, self.W-dx))
    
    def getNeighbor(self, state):
        '''
        Get neighbor point of randomly choosed state

        Parameters:
            state (tuple(int, int)): Randomly choosed state 
            
        Returns:
            state (tuple(int, int)): Nearest neighbor point of randomly choosed state

        '''
        idx = np.argmin([math.hypot(s[0] - state[0], s[1] - state[1]) for s in self.OPEN])
        return self.OPEN[idx]
    
    def getNewState(self, start, goal, step_len=0.5):
        '''
        Get new state

        Parameters:
            step_len (default: float=0.5): Length of step 
            
        Returns:
            state (tuple(int, int)): New state

        '''
        dist, theta = self.getDistance(start, goal)
        dist = min(step_len, dist)
        state = (start[0] + dist* math.cos(theta), start[1] + dist* math.sin(theta))
        self.PARENT[state] = start
        return state
    
    def extractPath(self, state):
        '''
        Get path from final state

        Parameters:
            state (tuple(int, int)): Last state 
            
        Returns:
            path (list(tuple, ..., tuple)): Path extracted from last state (found by RRT)

        '''
        path = [state]
        while True:
            state = self.PARENT[state]
            path.append(state)
            if state == self.start:
                break
        path = [(int(i[0]), int(i[1])) for i in path]
        return set(path)

    def getDistance(self, start, goal):
        '''
        Euclidian distance between points

        Parameters:
            start (tuple(int, int)): State 1
            goal (tuple(int, int)): State 2
            
        Returns:
            dist (float): Distance between points 

        '''
        dist  = math.hypot(goal[0] - start[0], goal[1] - start[1])
        theta = math.atan2(goal[1] - start[1], goal[0] - start[0])
        return dist, theta
