import math
import numpy as np

class RRT():
    def __init__(self,):
        self.step_len = 0.5
        self.goal_prob = 0.05
        self.delta = 0.5
        
    def run(self, grid, start, goal, max_iter=100000):
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
        if (self.grid[int(start[0]), int(start[1])]==0) or (self.grid[int(goal[0]), int(goal[1])]==0):
            return True
        
    def getRandomState(self, dx=1, eps=0.05):
        if np.random.uniform() < eps:
            return self.goal
        else:
            return (np.random.uniform(0+dx, self.H-dx), np.random.uniform(0+dx, self.W-dx))
    
    def getNeighbor(self, state):
        idx = np.argmin([math.hypot(s[0] - state[0], s[1] - state[1]) for s in self.OPEN])
        return self.OPEN[idx]
    
    def getNewState(self, start, goal, step_len=0.5):
        dist, theta = self.getDistance(start, goal)
        dist = min(step_len, dist)
        state = (start[0] + dist* math.cos(theta), start[1] + dist* math.sin(theta))
        self.PARENT[state] = start
        return state
    
    def extractPath(self, state):
        path = [state]
        while True:
            state = self.PARENT[state]
            path.append(state)
            if state == self.start:
                break
        path = [(int(i[0]), int(i[1])) for i in path]
        return set(path)

    def getDistance(self, start, goal):
        dist  = math.hypot(goal[0] - start[0], goal[1] - start[1])
        theta = math.atan2(goal[1] - start[1], goal[0] - start[0])
        return dist, theta
