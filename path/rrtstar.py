import math
import numpy as np

class RRTstar():
    def __init__(self,):
        self.step_len = 0.5
        self.goal_prob = 0.05
        self.search_radius=10
        self.iter_max = 10000
        self.delta = 0.5
        

    def search(self, grid, start, goal, max_iter=2000):
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
                n_index = self.getNNeighbor(state)
                self.OPEN.append(state)
                if n_index:
                    self.chooseParent(state, n_index)
                    self.Rewire(state, n_index)
        index = self.searchGoalParent()
        path = self.extractPath(self.OPEN[index])
        return path
        
    def isCollision(self, start, goal):
        if (self.grid[int(start[0]), int(start[1])]==0) or (self.grid[int(goal[0]), int(goal[1])]==0):
            return True
        
    def getRandomState(self, dx=1, eps=0.05):#dx=0.5
        if np.random.uniform() < eps:
            return self.goal
        else:
            return (np.random.uniform(0+dx, self.H-dx), np.random.uniform(0+dx, self.W-dx))
        
    def getNeighbor(self, state):
        idx = np.argmin([math.hypot(s[0] - state[0], s[1] - state[1]) for s in self.OPEN])
        return self.OPEN[idx]
    
    def getNNeighbor(self, state):
        n = len(self.OPEN) + 1
        r = min(self.search_radius * math.sqrt((math.log(n)/n)), self.step_len)

        dist_table = [math.hypot(s[0] - state[0], s[1] - state[1]) for s in self.OPEN]
        ids = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r and
                                       not self.isCollision(state, self.OPEN[ind])]
        return ids
    
    def getNewState(self, start, goal, step_len=10):
        dist, theta = self.getDistance(start, goal)
        dist = min(step_len, dist)
        state = (start[0] + dist* math.cos(theta), start[1] + dist* math.sin(theta))
        self.PARENT[state] = start
        return state

    def chooseParent(self, state, n_index):
        cost = [self.getNewCost(self.OPEN[i], state) for i in n_index]
        cost_min_index = n_index[int(np.argmin(cost))]
        self.PARENT[state] = self.OPEN[cost_min_index]

    def Rewire(self, state, n_index):
        for i in n_index:
            neighbor = self.OPEN[i]
            if self.getCost(neighbor) > self.getNewCost(state, neighbor):
                self.PARENT[neighbor] = state

    def searchGoalParent(self, step_len=10):
        dist_list = [math.hypot(s[0] - self.goal[0], s[1] - self.goal[1]) for s in self.OPEN]
        n_index = [i for i in range(len(dist_list)) if dist_list[i] <= self.step_len]
        if len(n_index) > 0:
            cost_list = [dist_list[i] + self.getCost(self.OPEN[i]) for i in n_index
                                   if not self.isCollision(self.OPEN[i], self.goal)]
            return n_index[int(np.argmin(cost_list))]
        return len(self.OPEN) - 1

    def getCost(self, state):
        cost = 0.
        while state in self.PARENT.keys():
            parent = self.PARENT[state]
            cost += math.hypot(state[0] - parent[0], state[1] - parent[1])
            state = parent
        return cost
    
    def getNewCost(self, start, goal):
        dist, _ = self.getDistance(start, goal)
        return self.getCost(start) + dist

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