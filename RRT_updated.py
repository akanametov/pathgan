import math
import numpy as np
from kdtrees import KDTree
from scipy.spatial.distance import cdist

class RRT():
    def __init__(self):
        self.OPEN=None
        self.max_iter = 100_000
        self.max_edge_length = 4


    def search(self, grid, start, goal, max_iter=None):
        self.start = start
        self.goal = goal
        self.grid = grid
        self.H, self.W = grid.shape
        self.OPEN =  KDTree.initialize([start])
        self.OPEN_ = [start]
        self.PARENT={}
        self.PARENT[start] = start
        if max_iter:
            self.max_iter = max_iter

        for _ in self:
            rand_state = self.getRandomState()
            nearest_state, distance = self.OPEN.nearest_neighbor(rand_state)[0]
            
            # nearest_state, distance = self.getNearestNeighbor(rand_state)
            nearest_state = tuple(nearest_state)

            state, current_dist = self.getNewState(nearest_state, rand_state, dist=distance)
            if self.IsOnObstacle(state):
                continue

            if not self.isCollision(nearest_state, state, current_dist):
                self.OPEN.insert(state)
                # self.OPEN_.append(state)


                # nearest_to_goal, distance_yo_goal = self.getNearestNeighbor(self.goal)

                nearest_to_goal, distance_to_goal = self.OPEN.nearest_neighbor(self.goal)[0]
                nearest_to_goal = tuple(nearest_to_goal)


                if distance_to_goal < self.max_edge_length and not self.isCollision(nearest_to_goal, self.goal, distance_to_goal):
                    self.PARENT[self.goal] = nearest_to_goal
                    return self.extractPath(self.goal)

                    


    def get_max_edge_length(self):
        return getattr(self, 'max_edge_length', None)

    def set_max_edge_length(self, value):
        setattr(self, 'max_edge_length', value)

    def __iter__(self):
        return iter(range(self.max_iter))
         
    def isCollision(self, start, end , dist):
        for point in self.points_along_line(start, end, dist):
            if self.IsOnObstacle(point):
                return True
        return False

    def IsOnObstacle(self, state):
        if self.grid[int(state[0]), int(state[1])] == 0:
            return True
        return False
        
    def getRandomState(self, dx=1):
        return (np.random.uniform(0, self.H - dx), np.random.uniform(0, self.W - dx))

    def getNewState(self, nearest, r_point, dist):

        d = None if dist > self.max_edge_length else dist
        state  = self.steer(nearest, r_point, dist)
        self.PARENT[state] = nearest

        return state, d
    
    def extractPath(self, state):
        path = [state]
        while True:
            state = self.PARENT[state]
            path.append(state)
            if state == self.start:
                break
        path = [(int(i[0]), int(i[1])) for i in path]
        return path #set(path)

    def getNearestNeighbor(self, state):
        state_arr = np.array(state).reshape((1, -1))
        distances = cdist(state_arr, np.array(self.OPEN_))
        node = self.OPEN_[np.argmin(distances)]
        d = distances.min()
        return node, d

    def points_along_line(self, start, end, dist,  r=1):
        """
        Equally-spaced points along a line defined by start, end, with resolution r
        :param start: starting point
        :param end: ending point
        :param dist: distance between start and  end 
        :param r: maximum distance between points
        :return: yields points along line from start to end, separated by distance r
        """

        d = self.max_edge_length if dist is None else dist

        n_points = int(np.ceil(d / r))
        if n_points > 1:
            step = d / (n_points - 1)
            for i in range(1, n_points):
                next_point = self.steer(start, end, i * step)
                yield next_point

    def steer(self, start_p, end_p, d):
        """
        Return a point in the direction of the goal, that is distance away from start
        :param start_p: start location
        :param end_p: end location
        :param d: distance away from start
        :return: point in the direction of the end, distance away from start
        """

        v = np.subtract(end_p, start_p)
        u = v / np.linalg.norm(v)
        steered_point = start_p + u * self.max_edge_length if d is None else start_p + u * d
        return tuple(steered_point)