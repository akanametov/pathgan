from rtree import index
import numpy as np
from operator import itemgetter
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
from skimage import draw
from copy import deepcopy
import matplotlib.animation as animation

from PIL import Image, ImageDraw
warnings.filterwarnings('ignore')

init_color = np.array([0, 0, 1])
goal_color = np.array([1, 0, 0])
roi_color = np.array([0, 1, 0])
best_path_color = np.array([1, 0, 1])
paths_history = []

def rgb2binary(img):
    return (img[..., :] > 150).astype(float)

load_dir = 'data/dataset'
map_dir = 'maps/map_99.png'
task = 'tasks/map_99.csv'

with open(Path(load_dir)/task) as f:
    for _ in range(4):
        next(f)
    task = f.readline().rstrip().split(',')
    euclid = float(task[-1])
    iinit, jinit, igoal, jgoal = list(map(int, task[:-1]))
    xy_init = (iinit, jinit)
    xy_goal = (igoal, jgoal)


map_img = Image.open(f'{load_dir}/{map_dir}').convert('RGB')
map_data = rgb2binary(np.array(map_img))


def draw_edge(xy_init, xy_goal, map_data, color=roi_color):
    xx, yy = draw.line_nd(xy_init, xy_goal, endpoint=False)
    map_data[xx, yy] = color
    

def draw_best_path(path, map_data, color=best_path_color):
    for xy1, xy2 in zip(path[:-1], path[1:]):
        draw_edge(xy1, xy2, map_data, color)
    

        
class RRTStar(object):
    def __init__(self, path_resolution, gamma=1, step_len=1, max_iter=1, mu=0.1, index_dim=2):
        self.path_resolution = path_resolution
        self.gamma = gamma # planning constant
        self.step_len = step_len
        self.max_iter = max_iter
        self.mu = mu
        self.best_cost = float('inf')
        self.best_path = []
        self.it_to_best = 0
        self.samples_taken = 0
        self.index_dim = index_dim
        self.V = self.create_nodes_index(index_dim)
        self.E = {}
        
    
    def create_nodes_index(self, index_dim):
        p = index.Property()
        p.dimension = index_dim
        V = index.Index(interleaved=True, properties=p)
        return V
    
    
    def reinitialize(self):
        self.best_cost = float('inf')
        self.samples_taken = 0
        self.V = self.create_nodes_index(self.index_dim)
        self.E = {}
        self.best_path = []
        self.it_to_best = 0
        
        
    def add_node(self, xy, idx=0):
        if self.V.count(xy) == 0:
            self.V.insert(idx, xy + xy, xy)
            self.samples_taken += 1
        
    
    def add_edge(self, parent_xy, child_xy):
        self.E[child_xy] = parent_xy
        
        
    def search(self, grid_map, xy_init, xy_goal, roi=None, max_iter=None):
        xy_max = grid_map.shape[:2]
        self.reinitialize()
        
        if max_iter is not None:
            self.max_iter = max_iter
            
        self.V.insert(0, xy_init + xy_init, xy_init)
        self.add_edge(None, xy_init)
        
        for t in range(self.max_iter):
            xy_rand = self.sample_free(grid_map, roi)
            xy_near = next(self.nearest(xy_rand, n=1))
            xy_new = self.steer(xy_near, xy_rand, self.step_len, xy_max)
            
            # not self.V.count(xy_new) == 0 - use or not
            if not self.is_cell_free(xy_new, grid_map):
                continue
            
            if self.V.count(xy_new) == 0:
                self.samples_taken += 1

            costs_near = self.choose_parent(xy_init, xy_near, xy_new, xy_goal, grid_map)
            if xy_new in self.E:
                self.rewire(xy_new, costs_near, grid_map)
            
            path, cost = self.get_path_and_cost(xy_init, xy_goal, grid_map)
            if path is not None and cost < self.best_cost:
                self.best_cost = cost
                self.best_path = path
                self.it_to_best = t + 1
                print('Found new path {} with cost {:.6f}'.format(path, cost))
        return None
            
            
    def sample_free(self, grid_map=None, roi=None):
        if roi is not None and np.random.random() > self.mu:
            return self.non_uniform_sample(roi)
        return self.uniform_sample(grid_map)
    
    
    def uniform_sample(self, grid_map):
        if len(grid_map.shape) > 2:
            h, w = grid_map.shape[:2]
        else:
            h, w = grid_map.shape
            
        while True:
            x = np.random.randint(0, h)
            y = np.random.randint(0, w)
            if self.is_cell_free((x, y), grid_map):
                return x, y
        
        
    def is_cell_free(self, xy, grid_map):
        x, y = xy
        return sum(grid_map[x, y]) != 0
    
    
    def non_uniform_sample(self, roi):
        pass
    
    
    def nearest(self, xy, n=1):
        n = min(n, self.V.get_size())
        return self.V.nearest(xy, num_results=n, objects='raw')
    
    
    def set_bounds(self, xy, xy_max):
        x, y = xy
        x_max, y_max = xy_max
        x = min(max(0, x), x_max - 1)
        y = min(max(0, y), y_max - 1)
        return int(x), int(y)
    
    
    def steer(self, xy1, xy2, step, xy_max):
        xy = np.array(xy2) - np.array(xy1)
        d = self.euclidean_distance(xy1, xy2)
        try:
            # cos and sin
            u = xy / d
        except:
            u = xy / (d + 1e-9)
        step = min(step, d)
        steered_point = np.array(xy1) + u * step
        bounded = self.set_bounds(tuple(steered_point), xy_max)
        return bounded


    def obstacle_free(self, xy1, xy2, grid_map):
        if len(grid_map.shape) > 2:
            xy_max = grid_map.shape[:2]
        else:
            xy_max = grid_map.shape[:2]
            
        d = self.euclidean_distance(xy1, xy2)
        n_pts = int(np.ceil(d / self.path_resolution))
        if n_pts > 1:
            step_size = d / n_pts
            for i in range(n_pts + 1):
                next_point = self.steer(xy1, xy2, i * step_size, xy_max)
                if not self.is_cell_free(next_point, grid_map):
                    break
            else:
                return True
        else:
            if xy1 == xy2 and self.is_cell_free(xy1, grid_map):
                return True
        return False
        
        
        
    def euclidean_distance(self, xy1, xy2):
        d = np.array(xy2) - np.array(xy1)
        return np.linalg.norm(d)
    
    
    def get_near_costs(self, xy_init, xy_near, near_nodes, xy_new):
        n = self.V.get_size()
        search_radius = self.gamma * np.sqrt(np.log(n) / n)
        costs_near = []
        for xy_near in near_nodes:
            if self.euclidean_distance(xy_near, xy_new) > search_radius:
                break
            costs_near.append((self.compute_cost(xy_init, xy_near) + self.euclidean_distance(xy_near, xy_new), xy_near))
            
        if len(costs_near) == 0:
            costs_near = [(self.compute_cost(xy_init, xy_near) + self.euclidean_distance(xy_near, xy_new), xy_near)]
        costs_near.sort(key=itemgetter(0))

        return costs_near
    
    
    def choose_parent(self, xy_init, xy_near, xy_new, xy_goal, grid_map):
        n = self.V.get_size()
        near_nodes = iter(self.nearest(xy_new, n))
        costs_near = self.get_near_costs(xy_init, xy_near, near_nodes, xy_new)
    
        for cost, xy_near in costs_near: 
            path_cost = cost + self.euclidean_distance(xy_new, xy_goal)
            if path_cost < self.best_cost and self.can_connect(xy_near, xy_new, grid_map):
                break
        return costs_near
            
    
    def can_connect(self, xy_near, xy_new, grid_map):
        if self.obstacle_free(xy_near, xy_new, grid_map):
            self.add_node(xy_new)
            self.add_edge(xy_near, xy_new)
            draw_edge(xy_near, xy_new, grid_map)
            grid_map[iinit, jinit] = init_color
            grid_map[igoal, jgoal] = goal_color
            paths_history.append((deepcopy(grid_map), deepcopy(self.best_path), self.best_cost))
            return True
        return False
    
    
    def rewire(self, xy_new, costs_near, grid_map):
        for cost, xy_near in costs_near:
            cur_cost = self.compute_cost(xy_init, xy_near)
            new_cost = self.compute_cost(xy_init, xy_new) + self.euclidean_distance(xy_new, xy_near)
            if new_cost < cur_cost and self.obstacle_free(xy_near, xy_new, grid_map):
                self.add_edge(xy_new, xy_near)
                draw_edge(xy_new, xy_near, grid_map)
                grid_map[iinit, jinit] = init_color
                grid_map[igoal, jgoal] = goal_color
                paths_history.append((deepcopy(grid_map), deepcopy(self.best_path), self.best_cost))

    
    def compute_cost(self, xy1, xy2, return_path=False):
        cost = 0.
        path = [xy2]
        while xy2 is not None and xy2 != xy1:
            parent = self.E[xy2]
            cost += self.euclidean_distance(parent, xy2)
            xy2 = parent
            path.append(parent)
        path.reverse()
        
        if return_path:
            return cost, path
        return cost
    
    
    def get_path_and_cost(self, xy_init, xy_goal, grid_map):
        flag, nearest_to_goal = self.is_goal_reachable(xy_goal, grid_map)
        if flag and nearest_to_goal is not None:
            self.add_edge(nearest_to_goal, xy_goal)
            draw_edge(nearest_to_goal, xy_goal, grid_map)
            grid_map[iinit, jinit] = init_color
            grid_map[igoal, jgoal] = goal_color
            paths_history.append((deepcopy(grid_map), deepcopy(self.best_path), self.best_cost))
            cost, path = self.compute_cost(xy_init, xy_goal, return_path=True)
            return path, cost
        return None, None
    
    
    def is_goal_reachable(self, xy_goal, grid_map):
        xy_near = next(self.nearest(xy_goal))
        if xy_goal in self.E and xy_near == self.E[xy_goal]:
            return True, xy_near
        if self.obstacle_free(xy_near, xy_goal, grid_map):
            return True, xy_near
        return False, None


rrt_star = RRTStar(path_resolution=1, gamma=10, step_len=4, max_iter=10000, mu=0.1, index_dim=2)
rrt_star.search(map_data, xy_init, xy_goal)
print('Samples taken {}'.format(rrt_star.samples_taken))
print('Euclidean distance {:.4f}, best cost {:.4f} (reached in #{} iterations)'.format(euclid, rrt_star.best_cost, rrt_star.it_to_best))

draw_best_path(rrt_star.best_path, map_data)
map_data[iinit, jinit] = init_color
map_data[igoal, jgoal] = goal_color
plt.imshow(map_data)
plt.show()

paths_history.append((deepcopy(map_data), deepcopy(rrt_star.best_path), rrt_star.best_cost))

fig = plt.figure( figsize=(8,8) )
im = plt.imshow(paths_history[0][0])

def init():
    im.set_data(np.zeros_like(map_data))
    return im,

def animate(i):
    im.set_array(np.array(paths_history[i][0]))
    return im,


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(paths_history), interval=20, blit=True)
#anim.save('rrt_star.gif', writer='imagemagick')
anim.save('test_anim.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
