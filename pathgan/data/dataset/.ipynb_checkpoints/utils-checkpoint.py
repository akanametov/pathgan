import os
import math
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar


class RRT():
    """RRT."""

    step_len = 0.5
    goal_prob = 0.05
    delta = 0.5

    def __init__(self):
        pass

    def run(self, grid, start, goal, max_iter: int = 100000):
        """Run RRT algorithm.

        Parameters
        ----------
        start: Tuple[int, int]
            Start position.
        goal: Tuple[int, int]
            Goal position.
        max_iter: int, optional (default=100000)
            Maximum number of RRT iterations.

        Returns
        -------
        List[Tuple]:
            Path found by RRT.
        """
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
        """Check if there is collision.

        Parameters
        ----------
        start: Tuple[int, int]
            Start position. 
        goal: Tuple[int, int]
            Goal position.

        Returns
        -------
        bool:
            If collision occured.

        """
        if (self.grid[int(start[0]), int(start[1])]==0) or (self.grid[int(goal[0]), int(goal[1])]==0):
            return True

    def getRandomState(self, dx: float = 1.0, eps: float = 0.05):
        """Get random point in 2d map

        Parameters
        ----------
        dx: float, optional (default=1.0)
            Step size.
        eps: float, optional (default=0.05)
            Probability with which goal position will be returned.

        Returns
        -------
        Tuple[int, int]
            Position.
        """
        if np.random.uniform() < eps:
            return self.goal
        else:
            return (np.random.uniform(0+dx, self.H-dx), np.random.uniform(0+dx, self.W-dx))

    def getNeighbor(self, state):
        """Get neighbor point of randomly choosed state.

        Parameters
        ----------
        state: Tuple[int, int]
            Randomly choosed state. 

        Returns
        -------
        Tuple[int, int]
            Nearest neighbor point of randomly choosed state.
        """
        idx = np.argmin([math.hypot(s[0] - state[0], s[1] - state[1]) for s in self.OPEN])
        return self.OPEN[idx]

    def getNewState(self, start, goal, step_len=0.5):
        """Get new state.

        Parameters
        ----------
        step_len: float, optional (default=0.5)
            Length of step.

        Returns
        -------
        Tuple[int, int]
            New state.
        """
        dist, theta = self.getDistance(start, goal)
        dist = min(step_len, dist)
        state = (start[0] + dist* math.cos(theta), start[1] + dist* math.sin(theta))
        self.PARENT[state] = start
        return state

    def extractPath(self, state):
        """Get path from final state.

        Parameters
        ----------
        state: Tuple[int, int]
            Last state.

        Returns
        -------
        List[Tuple]
            Path extracted from last state (found by RRT).
        """
        path = [state]
        while True:
            state = self.PARENT[state]
            path.append(state)
            if state == self.start:
                break
        path = [(int(i[0]), int(i[1])) for i in path]
        return set(path)

    def getDistance(self, start, goal):
        """Euclidian distance between points

        Parameters
        ----------
        start: Tuple[int, int]
            State 1.
        goal: Tuple[int, int]
            State 2.

        Returns
        -------
        float:
            Distance between points.
        """
        dist  = math.hypot(goal[0] - start[0], goal[1] - start[1])
        theta = math.atan2(goal[1] - start[1], goal[0] - start[0])
        return dist, theta


def rgb2binary(img):
    return (img[..., :] > 150.0).astype(float)


class MapAugmentator():
    """Map Augmentator."""

    def __init__(self):
        pass

    def set_parameters(
        self,
        height_shift: int = 2,
        width_shift: int = 2,
        shift_step: int = 1,
        rot_prob: float = 0.5,
        n_maps: int = 10,
        load_dir: str = "dataset/init_maps",
        save_dir: str = "dataset/maps",
    ):
        """Set parameters of augmentation.

        Parameters
        ----------
        height_shift: int, optional (default=2)
            Range of vertical shift.
        width_shift: int, optional (default=2)
            Range of horizontal shift.
        shift_step: int, optional (default=1)
            Step of shift.
        rot_prob: int, optional (default=0.5)
            Probability of map to be rotated.
        n_maps: int, optional (default=10)
            Number of map which will be obtained for each augmentation map.
        load_dir: int, optional (default="data/dataset/init_maps")
            Path where initial augmentation maps are located.
        save_dir: int, optional (default="data/dataset/maps")
            Path where generated (augmneted) maps will be saved.
        """
        self.h_shift = height_shift
        self.w_shift = width_shift
        self.h_range = np.arange(- height_shift, height_shift, shift_step)
        self.w_range = np.arange(- width_shift, width_shift, shift_step)
        self.step = shift_step
        self.t_prob = rot_prob
        self.n_maps = n_maps
        self.load_dir = load_dir
        self.save_dir = save_dir
        
    def augment(self, map_name: str):
        """Run augmentation.

        Parameters
        ----------
        map_name: str
            Name of map.
        """
        self.map_name = map_name.split(".")[0]
        map_img = Image.open(f"{self.load_dir}/{map_name}")
        map_data = rgb2binary(np.array(map_img))[..., 0]

        maps = []
        H, W = map_data.shape
        nH = H + 2*self.h_shift
        nW = W + 2*self.w_shift
        bar = IncrementalBar(f"{map_name}:", max=self.n_maps)
        for n in range(self.n_maps):
            grid = np.ones((nH, nW))
            
            h = np.random.choice(self.h_range, 1).item()
            w = np.random.choice(self.w_range, 1).item()
            
            grid[self.h_shift + self.step*h: H + self.h_shift + self.step*h,
                 self.w_shift + self.step*w: W + self.w_shift + self.step*w] = map_data
            g_map = grid[self.h_shift: - self.h_shift, self.w_shift: - self.w_shift].copy()
            if np.random.uniform() < self.t_prob:
                g_map = g_map.T
            assert g_map.shape == (H, W)
            maps.append(g_map)
            bar.next()
        bar.finish()
        self.aug_maps=maps
    
    def save(self):
        """Saving augmented maps."""
        if not os.path.exists(f"{self.save_dir}"):
            os.makedirs(f"{self.save_dir}", exist_ok=True)
        for i, aug_map in enumerate(self.aug_maps):
            save_path = f"{self.save_dir}/{self.map_name}{i}.png"
            plt.imsave(save_path, aug_map, cmap="gray")

    
class TaskGenerator():
    """Tasks generator."""

    def __init__(self,):
        pass
        
    def set_parameters(
        self,
        min_length: int = 30.0,
        n_tasks: int = 100,
        load_dir: str = "dataset/maps",
        save_dir: str = "dataset/tasks",
    ):
        """Set parameters of task generation.

        Parameters
        ----------
        min_length: float, optional (default=30.0)
            Minimal length between start and goal points. 
        n_tasks: int, optional (default=100)
            Number of tasks which will be obtained for each map.
        load_dir: str, optional (default="dataset/maps")
            Path where maps are located.
        save_dir: str, optional (default="dataset/tasks")
            Path where generated tasks will be saved.
        """
        self.min_length = min_length
        self.n_tasks = n_tasks
        self.load_dir = load_dir
        self.save_dir = save_dir

    def euclid(self, i1, j1, i2, j2):
        """Function to calculate Euclidian distance."""
        return math.sqrt((i1 - i2)**2 + (j1 - j2)**2)

    def generate(self, map_name: str):
        """Run augmentation.

        Parameters
        ----------
        map_name: str
            Name of map.
        """
        self.map_name = map_name.split('.')[0]
        map_img = Image.open(f'{self.load_dir}/{map_name}')
        map_data = rgb2binary(np.array(map_img))[..., 0]
        start_color = np.array([0, 0, 1])
        goal_color = np.array([1, 0, 0])
        tasks_data = {'istart': [], 'jstart': [],
                      'igoal' : [], 'jgoal' : [],
                      'euclid': []}
        task_maps=[]
        H, W = map_data.shape
        t=0
        bar = IncrementalBar(f'{map_name}:', max=self.n_tasks)
        while t < self.n_tasks:
            istart = np.random.choice(H, 1).item()
            jstart = np.random.choice(W, 1).item()
            igoal = np.random.choice(H, 1).item()
            jgoal = np.random.choice(W, 1).item()
            traversable = (map_data[istart, jstart] != 0) and (map_data[igoal, jgoal] != 0)
            dist = self.euclid(istart, jstart, igoal, jgoal)
            goodlength = (dist > self.min_length)
            if traversable and goodlength:
                tasks_data['istart'].append(int(istart))
                tasks_data['jstart'].append(int(jstart))
                tasks_data['igoal'].append(int(igoal))
                tasks_data['jgoal'].append(int(jgoal))
                tasks_data['euclid'].append(float(dist))
                task_map = np.ones((H, W, 3))
                task_map[istart, jstart] = start_color
                task_map[igoal, jgoal] = goal_color
                task_maps.append(task_map)
                t += 1
                bar.next()
        bar.finish()
        self.tasks_data = tasks_data
        self.task_maps = task_maps

    def save(self,):
        """Saving generated tasks."""
        if not os.path.exists(f"{self.save_dir}"):
            os.makedirs(f"{self.save_dir}", exist_ok=True)
        csv_file = pd.DataFrame.from_dict(self.tasks_data)
        fname = f"{self.save_dir}/{self.map_name}.csv"
        csv_file.to_csv(fname, index=False)
        if not os.path.exists(f"{self.save_dir}/{self.map_name}"):
            os.makedirs(f"{self.save_dir}/{self.map_name}", exist_ok=True)
        for i, task_map in enumerate(self.task_maps):
            save_path = f"{self.save_dir}/{self.map_name}/task_{i}.png"
            plt.imsave(save_path, task_map)


class ROIGenerator():
    """ROI generator."""

    def __init__(self):
        pass
    
    def set_parameters(
        self,
        algorithm,
        n_runs: int = 50,
        map_dir: str = "dataset/maps",
        task_dir: str = "dataset/tasks",
        save_dir: str = "dataset/tasks",
    ):
        """Set parameters of ROI generation.

        Parameters
        ----------
        algorithm
            Object of sampling-based pathfinding algorithm.
        n_runs: int, optional (default=50)
            Number of times pathfinding algorithm will be running on each task.
        map_dir: str, optional (default="dataset/maps")
            Path where maps are located.
        task_dir: str, optional (default="dataset/tasks")
            Path where tasks are located.
        save_dir: str, optional (default="dataset/tasks")
            Path where generated ROIs will be saved.
        """
        self.algorithm = algorithm
        self.n_runs = n_runs
        self.map_dir = map_dir
        self.task_dir = task_dir
        self.save_dir = save_dir
        
    def generate(self, map_name: str):
        """Run generation.

        Parameters
        ----------
        map_name: str
            Name of map.
        """
        self.map_name = map_name.split(".")[0]
        if not os.path.exists(f"{self.save_dir}"):
            os.makedirs(f"{self.save_dir}", exist_ok=True)
        if not os.path.exists(f"{self.save_dir}/{self.map_name}"):
            os.makedirs(f"{self.save_dir}/{self.map_name}", exist_ok=True)
        map_img = Image.open(f"{self.map_dir}/{map_name}").convert("RGB")
        map_data = rgb2binary(np.array(map_img))
        csv_file = pd.read_csv(f"{self.task_dir}/{self.map_name}.csv")

        colors = {
            "roi": np.array([0, 1, 0]),
            "start": np.array([0, 0, 1]),
            "goal": np.array([1, 0, 0]),
        }
        grids=[]
        rois=[]
        bar = IncrementalBar(f"{map_name}:", max=len(csv_file))
        for i, row in csv_file.iterrows():
            start = (int(row.istart), int(row.jstart))
            goal = (int(row.igoal), int(row.jgoal))
            grid = map_data.copy()
            roi = np.ones(grid.shape)
            for _ in range(self.n_runs):
                path = self.algorithm.run(map_data[..., 1], start, goal)
                if path:
                    for x in path:
                        grid[x[0], x[1], :] = colors["roi"]
                        roi[x[0], x[1], :] = colors["roi"]
            grid[start[0], start[1], :] = colors["start"]
            grid[goal[0], goal[1], :] = colors["goal"]
            res_path = f"{self.save_dir}/{self.map_name}/task_{i}_rrt.png"
            roi_path = f"{self.save_dir}/{self.map_name}/task_{i}_roi.png"
            plt.imsave(res_path, grid)
            plt.imsave(roi_path, roi)
            bar.next()
        bar.finish()
