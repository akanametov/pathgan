"""RRT Base and PathDescription."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional, Any, Generator

import random
import numpy as np
from rtree import index
from collections import OrderedDict

import warnings
warnings.filterwarnings('ignore')


class PathDescription(object):
    """PathDescription.

    Parameters
    ----------
    path: List
        List of points.
    cost: float, optional (default=float('inf'))
        Cost.
    time_sec: float, optional (default=0.0)
        Time in seconds.
    time_it: int, optional (default=0)
        Time on iteration.
    samples_taken: int, optional (default=0)
        Samples taken.
    nodes_taken: int, optional (default=0)
        Nodes taken.
    """
    def __init__(
        self,
        path: Optional[List] = None,
        cost: float = float('inf'),
        time_sec: float = 0.,
        time_it: int = 0,
        samples_taken: int = 0,
        nodes_taken: int = 0,
    ):
        """Initialize."""
        self.path = path if path is not None else []
        self.cost = cost
        self.time_sec = time_sec
        self.time_it = time_it
        self.samples_taken = samples_taken
        self.nodes_taken = nodes_taken

    def update(
        self,
        path: List[int],
        cost: float,
        time_sec: float,
        time_it: int,
        samples_taken: int,
        nodes_taken: int,
    ):
        """Update description.

        Parameters
        ----------
        path: List
            List of points.
        cost: float, optional (default=float('inf'))
            Cost.
        time_sec: float, optional (default=0.0)
            Time in seconds.
        time_it: int, optional (default=0)
            Time on iteration.
        samples_taken: int, optional (default=0)
            Samples taken.
        nodes_taken: int, optional (default=0)
            Nodes taken.
        """
        self.__init__(path, cost, time_sec, time_it, samples_taken, nodes_taken)

    def __lt__(self, other):
        """Check if new cost is smaller."""
        return self.cost < other.cost

    def __call__(self) -> dict:
        """Get parameters."""
        arguments = vars(self)
        return {name: value for name, value in arguments.items()}


class RRTBase(ABC):
    """RRT Base.

    Parameters
    ----------
    grid_map: np.ndarray
        Grid map.
    xy_init: Tuple[int, int]
        Start point coordinates.
    xy_goal: Tuple[int, int]
        Goal point coordinates.
    roi: List[tuple], optional (default=None)
        RoI for grid map.
    path_resolution: int, optional (default=1)
        Resolution of path.
    step_len: float, optional (default=0.5)
        Length of step to be made.
    max_iter: int, optional (default=10000)
        Number of maximum iterations.
    mu: float, optional (default=0.1)
        Mu value.
    seed: int, optional (default=42)
        Seed value.
    dist_init_goal: float, optional (default=None)
        Probability of selecting goal point.
    """
    def __init__(
        self,
        grid_map: np.ndarray,
        xy_init: Tuple[int, int],
        xy_goal: Tuple[int, int],
        roi: Optional[List[Tuple[int, int]]] = None,
        path_resolution: int = 1,
        step_len: float = 0.5,
        max_iter: int = 10000,
        gamma: float = 1,
        mu: float = 0.1,
        seed: int = 42,
        dist_init_goal: Optional[float] = None,
        **kwargs,
    ):
        """Initialize."""
        self.path_resolution = path_resolution
        self.step_len = step_len
        self.max_iter = max_iter
        self.index_dim = 2
        self.gamma = gamma
        self.mu = mu

        self.grid_map = grid_map
        self.xy_init = xy_init
        self.xy_goal = xy_goal
        self.roi = roi
        self.xy_max = grid_map.shape[:2] if len(grid_map.shape) > 2 else grid_map.shape

        self.first_path = PathDescription()
        self.first_found_flag = False
        self.best_path = PathDescription()

        self.costs_history = []
        self.path_lengths_history = []

        self.samples_taken = 0
        self.nodes_taken = 0
        self.samples_taken_history = []
        self.nodes_taken_history = []

        self.time_elapsed = 0
        self.dist_init_goal = dist_init_goal

        self.V = self.create_nodes_index()
        self.E = OrderedDict()

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def create_nodes_index(self) -> index.Index:
        """Create index of point."""
        p = index.Property()
        p.dimension = self.index_dim
        V = index.Index(interleaved=True, properties=p)
        return V

    def add_node(
        self,
        xy: Tuple[int, int],
        idx: int = 0,
    ):
        """Add node.

        Parameters
        ----------
        xy: Tuple[int, int]
            Point coordinates.
        idx: int, optional (default=0)
            Index of point.
        """
        self.nodes_taken += 1
        self.V.insert(idx, xy + xy, xy)

    def add_edge(
        self,
        parent_xy: Union[Any, Tuple[int, int]],
        child_xy: Tuple[int, int],
    ):
        """Add edge.

        Parameters
        ----------
        parent_xy: Tuple[int, int]
            Parent point coordinates.
        child_xy: Tuple[int, int]
            Child point coordinates.
        """
        self.E[child_xy] = parent_xy

    def nearest(
        self,
        xy: Tuple[int, int],
        n: int = 1,
    ) -> Generator:
        """Get nearest point(s).

        Parameters
        ----------
        xy: Tuple[int, int]
            Point coordinates.
        n: int, optional (default=1)
            Number of nearest points.
        """
        n = min(n, self.nodes_taken)
        return self.V.nearest(xy, num_results=n, objects='raw')

    def sample_free(self) -> Tuple[Union[int, Any], Union[int, Any]]:
        """Sample point.

        Returns
        -------
        Tuple[int, int]
            Point coordinates.
        """
        while True:
            if self.roi is not None and np.random.random() > self.mu:
                xy = self.non_uniform_sample()
            else:
                xy = self.uniform_sample()
            if self.is_cell_free(xy):
                return xy

    def uniform_sample(self) -> Tuple[int, int]:
        """Sample point according to uniform distribution.

        Returns
        -------
        Tuple[int, int]
            Point coordinates.
        """
        x = np.random.randint(0, self.xy_max[0])
        y = np.random.randint(0, self.xy_max[1])
        return x, y

    def non_uniform_sample(self) -> Tuple[int, int]:
        """Sample point according to non-uniform distribution.

        Returns
        -------
        Tuple[int, int]
            Point coordinates.
        """
        idx = np.random.choice(len(self.roi))
        xy = self.roi[idx]
        return xy

    def is_cell_free(self, xy: Tuple[int, int]) -> bool:
        """Check if cell is free.

        Parameters
        ----------
        xy: Tuple[int, int]
            Point coordinates.
        """
        x, y = xy
        # zeros by all channels = black = obstacle
        return sum(self.grid_map[x, y]) != 0

    def set_bounds(
        self,
        xy: Tuple[Union[int, float], Tuple[Union[int, float]]],
    ) -> Tuple[int, int]:
        """Set bounds.

        Parameters
        ----------
        xy: Tuple[int, int]
            Point coordinates.
        """
        x, y = xy
        x_max, y_max = self.xy_max
        x = min(max(0, x), x_max - 1)
        y = min(max(0, y), y_max - 1)
        return int(x), int(y)

    def steer(
        self,
        xy1: Tuple[int, int],
        xy2: Tuple[int, int],
        step: Union[int, float],
        eps: float = 1e-10,
    ) -> Tuple[int, int]:
        """Steer.

        Parameters
        ----------
        xy1: Tuple[int, int]
            First point coordinates.
        xy2: Tuple[int, int]
            Second point coordinates.
        step: Union[int, float]
            Step size.
        eps: float, optional (default=1e-10)
            Epsilon value to prevent zero division.
        """
        xy, d = self.euclidean_distance(xy1, xy2)
        # cos and sin
        u = xy / d if d > eps else 0
        step = min(d, step)
        steered_point = np.array(xy1) + u * step
        bounded = self.set_bounds(tuple(steered_point))
        return bounded

    def euclidean_distance(
        self,
        xy1: Tuple[int, int],
        xy2: Tuple[int, int],
    ) -> Tuple[float, float]:
        """Compute euclidean distance.

        Parameters
        ----------
        xy1: Tuple[int, int]
            First point coordinates.
        xy2: Tuple[int, int]
            Second point coordinates.
        """
        d = np.array(xy2) - np.array(xy1)
        return d, np.linalg.norm(d)

    def is_obstacle_free(
        self,
        xy1: Tuple[int, int],
        xy2: Tuple[int, int],
    ) -> bool:
        """Check if path is obstacle free.

        Parameters
        ----------
        xy1: Tuple[int, int]
            First point coordinates.
        xy2: Tuple[int, int]
            Second point coordinates.
        """
        _, d = self.euclidean_distance(xy1, xy2)
        n_pts = int(np.ceil(d / self.path_resolution))
        if n_pts >= 1:
            step_size = d / n_pts
            for i in range(n_pts + 1):
                next_point = self.steer(xy1, xy2, i * step_size)
                if not self.is_cell_free(next_point):
                    break
            else:
                return True
        else:
            if xy1 == xy2 and self.is_cell_free(xy1):
                return True
        return False

    def is_goal_reachable(
        self,
        xy_goal: Tuple[int, int],
    ) -> Tuple[bool, Union[Tuple[int, int], Any]]:
        """Check if goal point reachable.

        Parameters
        ----------
        xy_goal: Tuple[int, int]
            Point coordinates.
        """
        xy_near = next(self.nearest(xy_goal))
        if self.euclidean_distance(xy_near, xy_goal)[1] <= self.step_len:
            if xy_goal in self.E and xy_near == self.E[xy_goal]:
                return True, xy_near
            if self.is_obstacle_free(xy_near, xy_goal):
                return True, xy_near
        return False, None

    def compute_cost(
        self,
        xy1: Tuple[int, int],
        xy2: Tuple[int, int],
    ) -> Tuple[float, list]:
        """Compute cost.

        Parameters
        ----------
        xy1: Tuple[int, int]
            First point coordinates.
        xy2: Tuple[int, int]
            Second point coordinates.
        """
        cost = 0.0
        path = [xy2]
        for _ in range(len(self.E)):
            parent = self.E[xy2]
            if parent is None:
                break
            cost += self.euclidean_distance(parent, xy2)[1]
            xy2 = parent
            path.append(parent)
            if xy2 == xy1:
                break
        path.reverse()
        return cost, path

    def get_path_and_cost(self) -> Tuple[Union[list, Any], Union[float, Any]]:
        """Get cost."""
        flag, nearest_to_goal = self.is_goal_reachable(self.xy_goal)
        if flag and nearest_to_goal is not None:
            self.add_edge(nearest_to_goal, self.xy_goal)
            cost, path = self.compute_cost(self.xy_init, self.xy_goal)
            return path, cost
        return None, None

    def is_connectable(
        self,
        xy1: Tuple[int, int],
        xy2: Tuple[int, int],
    ) -> bool:
        """Check if points are connectable.

        Parameters
        ----------
        xy1: Tuple[int, int]
            First point coordinates.
        xy2: Tuple[int, int]
            Second point coordinates.
        """
        if self.V.count(xy2) == 0 and self.is_obstacle_free(xy1, xy2):
            self.add_node(xy2)
            self.add_edge(xy1, xy2)
            return True
        return False

    @abstractmethod
    def run(self, max_iter: Optional[int] = None):
        """Run RRT algorithm on grid map.

        Parameters
        ----------
        max_iter: int, optional (default=None)
            Number of maximum iterations.
        """
        raise NotImplementedError
