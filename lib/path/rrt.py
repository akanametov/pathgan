import time
from typing import Tuple, List, Optional
import numpy as np

from pathlib import Path
from PIL import Image
from copy import deepcopy

from path.rrt_base import RRTBase


class RRT(RRTBase):
    def __init__(self, grid_map: np.ndarray,
                 xy_init: Tuple[int, int],
                 xy_goal: Tuple[int, int],
                 roi: Optional[List[tuple]] = None,
                 path_resolution: int = 1,
                 step_len: float = 0.5,
                 max_iter: int = 1,
                 mu: float = 0.1,
                 seed: int = 42,
                 dist_init_goal: Optional[float] = None):

        params = {name: value for name, value in locals().items() \
                  if name in RRTBase.__init__.__code__.co_varnames and name != 'self'}
        super().__init__(**params)

    def run(self, max_iter: Optional[int] = None):
        if max_iter is not None:
            self.max_iter = max_iter

        start_time = time.time()
        self.add_node(self.xy_init, 0)
        self.add_edge(None, self.xy_init)
        for t in range(self.max_iter):
            if (t + 1) % 10 == 0:
                self.samples_taken_history.append(self.samples_taken)
                self.nodes_taken_history.append(self.nodes_taken)

            xy_rand = self.sample_free()
            xy_near = next(self.nearest(xy_rand))
            xy_new = self.steer(xy_rand, xy_near, self.step_len)
            if not self.V.count(xy_new) == 0 or not self.is_cell_free(xy_new):
                continue
            self.samples_taken += 1

            self.is_connectable(xy_near, xy_new)
            path, cost = self.get_path_and_cost()

            if path and cost and cost < self.best_path.cost:
                end_time = time.time()
                self.time_elapsed = (end_time - start_time) / 10 ** 3

                self.first_found_flag = True
                self.first_path.update(path, cost,
                                       self.time_elapsed, t + 1,
                                       self.samples_taken, self.nodes_taken)
                self.best_path = deepcopy(self.first_path)
                return True
        return False

