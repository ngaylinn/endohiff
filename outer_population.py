"""A population of CPPNs for generating evolved environments for symbionts.
"""

import polars as pl

from constants import OUTER_POPULATION_SIZE
from environments import STATIC_ENVIRONMENTS


class OuterPopulation:
    def __init__(self):
        self.fitness_frames = []
        ...

    def randomize(self):
        # TODO: Randomize the CPPNs.
        ...

    def propagate(self, fitness_scores, generation):
        # TODO: Propagate the CPPNs.
        self.fitness_frames.append(pl.DataFrame({
            'Generation': generation,
            'fitness': fitness_scores,
        }))

    def make_environments(self):
        # TODO: Render environments from the CPPNs.
        return STATIC_ENVIRONMENTS['baym'](OUTER_POPULATION_SIZE)

    def get_logs(self):
        return pl.concat(self.fitness_frames)
