"""A population of CPPNs for generating evolved environments for symbionts.
"""

from neatchi import CppnPopulation, Matchmaker
import numpy as np
import polars as pl
import taichi as ti

from constants import (
    MAX_HIFF, OUTER_GENERATIONS, OUTER_POPULATION_SIZE, OUTPUT_PATH)
from environments import make_field
from visualize_inner_population import save_env_map


@ti.data_oriented
class OuterPopulation:
    def __init__(self, count=1):
        # We will evolve count populations of OUTER_POPULATION_SIZE each, so
        # the number of environments is just the product of those two values.
        self.pop_shape = (count, OUTER_POPULATION_SIZE)
        num_environments = int(np.prod(self.pop_shape))

        # Set up an index mapping from environment number to the trial and
        # trial-relative index of that environment. This way, the Environments
        # and InnerPopulation classes can work on a flat list without knowing
        # whether / how there is a structured outer population.
        self.index = ti.Vector.field(n=2, dtype=int, shape=num_environments)
        self.index.from_numpy(np.array(
            list(np.ndindex(*self.pop_shape)), dtype=np.int32))

        # Inputs are (x, y) position, output is min fitness.
        cppn_shape = (2, 1)

        # For performing tournament selection on the CPPNs.
        self.matchmaker = Matchmaker(self.pop_shape, OUTER_GENERATIONS)

        # The CPPNs used to evolve and render environments.
        self.cppns = CppnPopulation(
            self.pop_shape, cppn_shape, self.index, self.matchmaker)

        # A space to hold the Environments generated using the CPPNs above.
        self.env = make_field(num_environments)

    def randomize(self):
        self.cppns.randomize()

    def propagate(self, generation):
        # NOTE: make sure to set fitness scores in self.matchmaker.fitness
        # before calling this method!
        # We use a very high mutation rate to increase diversity in the
        # population.
        self.cppns.propagate(generation, mutation_rate=0.1)

    @ti.kernel
    def render_environments(self):
        # Render the CPPNs to get the minimum fitness threshold for every
        # location in every environment.
        ne, ew, eh = self.env.shape
        for e, x, y, in ti.ndrange(ne, ew, eh):
            inputs = ti.Vector([x / ew, y / eh])
            outputs = self.cppns.activate(e, inputs)
            self.env[e, x, y] = ti.cast(outputs[0] * MAX_HIFF, ti.float16)

    def make_environments(self):
        self.render_environments()
        return self.env

    def get_logs(self):
        # All fitness tracking happens on the GPU to minimize data transfers,
        # so grab it and put it into a dataframe only when needed.
        fitness = self.matchmaker.fitness.to_numpy()

        # Enumerate the dimensions of the fitness data to index it in the logs.
        generations, trials, envs = zip(*np.ndindex(fitness.shape))

        return pl.DataFrame({
            'Generation': generations,
            'Trial': trials,
            'Fitness': fitness.flatten(),
        })

    def visualize(self, path, trial=None):
        """Save a map for each environment in this population.
        """
        # TODO: This is painfully slow. If using this often, maybe consider
        # optimizing the map rendering?
        environments = self.env.to_numpy()
        for e, env in enumerate(environments):
            # If a trial was specified, skip enviornments NOT from that trial.
            if trial is not None and self.index[e][0] != trial:
                continue
            path.mkdir(exist_ok=True, parents=True)
            save_env_map(path, env, f'cppn{e}')


# A demo to visualize what a random initial population of CPPNs looks like.
if __name__ == '__main__':
    ti.init(ti.cuda)

    outer_population = OuterPopulation()
    outer_population.randomize()
    outer_population.make_environments()
    outer_population.visualize(OUTPUT_PATH / 'random_cppn_environments')
