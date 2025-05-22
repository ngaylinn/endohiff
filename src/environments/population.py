from argparse import ArgumentParser
from pathlib import Path
import sys

from neatchi import CppnPopulation, Matchmaker
import numpy as np
import polars as pl
import taichi as ti

from ..constants import MAX_HIFF, OUTER_GENERATIONS, OUTER_POPULATION_SIZE
from .fitness import eval_env_fitness
from .util import make_env_field


@ti.data_oriented
class EnvironmentPopulation:
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
        self.env = make_env_field(num_environments)

    def randomize(self):
        self.cppns.randomize()

    def evaluate_fitness(self, bitstr_pop, generation):
        eval_env_fitness(
            self.matchmaker.fitness, self.index, bitstr_pop, generation)

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


def main(path):
    from .visualize_one import save_env_map
    ti.init(ti.cuda)

    path.mkdir(exist_ok=True, parents=True)
    population = EnvironmentPopulation()
    population.randomize()
    population.make_environments()
    evironments = population.env.to_numpy()
    for e, env_data in enumerate(evironments):
        save_env_map(env_data, path / f'cppn_{e}.png')

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Visualize a random initial popuation of CPPN environments')
    parser.add_argument(
        'path', type=Path,
        help='Where to save the sample environment visualizations')
    args = vars(parser.parse_args())
    sys.exit(main(**args))
