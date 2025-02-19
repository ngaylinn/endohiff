"""A population of CPPNs for generating evolved environments for symbionts.
"""

from neatchi import CppnPopulation, Matchmaker
import numpy as np
import polars as pl
import taichi as ti

from constants import (
    BITSTR_LEN, BITSTR_POWER, MAX_HIFF, NUM_WEIGHTS, OUTER_GENERATIONS,
    OUTER_POPULATION_SIZE, OUTPUT_PATH)
from environments import Environments
from visualize_inner_population import save_env_map


@ti.data_oriented
class OuterPopulation:
    def __init__(self, count=1, use_weights=False):
        # Whether or not to evolve substring weights in the environment.
        self.use_weights = use_weights

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

        # We need to NUM_WEIGHTS values (+1 for min fitness) at each position
        # in the environment. That would require a very large CPPN, though, so
        # we use a trick to reduce the dimensionality. Instead of convolving
        # the 2D space of the environment, we make an extra dimension for the
        # bitstring itself, and instead of having one output per substring, we
        # have one output per LENGTH of substring. That way, we have one output
        # for all the length-2 substrings; to get a value for each substring,
        # we call the CPPN once for each of those length-2 substrings at each
        # location (the CPPN takes 3 input values: x, y, and s for substring).
        cppn_shape = (3, BITSTR_POWER + 1) if self.use_weights else (2, 1)

        # For performing tournament selection on the CPPNs.
        self.matchmaker = Matchmaker(self.pop_shape, OUTER_GENERATIONS)

        # The CPPNs used to evolve and render environments.
        self.cppns = CppnPopulation(
            self.pop_shape, cppn_shape, self.index, self.matchmaker)

        # A space to hold the Environments generated using the CPPNs above.
        self.env = Environments(num_environments)

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
        # Populate the settings for each location in each evolved environment
        ne, ew, eh = self.env.shape
        for e, x, y, in ti.ndrange(ne, ew, eh):
            if ti.static(self.use_weights):
                weights = ti.Vector([0.0] * NUM_WEIGHTS)
                outputs = ti.Vector([0.0] * (BITSTR_POWER + 1))

                # For each location, we must compute values for all the weights,
                # which are broken down by substring length. The shortest
                # substrings have length 2, and there are BITSR_LEN // 2 of them,
                # so for each location we sample that many points along the third
                # dimension that represents the substrings of each length.
                num_samples = BITSTR_LEN // 2
                for s in range(num_samples):
                    # Activate the CPPN at this sample point to figure out the
                    # weights for substrings of all lengths at this point.
                    inputs = ti.Vector([x / ew, y / eh, s / num_samples])
                    outputs = self.cppns.activate(e, inputs)

                    # An example of computing weight indices when BITSTR_POWER == 4
                    # The outer loop iterates over values of s while the inner loop
                    # iterates over values of p, generating 2**BITSTR_POWER - 1
                    # weights indexed by the value w.
                    #
                    #        values of s:               values of w:
                    # p = 0: [0, 1, 2, 3, 4, 5, 6, 7]   [0, 1, 2, 3, 4, 5, 6, 7]
                    # p = 1: [0,    2,    4,    6   ]   [8,    9,   10,   11   ]
                    # p = 2: [0,          4         ]   [12,        13         ]
                    # p = 3: [0                     ]   [14                    ]

                    # We start off with the length-2 substrings, which are at the
                    # beginning of the weights list, so the offset into the weights
                    # list starts at 0.
                    w_off = 0

                    # Find corresponding weights for substrings of all lengths...
                    for p in range(BITSTR_POWER):
                        # For each length of substring, there's a different number
                        # of weights to compute. So, the first step is to decide if
                        # this sample point is relevant to substrings of this size.
                        # If it's not, then it won't be relevant to higher powers,
                        # either, since they care about even fewer sample points.
                        substr_len = 2**(p + 1)
                        if (2 * s) % substr_len != 0:
                            break

                        # Find the current weight index, then store the associated
                        # output from the CPPN into the weights vector.
                        w = s // (2**p) + w_off
                        weights[w] = outputs[p]

                        # Now we prepare to look at weights for longer substrings.
                        # Skip the weight index past all the weights of the current
                        # size.
                        substr_count = BITSTR_LEN // substr_len
                        w_off += substr_count

                # Actually save the weight values to the environment.
                self.env.weights[e, x, y] = weights
                # Use the last output from the CPPN to set the min fitness
                # threshold. It doesn't matter which value of s was used as the
                # input; since min_fitness has nothing to do with the various
                # substrings, it's totally arbitrary.

                self.env.min_fitness[e, x, y] = outputs[BITSTR_POWER] * MAX_HIFF
            else:
                # If we're not evolving substring weights, then we just need to
                # find the min fitness threshold, which is much simpler (and
                # set all weights to 1.0).
                inputs = ti.Vector([x / ew, y / eh])
                outputs = self.cppns.activate(e, inputs)
                self.env.min_fitness[e, x, y] = outputs[0] * MAX_HIFF
                self.env.weights[e, x, y].fill(1.0)

    def make_environments(self):
        self.render_environments()
        return self.env

    def get_logs(self):
        # All fitness tracking happens on the GPU to minimize data transfers,
        # so grab it and put it into a dataframe only when needed.
        fitness = self.matchmaker.fitness.to_numpy()
        parents, mates = self.matchmaker.matches.to_numpy().reshape(-1, 2).T

        # Enumerate the dimensions of the fitness data to index it in the logs.
        generations, trials, envs = zip(*np.ndindex(fitness.shape))

        return pl.DataFrame({
            'Generation': generations,
            'Trial': trials,
            'Fitness': fitness.flatten(),
            'parent': parents,
            'mate': mates,
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
    ti.init(ti.cuda, unrolling_limit=0)

    outer_population = OuterPopulation(use_weights=False)
    outer_population.randomize()
    outer_population.make_environments()
    outer_population.visualize(
        OUTPUT_PATH / 'random_cppn_environments_without_weights')

    outer_population = OuterPopulation(use_weights=True)
    outer_population.randomize()
    outer_population.make_environments()
    outer_population.visualize(
        OUTPUT_PATH / 'random_cppn_environments_with_weights')
