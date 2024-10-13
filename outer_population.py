from neatchi import CppnPopulation
import numpy as np
import taichi as ti

from constants import (
    ALL_ENVIRONMENTS_SHAPE, BITSTR_POWER, BITSTR_LEN, NUM_HOST_ENVIRONMENTS,
    NUM_WEIGHTS)
from environment import Environment

class OuterPopulation:
    def __init__(self):
        self.cppns = CppnPopulation(
            # My laptop can't handle running multiple trials at a time, so we
            # just use a single population of NUM_HOST_ENVIRONMENTS CPPNs.
            (1, NUM_HOST_ENVIRONMENTS),
            # Each CPPN is used to convolve a 3D volume: the bitstring at each
            # position in a 2D space. The CPPN output is one per substring
            # length (so there is a 1D map for each length 2 substring at each
            # position in the environment, and another for length 4, etc.) as
            # well as one additional output for the minimum fitness.
            (3, BITSTR_POWER + 1),
            # Associate each individual in this population with a single world.
            np.array(list(np.ndindex(1, NUM_HOST_ENVIRONMENTS))),
            # TODO: Use a matchmaker for selection.
            None)
        self.env = Environment()

    def randomize(self):
        self.cppns.randomize()

    def propagate(self):
        # TODO: Generate the next generation from the current one.
        self.cppns.randomize()

    @ti.kernel
    def make_environment(self):
        # For each individual render each cell of the environment.
        for e, x, y in ti.ndrange(ALL_ENVIRONMENTS_SHAPE):
            weights = ti.Vector([0] * NUM_WEIGHTS)
            # For this position in the environment, sample at several points
            # along the bitstring to figure out what the weights for that
            # portion of the bitstring should be. Since the smallest substring
            # we consider is 2, the samples are separated by that much.
            for s in range(BITSTR_LEN // 2):
                # Activate the CPPN and figure out what the weights should be
                # across all the different substring sizes
                inputs = ti.Vector([x, y, s])
                outputs = self.cppns.activate(e, inputs)

                # Consider the different sizes of substring, and if this sample
                # point is aligned with that size, then use the output value to
                # indicate the weight of that particular substring.
                for p in range(BITSTR_POWER):
                    # If this sample point is relevant for this power...
                    if s % (2**p):
                        ...
            for p in range(BITSTR_POWER):
                substr_len = 2**(p + 1)
                for s in range(BITSTR_LEN // substr_len):
                    # TODO: This isn't right. We want to organize this so that
                    # we can use ALL the outputs from each activation, not just
                    # one of them.
                    inputs = ti.Vector([x, y, s])
                    outputs = self.cppns.activate(e, inputs)
                    weights[w] = outputs[p]
                    w += 1
            self.env.min_fitness[e, x, y] = ...
            self.env.weights[e, x, y] = weights
