import numpy as np
import taichi as ti

from constants import (
    BITSTR_DTYPE, CARRYING_CAPACITY, ENVIRONMENT_SHAPE, NUM_GENERATIONS)
from hiff import weighted_hiff


@ti.dataclass
class Individual:
    bitstr: BITSTR_DTYPE
    # Each individual across all generations has a unique identifier. Zero
    # indicates this individual is not alive.
    id: ti.uint32
    # The identifier of the parent of this individual. Zero indicates this
    # individual was spontaneously generated.
    parent: ti.uint32
    # The fitness score of this individual (weighted HIFF).
    fitness: ti.float32
    # The raw HIFF score of this individual.
    hiff: ti.uint32


@ti.data_oriented
class InnerPopulation:
    def __init__(self):
        self.shape = ENVIRONMENT_SHAPE + (CARRYING_CAPACITY,)
        self.size = np.prod(self.shape)
        self.pop = Individual.field(shape=(NUM_GENERATIONS,) + self.shape)
        self.generation = ti.field(dtype=ti.uint32, shape=())
        self.next_id = ti.field(dtype=ti.uint32, shape=())
        self.next_id[None] = 1

    @ti.kernel
    def randomize(self, g: int): # TODO: Remove g once propagate is written.
        for x, y, i in ti.ndrange(*self.shape):
            # We generate a full population in every cell in parallel, so
            # figure out what the position of this individual would be if we
            # were computing that in sequence, and use that for setting its id.
            offset = x * self.shape[1] * self.shape[2] + y * self.shape[2] + i
            self.pop[g, x, y, i] = Individual(
                bitstr=ti.cast(ti.random(int), BITSTR_DTYPE),
                id=self.next_id[None] + offset,
                parent=0, fitness=0.0, hiff=0)
        self.next_id[None] += ti.static(self.size)

    @ti.kernel
    def evaluate(self, environment: ti.template(), g: int) -> ti.uint32:
        best_hiff = ti.cast(0, ti.uint32)
        for x, y, i in ti.ndrange(*self.shape):
            fitness, hiff = weighted_hiff(
                self.pop[g, x, y, i].bitstr, environment.weights[x, y])
            ti.atomic_max(best_hiff, hiff)
            self.pop[g, x, y, i].fitness = fitness
            self.pop[g, x, y, i].hiff = hiff
        return best_hiff

    def propagate(self, g):
        # TODO: Generate the next generation from the current one.
        self.randomize(g + 1)

    def to_numpy(self):
        return self.pop.to_numpy()
