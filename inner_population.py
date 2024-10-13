import numpy as np
import taichi as ti

from constants import (
    ALL_ENVIRONMENTS_SHAPE, BITSTR_DTYPE, CARRYING_CAPACITY, INNER_GENERATIONS)
from hiff import weighted_hiff
from reproduction import mutation


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

# Unoccupied spaces are marked with a DEAD individual (all fields set to 0)
DEAD = Individual()
DEAD_ID = 0


@ti.data_oriented
class InnerPopulation:
    def __init__(self):
        self.shape = ALL_ENVIRONMENTS_SHAPE + (CARRYING_CAPACITY,)
        self.size = np.prod(self.shape)
        self.pop = Individual.field(shape=(INNER_GENERATIONS,) + self.shape)
        self.next_id = ti.field(dtype=ti.uint32, shape=())
        self.next_id[None] = 1

    @ti.func
    def get_next_id(self, e, x, y, i):
        # We generate a full population in every cell in parallel, so
        # figure out what the position of this individual would be if we
        # were computing that in sequence, and use that for setting its id.
        offset = (
            e * self.shape[1] * self.shape[2] * self.shape[3] +
            x * self.shape[2] * self.shape[3] +
            y * self.shape[3] +
            i)
        return self.next_id[None] + offset

    @ti.kernel
    def randomize(self):
        for e, x, y, i in ti.ndrange(*self.shape):
            self.pop[0, e, x, y, i] = Individual(
                bitstr=ti.cast(ti.random(int), BITSTR_DTYPE),
                id=self.get_next_id(e, x, y, i), parent=0, fitness=0.0, hiff=0)
        self.next_id[None] += ti.static(self.size)

    @ti.kernel
    def evaluate(self, environment: ti.template(), g: int) -> ti.uint32:
        for e, x, y, i in ti.ndrange(*self.shape):
            fitness, hiff = 0.0, 0
            # Only evaluate fitness of individuals that are alive.
            individual = self.pop[g, e, x, y, i]
            if individual.id != DEAD_ID:
                fitness, hiff = weighted_hiff(
                    individual.bitstr, environment.weights[x, y])
            self.pop[g, e, x, y, i].fitness = fitness
            self.pop[g, e, x, y, i].hiff = hiff

    @ti.kernel
    def propagate(self, environment: ti.template(), g: int):
        # TODO: For now, this function implements mortality and random
        # mutation, but nothing more advanced like biasing towards more fit
        # individuals, crossover, or migration.
        for x, y, i in ti.ndrange(*self.shape):
            individual = self.pop[g, x, y, i]
            # If this individual isn't fit enough to survive here...
            if individual.fitness < environment.min_fitness[x, y]:
                # Then mark it as dead in the next generation.
                self.pop[g + 1, x, y, i] = DEAD
            else:
                # Otherwise, make a random variation of this individual in the
                # same spot for the next generation.
                individual.bitstr ^= mutation()
                individual.parent = individual.id
                individual.id = self.get_next_id(x, y, i)
                individual.fitness = 0.0
                individual.hiff = 0
                self.pop[g + 1, x, y, i] = individual
        self.next_id[None] += ti.static(self.size)

    @ti.kernel
    def compute_diversity(self, diversity: ti.types.ndarray(dtype=ti.float32, ndim=4)):
        for g, e, x, y in ti.ndrange(*self.shape[:-1]):
            # TODO: Compute bitwise standard deviation for each sub population.
            diversity[g, e, x, y] = 0.0


    def to_numpy(self):
        return self.pop.to_numpy()
