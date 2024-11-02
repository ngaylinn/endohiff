import taichi as ti
import pandas as pd
import numpy as np

from constants import (
    BITSTR_DTYPE, CARRYING_CAPACITY, ENVIRONMENT_SHAPE, INNER_GENERATIONS,
    MAX_POPULATION_SIZE, REFILL_RATE)
from hiff import weighted_hiff
from reproduction import mutation, crossover, diverse_crossover, tournament_selection


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
        self.refill_rate = REFILL_RATE
        self.random_refill = True
        self.shape = ENVIRONMENT_SHAPE + (CARRYING_CAPACITY,)
        self.pop = Individual.field(shape=(INNER_GENERATIONS,) + self.shape)
        self.next_id = ti.field(dtype=ti.uint32, shape=())
        self.next_id[None] = 1
        self.fitness_values = ti.Vector.field(1, dtype=ti.float32, shape=(CARRYING_CAPACITY,))  # Temporary field for fitness values

        # holding diversity over generations
        self.pop_diversity = ti.field(dtype=ti.f32, shape=(INNER_GENERATIONS,))
        self.pop_sum_fitness = ti.field(dtype=ti.f32, shape=(INNER_GENERATIONS,))  # Sum of all the fitnesses


    @ti.func
    def get_next_id(self, x, y, i):
        # We generate a full population in every cell in parallel, so
        # figure out what the position of this individual would be if we
        # were computing that in sequence, and use that for setting its id.
        offset = x * self.shape[1] * self.shape[2] + y * self.shape[2] + i
        return self.next_id[None] + offset

    @ti.kernel
    def randomize(self):
        for x, y, i in ti.ndrange(*self.shape):
            self.pop[0, x, y, i] = Individual(
                bitstr=ti.cast(ti.random(int), BITSTR_DTYPE),
                id=self.get_next_id(x, y, i), parent=0, fitness=0.0, hiff=0)
        self.next_id[None] += ti.static(MAX_POPULATION_SIZE)

    @ti.kernel
    def evaluate(self, environment: ti.template(), g: int):
        fitness_sum = 0.0
        count = 0
        # Variables for calculating standard deviation
        fitness_squared_sum = 0.0

        for x, y, i in ti.ndrange(*self.shape):
            fitness, hiff = 0.0, 0
            # Only evaluate fitness of individuals that are alive.
            individual = self.pop[g, x, y, i]
            if individual.id != DEAD_ID:
                fitness, hiff = weighted_hiff(
                    individual.bitstr, environment.weights[x, y])
                fitness_sum += fitness
                fitness_squared_sum += fitness ** 2  # Sum of squares for std deviation calculation
                count += 1
                self.fitness_values[i][0] = fitness  # Store fitness value in the temporary field
            self.pop[g, x, y, i].fitness = fitness
            self.pop[g, x, y, i].hiff = hiff

        # Calculate diversity and total fitness
        if count > 0:
            average_fitness = fitness_sum / count
            variance = (fitness_squared_sum / count) - (average_fitness ** 2)
            self.pop_diversity[g] = ti.sqrt(variance)  # Standard deviation
        else:
            self.pop_diversity[g] = 0.0  # No individuals
        self.pop_sum_fitness[g] = fitness_sum  # Assigning to the field

    @ti.kernel
    def propagate(self, environment: ti.template(), g: int):
        # TODO: For now, this function implements mortality and random
        # mutation, but nothing more advanced like biasing towards more fit
        # individuals, crossover, or migration.
        for x, y, i in ti.ndrange(*self.shape):
            parent = self.pop[g, x, y, i]

            # If this cell was unoccupied in the last generation...
            if parent.id == DEAD_ID:
                # Maybe let another individual spawn into this cell.
                if ti.random() < self.refill_rate:
                    # Either pick a parent at random from the sub population at
                    # this location, or do tournament selection to pick a more
                    # fit parent.
                    if self.random_refill:
                        parent_index = ti.random(int) %  CARRYING_CAPACITY
                        parent = self.pop[g, x, y, parent_index]
                    else:
                        parent = tournament_selection(self.pop, g, x, y)

            # If this individual isn't fit enough to survive here...
            if parent.fitness < environment.min_fitness[x, y]:
                # Then mark it as dead in the next generation.
                self.pop[g + 1, x, y, i] = DEAD
            else:
                # select another individual from same sub-pop for crossover
                # Select a mate via tournament selection for crossover
                mate = tournament_selection(self.pop, g, x, y)

                # creating a child from the individual and performing crossover
                child = Individual()
                child.bitstr = diverse_crossover(parent.bitstr, mate.bitstr) #CHANGED TO UNIFORM_CROSSOVER

                # Apply mutation to new child
                child.bitstr ^= mutation()

                #update child's metadata
                child.parent = parent.id
                child.id = self.get_next_id(x, y, i)
                child.fitness = 0.0
                child.hiff = 0

                # Place the child in the next generation
                self.pop[g + 1, x, y, i] = child

        self.next_id[None] += ti.static(MAX_POPULATION_SIZE)

    def to_numpy(self):
        return self.pop.to_numpy()
