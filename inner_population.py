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
    def migrate(self, g: int):
        # TODO: Try out different migration strategies!
        # Ways to do migration:
        # - When does migration happen?
        #   - Before / after reproduction?
        #   - # of migrations per generation?
        # - How do individuals move?
        #   - "Acorn drop"
        #   - "Walk" to adjacent cell
        #   - Random shuffling
        #   - Move, or clone?
        # - How do you resolve competition?
        #   - Reserve some spots for migrants
        #   - Fill empty spots first
        #   - Let migrants overwrite locals
        #   - Replace local only if fitter
        # - How selective?
        #   - Random / more fit individuals migrate
        #   - Replace random / less fit individuals
        for x, y, i in ti.ndrange(*self.shape):
            # This magic number was calculated by brute force such that two
            # samples from a standard normal distribution will be (0, 0) 90% of
            # the time (meaning, migration happens 10% of the time).
            scale_factor = 0.5

            # "Acorn drop": draw from a standard normal distribution to find a
            # new home for this individual.
            dx = ti.round(scale_factor*ti.randn())
            dy = ti.round(scale_factor*ti.randn())

            # Make sure the new location is in bounds. This might cause
            # migrants to "pile up" at the edges of the environment.
            new_x = int(ti.math.clamp(x + dx, 0, ENVIRONMENT_SHAPE[0]))
            new_y = int(ti.math.clamp(y + dy, 0, ENVIRONMENT_SHAPE[1]))
            new_i = ti.random(ti.int32) % CARRYING_CAPACITY

            if new_x != x and new_y != y:
                # Move this individual to a random spot in a nearby location,
                # possibly overwriting an existing individual there. Then,
                # leave this spot empty.
                individual = self.pop[g, x, y, i]
                self.pop[g, new_x, new_y, new_i] = individual
                self.pop[g, x, y, i] = DEAD

    @ti.kernel
    def refill_empty_spaces(self, g: int):
        for x, y, i in ti.ndrange(*self.shape):
            individual = self.pop[g, x, y, i]

            # If this spot was unoccupied last generation...
            if individual.id == DEAD_ID:
                # Maybe let another individual spawn into this cell.
                if ti.random() < self.refill_rate:
                    # Either pick a random individual from the sub population
                    # at this location, or do tournament selection to pick a
                    # more fit individual.
                    if self.random_refill:
                        new_index = ti.random(int) %  CARRYING_CAPACITY
                        individual = self.pop[g, x, y, new_index]
                    else:
                        individual = tournament_selection(self.pop, g, x, y)

            self.pop[g, x, y, i] = individual


    @ti.kernel
    def populate_children(self, environment: ti.template(), g: int):
        for x, y, i in ti.ndrange(*self.shape):
            parent = self.pop[g, x, y, i]

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

        # Increment the next_id by the maximum number of individuals we might
        # have populated this generation. There may be unused ids, but that's
        # fine, we just care that ids are never reused.
        self.next_id[None] += ti.static(MAX_POPULATION_SIZE)


    def propagate(self, environment, generation):
        # TODO: Experiment with different orders...
        self.migrate(generation)
        self.refill_empty_spaces(generation)
        self.populate_children(environment, generation)


    def to_numpy(self):
        return self.pop.to_numpy()
