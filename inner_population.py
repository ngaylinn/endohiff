"""Data types for representing an evolvable population of symbionts.

The InnerPopulation class does most of the work of simulating a population,
including propagating across generations and migrating across space. It's
mostly just a wrapper around a field of Individuals, used to track and record
the full state of the simulation.
"""

import taichi as ti

from constants import (
    BITSTR_DTYPE, CARRYING_CAPACITY, CROSSOVER_RATE, DEAD_ID,
    ENVIRONMENT_SHAPE, INNER_GENERATIONS, MAX_POPULATION_SIZE, REFILL_RATE)
from hiff import weighted_hiff
from reproduction import mutation, crossover, tournament_selection


@ti.dataclass
class Individual:
    bitstr: BITSTR_DTYPE
    # Each individual across all generations has a unique identifier. Zero
    # indicates this individual is not alive.
    id: ti.uint32
    # The identifier of the primary parent of this individual. This is either
    # the only parent (clonal reproduction) or one of two parents whose genes
    # were combined via crossover. Zero indicates this individual was
    # spontaneously generated.
    parent1: ti.uint32
    # The identifier of the secondary parent of this individual. This is zero
    # except in the case where crossover was performed.
    parent2: ti.uint32
    # The fitness score of this individual (weighted HIFF).
    fitness: ti.float32
    # The raw HIFF score of this individual.
    hiff: ti.uint32

# Unoccupied spaces are marked with a DEAD individual (all fields set to 0)
DEAD = Individual()


@ti.data_oriented
class InnerPopulation:
    def __init__(self):
        self.shape = ENVIRONMENT_SHAPE + (CARRYING_CAPACITY,)
        self.pop = Individual.field(shape=(INNER_GENERATIONS,) + self.shape)
        # TODO: Instead of a counter, why not make it determined solely by
        # g, x, y, and i?
        self.next_id = ti.field(dtype=ti.uint32, shape=())
        self.next_id[None] = 1

    @ti.func
    def get_next_id(self, x, y, i):
        # We generate a full population in every cell in parallel, so
        # figure out what the position of this individual would be if we
        # were computing that in sequence, and use that for setting its id.
        offset = x * self.shape[1] * self.shape[2] + y * self.shape[2] + i
        return self.next_id[None] + offset

    @ti.kernel
    def randomize(self):
        # Randomize the population.
        for x, y, i in ti.ndrange(*self.shape):
            self.pop[0, x, y, i] = Individual(
                bitstr=ti.random(BITSTR_DTYPE),
                id=self.get_next_id(x, y, i))

        # Keep assigning new ids instead of reusing them across experiments.
        self.next_id[None] += ti.static(MAX_POPULATION_SIZE)

    @ti.kernel
    def evaluate(self, environment: ti.template(), g: ti.i32):
        for x, y, i in ti.ndrange(*self.shape):
            fitness, hiff = ti.cast(0.0, ti.float32), ti.cast(0, ti.uint32)

            # Only evaluate fitness of individuals that are alive.
            individual = self.pop[g, x, y, i]
            if individual.id != DEAD_ID:
                fitness, hiff = weighted_hiff(
                    individual.bitstr, environment.weights[x, y])

            self.pop[g, x, y, i].fitness = fitness
            self.pop[g, x, y, i].hiff = hiff

    @ti.kernel
    def migrate(self, g: int):
        # Migrants replace local individuals only if they have a higher fitness
        # than the current inhabitant. This ensures that more fit individuals
        # are prioritized for migration.
        for x, y, i in ti.ndrange(*self.shape):
            # This magic number was calculated by brute force such that two
            # samples from a standard normal distribution will be (0, 0) 90% of
            # the time (meaning, each individual migrates 10% of the time).
            scale_factor = 0.25
            dx = ti.round(scale_factor * ti.randn())
            dy = ti.round(scale_factor * ti.randn())

            # Find the target location to migrate to.
            new_x = int(ti.math.clamp(x + dx, 0, ENVIRONMENT_SHAPE[0]))
            new_y = int(ti.math.clamp(y + dy, 0, ENVIRONMENT_SHAPE[1]))
            new_i = ti.random(ti.int32) % CARRYING_CAPACITY

            # If this individual is moving to a new location...
            if new_x != x or new_y != y:
                migrant = self.pop[g, x, y, i]
                local = self.pop[g, new_x, new_y, new_i]

                # If the migrant is more fit than the resident in the location
                # it's moving to, replace the local and leave an empty space.
                if migrant.fitness > local.fitness:
                    self.pop[g, new_x, new_y, new_i] = migrant
                    self.pop[g, x, y, i] = DEAD

    @ti.kernel
    def refill_empty_spaces(self, g: int):
        for x, y, i in ti.ndrange(*self.shape):
            individual = self.pop[g, x, y, i]

            # If this spot was unoccupied last generation...
            if individual.id == DEAD_ID:
                # Maybe let another individual spawn into this cell.
                if ti.random() < REFILL_RATE:
                    # If you pick a DEAD individual, try a few more times. This
                    # makes it significantly easier for a migrant to colonize
                    # an empty cell.
                    for _ in range(4):
                        new_index = ti.random(int) %  CARRYING_CAPACITY
                        individual = self.pop[g, x, y, new_index]
                        if individual.id != DEAD_ID:
                            break

            # TODO: Sometimes we repeat an id because of this line. Should we
            # handle this differently? Do we even need to track ids?
            self.pop[g, x, y, i] = individual

    @ti.kernel
    def populate_children(self, environment: ti.template(), g: int, crossover_enabled: bool):
        for x, y, i in ti.ndrange(*self.shape):
            min_fitness = environment.min_fitness[x, y]
            # TODO: Without selection here, there's a big difference between
            # baym and flat, but that's because flat is basically just doing a
            # random search, not hill-climbing! If I do selection here, then
            # the effect dissappears. Is there a happy-medium?
            # p = tournament_selection(self.pop, g, x, y, min_fitness)
            p = ti.select(self.pop[g, x, y, i].fitness >= min_fitness, i, -1)

            # If no one in this location is fit to reproduce...
            if p < 0:
                # Then mark it as dead in the next generation.
                self.pop[g + 1, x, y, i] = DEAD
            else:
                # Otherwise, make a child from the selected parent.
                parent = self.pop[g, x, y, p]
                child = Individual(
                    parent.bitstr, self.get_next_id(x, y, i), parent.id
                )

                # Maybe pick a mate and perform crossover
                if crossover_enabled and ti.random() < CROSSOVER_RATE:
                    m = tournament_selection(self.pop, g, x, y, min_fitness)
                    if m >= 0:
                        mate = self.pop[g, x, y, m]
                        child.bitstr = crossover(parent.bitstr, mate.bitstr)
                        child.parent2 = mate.id

                # Apply mutation to new child
                child.bitstr ^= mutation()

                # Place the child in the next generation
                self.pop[g + 1, x, y, i] = child

        # Increment the next_id by the maximum number of individuals we might
        # have populated this generation. There may be unused ids, but that's
        # fine, we just care that ids are never reused.
        self.next_id[None] += ti.static(MAX_POPULATION_SIZE)

    def propagate(self, environment, generation, migrate_enabled, crossover_enabled):
        if migrate_enabled:
            self.migrate(generation)
        self.refill_empty_spaces(generation)
        self.populate_children(environment, generation, crossover_enabled)

    def to_numpy(self):
        return self.pop.to_numpy()



