"""Data types for representing an evolvable population of symbionts.

The InnerPopulation class does most of the work of simulating a population,
including propagating across generations and migrating across space. It's
mostly just a wrapper around a field of Individuals, used to track and record
the full state of the simulation.
"""

import numpy as np
import polars as pl
import taichi as ti

from constants import (
    BITSTR_DTYPE, CARRYING_CAPACITY, CROSSOVER_RATE, ENVIRONMENT_SHAPE,
    INNER_GENERATIONS, MIGRATION_RATE, REFILL_RATE)
from inner_fitness import weighted_hiff, count_ones
from reproduction import mutation, crossover, tournament_selection


@ti.dataclass
class Individual:
    bitstr: BITSTR_DTYPE
    # The fitness score of this individual (weighted HIFF).
    fitness: ti.float32
    # The raw HIFF score of this individual.
    hiff: ti.int16
    # How many ones are in the bitstr.
    one_count: ti.int8

    @ti.func
    def is_dead(self):
        return ti.math.isnan(self.fitness)

    @ti.func
    def is_alive(self):
        return not self.is_dead()

    @ti.func
    def mark_dead(self):
        self.fitness = ti.math.nan


@ti.data_oriented
class InnerPopulation:
    def __init__(self, num_environments=1):
        self.num_environments = num_environments
        ne = num_environments
        ig = INNER_GENERATIONS
        ew, eh = ENVIRONMENT_SHAPE
        cc = CARRYING_CAPACITY

        # The shape of the population in each generation
        self.shape = (ne, ew, eh, cc)
        # A full population of individuals for all generations.
        self.pop = Individual.field(shape=(ne, ig, ew, eh, cc))

        # An index with positional metadata for each Individual in self.pop,
        # used to generate annotated log files.
        self.index = pl.DataFrame({
            'Generation':  np.arange(ig).repeat(ew * eh * cc),
            'x': np.tile(np.arange(ew).repeat(eh * cc), ig),
            'y': np.tile(np.arange(eh).repeat(cc), ig * ew),
        })

    @ti.kernel
    def randomize(self):
        # Randomize the population.
        for e, x, y, i in ti.ndrange(*self.shape):
            self.pop[e, 0, x, y, i] = Individual(ti.random(BITSTR_DTYPE))

    @ti.kernel
    def evaluate(self, environment: ti.template(), g: ti.i32):
        for e, x, y, i in ti.ndrange(*self.shape):
            # Only evaluate fitness of individuals that are alive.
            individual = self.pop[e, g, x, y, i]
            if individual.is_alive():
                fitness, hiff = weighted_hiff(
                    individual.bitstr, environment.weights[e, x, y])
                individual.fitness = fitness
                individual.hiff = hiff
                individual.one_count = count_ones(individual.bitstr)
                self.pop[e, g, x, y, i] = individual

    @ti.kernel
    def migrate(self, g: int):
        # Migrants replace local individuals only if they have a higher fitness
        # than the current inhabitant. This ensures that more fit individuals
        # are prioritized for migration.
        for e, x, y, i in ti.ndrange(*self.shape):
            dx = ti.round(MIGRATION_RATE * ti.randn())
            dy = ti.round(MIGRATION_RATE * ti.randn())

            # Find the target location to migrate to.
            new_x = int(ti.math.clamp(x + dx, 0, ENVIRONMENT_SHAPE[0] - 1))
            new_y = int(ti.math.clamp(y + dy, 0, ENVIRONMENT_SHAPE[1] - 1))
            # NOTE: Careful typing to satisfy Taichi'a debugger.
            new_i = ti.cast(ti.random(ti.uint32) % CARRYING_CAPACITY, ti.int32)

            # If this individual is moving to a new location...
            if new_x != x or new_y != y:
                migrant = self.pop[e, g, x, y, i]
                local = self.pop[e, g, new_x, new_y, new_i]

                # If the migrant is more fit than the resident in the location
                # it's moving to, replace the local and leave an empty space.
                migrant_is_better = migrant.is_alive() and (
                    (local.is_dead() or migrant.fitness > local.fitness))
                if migrant_is_better:
                    # To avoid race conditions, use the next generation as a
                    # scratch space, so this block of threads isn't reading and
                    # writing to the same memory location. After this kernel,
                    # call commit_scratch_space to copy the results back to the
                    # present generation.
                    self.pop[e, g + 1, new_x, new_y, new_i] = migrant
                    self.pop[e, g, x, y, i].mark_dead()

    @ti.kernel
    def refill_empty_spaces(self, g: int):
        for e, x, y, i in ti.ndrange(*self.shape):
            individual = self.pop[e, g, x, y, i]

            # If this spot was unoccupied last generation...
            if individual.is_dead():
                # Maybe let another individual spawn into this cell.
                if ti.random() < REFILL_RATE:
                    # Try a few times to find an individual in this location
                    # that's not dead, and let them take over the empty spot.
                    for _ in range(4):
                        # NOTE: Careful typing to satisfy Taichi'a debugger.
                        new_i = ti.cast(
                            ti.random(ti.uint32) % CARRYING_CAPACITY, ti.int32)
                        individual = self.pop[e, g, x, y, new_i]
                        if individual.is_alive():
                            # TODO: Sometimes we repeat an id because of this
                            # line. Should we handle this differently? Do we
                            # even need to track ids?
                            # To avoid race conditions, use the next generation
                            # as a scratch space, so this block of threads
                            # isn't reading and writing to the same memory
                            # location. After this kernel, call
                            # commit_scratch_space to copy the results back to
                            # the present generation.
                            self.pop[e, g + 1, x, y, i] = individual
                            break

    @ti.kernel
    def clear_scratch_space(self, g: int):
        for e, x, y, i in ti.ndrange(*self.shape):
            self.pop[e, g + 1, x, y, i].mark_dead()

    @ti.kernel
    def commit_scratch_space(self, g: int):
        for e, x, y, i in ti.ndrange(*self.shape):
            if self.pop[e, g + 1, x, y, i].is_alive():
                self.pop[e, g, x, y, i] = self.pop[e, g + 1, x, y ,i]

    @ti.kernel
    def populate_children(self, environment: ti.template(), g: int, crossover_enabled: bool):
        for e, x, y, i in ti.ndrange(*self.shape):
            min_fitness = environment.min_fitness[e, x, y]
            # TODO: Without selection here, there's a big difference between
            # baym and flat, but that's because flat is basically just doing a
            # random search, not hill-climbing! If I do selection here, then
            # the effect dissappears. Is there a happy-medium?
            # p = tournament_selection(self.pop, g, x, y, min_fitness)
            parent = self.pop[e, g, x, y, i]

            # If the individual in this location isn't fit to reproduce...
            if parent.is_dead() or parent.fitness < min_fitness:
                # Then mark it as dead in the next generation.
                self.pop[e, g + 1, x, y, i].mark_dead()
            else:
                # Otherwise, make a child from the selected parent.
                child = Individual(parent.bitstr)

                # Maybe pick a mate and perform crossover
                if crossover_enabled and ti.random() < CROSSOVER_RATE:
                    m = tournament_selection(self.pop, e, g, x, y, min_fitness)
                    if m >= 0:
                        mate = self.pop[e, g, x, y, m]
                        child.bitstr = crossover(parent.bitstr, mate.bitstr)

                # Apply mutation to new child
                child.bitstr ^= mutation()

                # Place the child in the next generation
                self.pop[e, g + 1, x, y, i] = child

    def propagate(self, environment, generation, migrate_enabled, crossover_enabled):
        # TODO: Find a better way to do this! Using the next generation as a
        # scractch space like this is ugly and inefficient, but it was a
        # simple way to fix a race condition. When redesigning the propagation
        # process, redesign this code to be more streamlined.
        self.clear_scratch_space(generation)
        if migrate_enabled:
            self.migrate(generation)
        self.commit_scratch_space(generation)

        self.clear_scratch_space(generation)
        self.refill_empty_spaces(generation)
        self.commit_scratch_space(generation)

        self.populate_children(environment, generation, crossover_enabled)

    def evolve(self, environments, migration, crossover):
        self.randomize()
        for generation in range(INNER_GENERATIONS):
            self.evaluate(environments, generation)
            if generation + 1 < INNER_GENERATIONS:
                self.propagate(environments, generation, migration, crossover)

    # Taichi fields don't support slicing, and the pop field can get to be so
    # big that the to_numpy() method causes an OOM error! So, unfortunately we
    # need this awkward kernel to copy just the data we need from the field
    # into a set of numpy arrays (there's no way to use a single ndarray for a
    # dataclass like Individual with Taichi, either).
    @ti.kernel
    def get_logs_kernel(self, e: int,
                        bitstr: ti.types.ndarray(),
                        fitness: ti.types.ndarray(),
                        hiff: ti.types.ndarray(),
                        one_count: ti.types.ndarray()):
        ne, ig, ew, eh, cc = self.pop.shape
        for g, x, y, i in ti.ndrange(ig, ew, eh, cc):
            individual = self.pop[e, g, x, y, i]
            bitstr[g, x, y, i] = individual.bitstr
            fitness[g, x, y, i] = individual.fitness
            hiff[g, x, y, i] = individual.hiff
            one_count[g, x, y, i] = individual.one_count

    def get_logs(self, env_index):
        # Grab just the logs we need from the GPU...
        shape = self.pop.shape[1:]
        bitstr = np.zeros(shape, dtype=np.uint64)
        fitness = np.zeros(shape, dtype=np.float32)
        hiff = np.zeros(shape, dtype=np.int16)
        one_count = np.zeros(shape, dtype=np.uint8)
        self.get_logs_kernel(env_index, bitstr, fitness, hiff, one_count)

        # Make a data frame and annotate it with the premade index.
        return pl.DataFrame({
            'bitstr': bitstr.flatten(),
            'fitness': fitness.flatten(),
            'alive': ~np.isnan(fitness.flatten()),
            'hiff': hiff.flatten(),
            'one_count': one_count.flatten(),
        }).hstack(
            self.index
        )
