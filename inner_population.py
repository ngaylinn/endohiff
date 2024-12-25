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
    BITSTR_DTYPE, CARRYING_CAPACITY, CROSSOVER_RATE, DEAD_ID,
    ENVIRONMENT_SHAPE, INNER_GENERATIONS, MIGRATION_RATE, REFILL_RATE)
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

    @ti.func
    def get_id(self, e, g, x, y, i):
        ne, ig, ew, eh, cc = self.pop.shape
        # Return a unique number (>= 1) to identify this individual. The id 0
        # is reserved to indicate no living individual present.
        return 1 + i + cc * (y + eh * (x + ew * (g + ig * e)))

    @ti.kernel
    def randomize(self):
        # Randomize the population.
        for e, x, y, i in ti.ndrange(*self.shape):
            self.pop[e, 0, x, y, i] = Individual(
                bitstr=ti.random(BITSTR_DTYPE),
                id=self.get_id(e, 0, x, y, i))

    @ti.kernel
    def evaluate(self, environment: ti.template(), g: ti.i32):
        for e, x, y, i in ti.ndrange(*self.shape):
            fitness, hiff = ti.cast(0.0, ti.float32), ti.cast(0, ti.uint32)

            # Only evaluate fitness of individuals that are alive.
            individual = self.pop[e, g, x, y, i]
            if individual.id != DEAD_ID:
                fitness, hiff = weighted_hiff(
                    individual.bitstr, environment.weights[e, x, y])

            self.pop[e, g, x, y, i].fitness = fitness
            self.pop[e, g, x, y, i].hiff = hiff

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
            new_i = ti.random(ti.int32) % CARRYING_CAPACITY

            # If this individual is moving to a new location...
            if new_x != x or new_y != y:
                migrant = self.pop[e, g, x, y, i]
                local = self.pop[e, g, new_x, new_y, new_i]

                # If the migrant is more fit than the resident in the location
                # it's moving to, replace the local and leave an empty space.
                if migrant.fitness > local.fitness:
                    # To avoid race conditions, use the next generation as a
                    # scratch space, so this block of threads isn't reading and
                    # writing to the same memory location. After this kernel,
                    # call commit_scratch_space to copy the results back to the
                    # present generation.
                    self.pop[e, g + 1, new_x, new_y, new_i] = migrant
                    self.pop[e, g, x, y, i] = DEAD

    @ti.kernel
    def refill_empty_spaces(self, g: int):
        for e, x, y, i in ti.ndrange(*self.shape):
            individual = self.pop[e, g, x, y, i]

            # If this spot was unoccupied last generation...
            if individual.id == DEAD_ID:
                # Maybe let another individual spawn into this cell.
                if ti.random() < REFILL_RATE:
                    # Try a few times to find an individual in this location
                    # that's not dead, and let them take over the empty spot.
                    for _ in range(4):
                        new_i = ti.random(ti.int32) % CARRYING_CAPACITY
                        individual = self.pop[e, g, x, y, new_i]
                        if individual.id != DEAD_ID:
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
            self.pop[e, g + 1, x, y, i] = DEAD

    @ti.kernel
    def commit_scratch_space(self, g: int):
        for e, x, y, i in ti.ndrange(*self.shape):
            if self.pop[e, g + 1, x, y, i].id != DEAD_ID:
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

            # If no one in this location is fit to reproduce...
            if parent.id == DEAD_ID or parent.fitness < min_fitness:
                # Then mark it as dead in the next generation.
                self.pop[e, g + 1, x, y, i] = DEAD
            else:
                # Otherwise, make a child from the selected parent.
                child = Individual(
                    parent.bitstr, self.get_id(e, g + 1, x, y, i), parent.id
                )

                # Maybe pick a mate and perform crossover
                if crossover_enabled and ti.random() < CROSSOVER_RATE:
                    m = tournament_selection(self.pop, e, g, x, y, min_fitness)
                    if m >= 0:
                        mate = self.pop[e, g, x, y, m]
                        child.bitstr = crossover(parent.bitstr, mate.bitstr)
                        child.parent2 = mate.id

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

    def get_logs(self, env_index=None):
        def annotate_log_data(log_data, e):
            return pl.DataFrame({
                key: field_data[e].flatten()
                for key, field_data in log_data.items()
            }).hstack(
                self.index
            )

        # Return either the inner logs from a single environment, or a list of
        # logs from each environment.
        log_data = self.pop.to_numpy()
        if env_index is not None:
            return annotate_log_data(log_data, env_index)
        else:
            return [
                annotate_log_data(log_data, e)
                for e in range(self.num_environments)
            ]
