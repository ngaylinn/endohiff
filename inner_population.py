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
    BITSTR_DTYPE, CARRYING_CAPACITY, ENVIRONMENT_SHAPE, INNER_GENERATIONS)
from inner_fitness import weighted_hiff, count_ones
from reproduction import mutation, crossover, TournamentArena


@ti.dataclass
class Params:
    migration_rate: ti.float16
    crossover_rate: ti.float16
    mortality_rate: ti.float16
    max_fertility: ti.int8
    tournament_size: ti.int8


def get_default_params(shape=1):
    field = Params.field(shape=shape)
    field.migration_rate.fill(1.0)
    field.crossover_rate.fill(0.5)
    field.mortality_rate.fill(0.0)
    field.max_fertility.fill(25)
    field.tournament_size.fill(2)
    return field


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

        # Per generation metadata that does not get saved.
        # Selections lets us pick fit, living individuals from pop and remember
        # the results as we proceed to modify pop.
        self.arena = TournamentArena(self.pop)
        # How many children were produced from each individual in the
        # population, used to enforce a maximum count.
        self.num_children = ti.field(int, shape=(ne, ew, eh, cc))

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
    def evaluate(self, g: ti.i32, environment: ti.template()):
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
    def cull(self, g: int, environment: ti.template(), params: ti.template()):
        for e, x, y, i in ti.ndrange(*self.shape):
            individual = self.pop[e, g, x, y, i]
            min_fitness = environment.min_fitness[e, x, y]
            unfit = individual.is_dead() or individual.fitness < min_fitness
            unlucky = ti.random() < params[e].mortality_rate
            # Update the next generation to indicate which individuals from
            # this generation survived.
            if unfit or unlucky:
                self.pop[e, g + 1, x, y, i].mark_dead()
            else:
                self.pop[e, g + 1, x, y, i] = individual

    @ti.kernel
    def select(self, g: int, params: ti.template()):
        for e, x, y, i in ti.ndrange(*self.shape):
            # For each unit of carrying capacity, use tournament selection to
            # find one alive and relatively fit individual who is vying for
            # that carrying capacity. They might try to mate with whoever's
            # already there, or reproduce into that spot themselves there if
            # it's vacant. Note, we do selection on generation g+1 because
            # that's how we know who survived the cull() operation, but we
            # should look at generation g to find those individuals, since g+1
            # will hold their children, which will be mutated.
            self.selections[e, x, y, i] = tournament_selection(
                self.pop, e, g + 1, x, y, params[e].tournament_size)
            # Count how many children each individual has in the current
            # generation, so we can enforce a maximum number.
            self.num_children[e, x, y, i] = 0

    @ti.func
    def get_child(self, parent, e, g, x, y, i, m, params):
        # Add one to the children so far, using atomic add to ensure thread
        # safety. The return value is unique to each thread.
        num_children = ti.atomic_add(self.num_children[e, x, y, i], 1)
        child = Individual()
        # If this parent hasn't had too many already, generate a new child.
        if num_children + 1 < params[e].max_fertility:
            # Do crossover if a mate index was specified.
            if m >= 0:
                # Lookup the mate from generation g to populate a child in
                # generation g+1.
                mate = self.pop[e, g, x, y, m]
                child.bitstr = crossover(parent.bitstr, mate.bitstr)
            else:
                child.bitstr = parent.bitstr
            child.bitstr ^= mutation()
        # Otherwise, return a NULL child.
        else:
            child.mark_dead()
        return child

    @ti.kernel
    def reproduce(self, g: int, params: ti.template()):
        for e, x, y, i in ti.ndrange(*self.shape):
            # If there should be a living individual here at generation g+1
            if self.pop[e, g + 1, x, y, i].is_alive():
                # Then lookup the parent from generation g
                parent = self.pop[e, g, x, y, i]
                # Maybe use a mate for crossover.
                m = -1
                if ti.random() < params[e].crossover_rate:
                    m = self.arena.selections[e, x, y, i]
                # Generate a child and populate them into this location in g+1.
                child = self.get_child(parent, e, g, x, y, i, m, params)
                self.pop[e, g + 1, x, y, i] = child

    @ti.kernel
    def spread(self, g: int, params: ti.template()):
        for e, x, y, i in ti.ndrange(*self.shape):
            # If this location would be empty at generation g+1, try to fill it.
            if self.pop[e, g + 1, x, y, i].is_dead():
                # Find the location of the parent who may produce a child here.
                # By default, we draw a parent from this current location.
                px, py = x, y

                # If we're migration is enabled, consider taking a parent from a
                # nearby location instead.
                dx = ti.round(params[e].migration_rate * ti.randn())
                dy = ti.round(params[e].migration_rate * ti.randn())
                if dx != 0 or dy != 0:
                    px = int(ti.math.clamp(x + dx, 0, ENVIRONMENT_SHAPE[0] - 1))
                    py = int(ti.math.clamp(y + dy, 0, ENVIRONMENT_SHAPE[1] - 1))

                # Lookup a selected individual from the chosen location in the
                # previous generation, if there was any.
                pi = self.arena.selections[e, px, py, i]
                if pi > -1:
                    # Generate a child and place it into generation g+1.
                    parent = self.pop[e, g, px, py, pi]
                    child = self.get_child(parent, e, g, px, py, pi, -1, params)
                    self.pop[e, g + 1, x, y, i] = child

    def evolve(self, environments, params):
        self.randomize()
        for generation in range(INNER_GENERATIONS):
            # Look at all the bitstrings and determine their fitness, given
            # their local environmental conditions.
            self.evaluate(generation, environments)
            if generation + 1 < INNER_GENERATIONS:
                # Look at individuals in the current generation, see who lives,
                # and indicate that in the population in the next generation.
                self.cull(generation, environments, params)
                # Select fitter individuals from the ones that survived using
                # tournament selection.
                self.arena.select_all(generation + 1, params)
                # Individuals that survived replace themselves with children,
                # possibly crossing over with a selected mate.
                self.num_children.fill(0)
                self.reproduce(generation, params)
                # Empty spaces may get refilled by fit individuals in this or
                # nearby locations producing additional children.
                self.spread(generation, params)

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
