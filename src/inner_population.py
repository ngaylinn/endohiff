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
    BITSTR_DTYPE, BITSTR_LEN, CARRYING_CAPACITY, CROSSOVER_RATE,
    ENVIRONMENT_SHAPE, INNER_GENERATIONS)
from inner_fitness import score_hiff
from reproduction import mutation, crossover, TournamentArena


@ti.dataclass
class Params:
    migration_rate: ti.float16
    mortality_rate: ti.float16
    fertility_rate: ti.int8
    tournament_size: ti.int8


PARAMS_DTYPE = np.dtype([
    ('migration_rate', np.float16),
    ('mortality_rate', np.float16),
    ('fertility_rate', np.int8),
    ('tournament_size', np.int8),
])


def get_default_params(shape=(1,)):
    field = Params.field(shape=shape)
    field.migration_rate.fill(1.0)
    field.mortality_rate.fill(0.125)
    field.fertility_rate.fill(25)
    field.tournament_size.fill(6)
    return field


def get_default_params_numpy(shape=(1,)):
    params = np.empty(shape, dtype=PARAMS_DTYPE)
    params['migration_rate'] = 1.0
    params['mortality_rate'] = 0.125
    params['fertility_rate'] = 25
    params['tournament_size'] = 6
    return params


@ti.dataclass
class Individual:
    bitstr: BITSTR_DTYPE
    fitness: ti.float16

    @ti.func
    def is_dead(self):
        return ti.math.isnan(self.fitness)

    @ti.func
    def is_alive(self):
        return not self.is_dead()

    @ti.func
    def mark_dead(self):
        self.fitness = ti.cast(ti.math.nan, ti.float16)


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
    def evaluate(self, g: ti.i32):
        for e, x, y, i in ti.ndrange(*self.shape):
            # Only evaluate fitness of individuals that are alive.
            individual = self.pop[e, g, x, y, i]
            if individual.is_alive():
                individual.fitness = score_hiff(individual.bitstr)
                self.pop[e, g, x, y, i] = individual

    @ti.kernel
    def cull(self, g: int, environment: ti.template(), params: ti.template()):
        for e, x, y, i in ti.ndrange(*self.shape):
            individual = self.pop[e, g, x, y, i]
            min_fitness = environment[e, x, y]
            unfit = individual.is_dead() or individual.fitness < min_fitness
            unlucky = ti.random() < params[e].mortality_rate
            # Update the next generation to indicate which individuals from
            # this generation survived.
            if unfit or unlucky:
                self.pop[e, g + 1, x, y, i].mark_dead()
            else:
                self.pop[e, g + 1, x, y, i] = individual

    @ti.func
    def get_child(self, parent, e, g, x, y, i, m, params):
        # Add one to the children so far, using atomic add to ensure thread
        # safety. The return value is unique to each thread.
        num_children = ti.atomic_add(self.num_children[e, x, y, i], 1)
        child = Individual()
        # If this parent hasn't had too many already, generate a new child.
        if num_children + 1 < params[e].fertility_rate:
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
                if ti.random() < CROSSOVER_RATE:
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
            self.evaluate(generation)
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
                        fitness: ti.types.ndarray()):
        ne, ig, ew, eh, cc = self.pop.shape
        for g, x, y, i in ti.ndrange(ig, ew, eh, cc):
            individual = self.pop[e, g, x, y, i]
            bitstr[g, x, y, i] = individual.bitstr
            fitness[g, x, y, i] = individual.fitness

    def get_logs(self, env_index):
        # Grab just the logs we need from the GPU...
        shape = self.pop.shape[1:]
        bitstr = np.zeros(shape, dtype=np.uint64)
        fitness = np.zeros(shape, dtype=np.float16)
        self.get_logs_kernel(env_index, bitstr, fitness)

        # Make a data frame and annotate it with the premade index.
        return pl.DataFrame({
            'bitstr': bitstr.ravel(),
            'fitness': fitness.ravel(),
            'alive': ~np.isnan(fitness.ravel()),
        }).hstack(
            self.index
        )


# A demo to show that evolution is working.
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from environments import make_baym, make_field
    from visualize_inner_population import render_fitness_map

    ti.init(ti.cuda)

    env = make_field()
    env.from_numpy(make_baym())
    params = get_default_params()
    inner_population = InnerPopulation()

    inner_population.evolve(env, params)
    inner_log = inner_population.get_logs(0).filter(
        pl.col('Generation') == INNER_GENERATIONS - 1
    )
    render_fitness_map(inner_log)
    plt.show()
