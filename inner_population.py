import taichi as ti
import pandas as pd
import numpy as np

from constants import (
    BITSTR_DTYPE, CARRYING_CAPACITY, ENVIRONMENT_SHAPE, INNER_GENERATIONS,
    MAX_POPULATION_SIZE, REFILL_RATE, BITSTR_LEN)
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
        self.next_id[None] = ti.cast(1, ti.uint32)
        self.fitness_values = ti.Vector.field(1, dtype=ti.float32, shape=(CARRYING_CAPACITY,))  # Temporary field for fitness values

        # holding diversity over generations
        self.pop_diversity = ti.field(dtype=ti.f32, shape=(INNER_GENERATIONS,))
        self.pop_sum_fitness = ti.field(dtype=ti.f32, shape=(INNER_GENERATIONS,))  # Sum of all the fitnesses
        self.pop_sum_genetic = ti.field(dtype=ti.f32, shape=(INNER_GENERATIONS,))  # Sum of all the fitnesses

        self.fitness_diversity = ti.field(dtype=ti.f32, shape=(INNER_GENERATIONS,))
        self.genetic_diversity = ti.field(dtype=ti.f32, shape=(INNER_GENERATIONS,))
        self.spatial_genetic_diversity = ti.field(dtype=ti.f32, shape=ENVIRONMENT_SHAPE) #new for genetic diversity

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
                bitstr=ti.cast(ti.random(int), BITSTR_DTYPE),
                id=self.get_next_id(x, y, i), parent=0, fitness=0.0, hiff=0)

        # Keep assigning new ids instead of reusing them across experiments.
        self.next_id[None] += ti.static(MAX_POPULATION_SIZE)

        # Reset all the metrics we're tracking to 0 for a new experiment.
        self.fitness_values.fill(0.0)
        self.pop_diversity.fill(0.0)
        self.pop_sum_fitness.fill(0.0)
        self.pop_sum_genetic.fill(0.0)
        self.fitness_diversity.fill(0.0)
        self.genetic_diversity.fill(0.0)
        self.spatial_genetic_diversity.fill(0.0)

    @ti.func
    def hamming_distance(self, individual1: ti.template(), individual2: ti.template()) -> ti.i32:
        # Evaluates the Hamming distance between 2 different bitstrings
        distance = 0
        # Iterate over all bits in the bitstring (64 bits)
        for bit in range(BITSTR_LEN):
            # Use bitwise shift and bitwise AND to check each individual bit
            bit1 = (individual1.bitstr >> bit) & 1
            bit2 = (individual2.bitstr >> bit) & 1
            if bit1 != bit2:
                distance += 1
        return distance

    @ti.func
    def evaluate_fitness_diversity(self, fitness_sum: ti.f32, fitness_squared_sum: ti.f32, count: ti.i32) -> ti.types.vector(2, ti.f32):
        fitness_diversity = 0.0
        if count > 0:
            average_fitness = fitness_sum / count
            variance = (fitness_squared_sum / count) - (average_fitness ** 2)
            fitness_diversity = ti.sqrt(variance)
        return ti.Vector([fitness_diversity, fitness_sum])

    @ti.func
    def evaluate_genetic_diversity(self, distance_sum: ti.f32, distance_squared_sum: ti.f32, distance_count: ti.i32) -> ti.types.vector(2, ti.f32):
        genetic_diversity = 0.0
        if distance_count > 0:
            average_genetic = distance_sum / distance_count
            variance = (distance_squared_sum / distance_count) - (average_genetic ** 2)
            genetic_diversity = ti.sqrt(variance) #sqrt of the variance
        return ti.Vector([genetic_diversity, distance_sum])


    @ti.kernel
    def evaluate(self, environment: ti.template(), g: ti.i32):
        # For computing fitness diversity:
        fitness_sum = 0.0
        fitness_squared_sum = 0.0
        fitness_count = 0

        # For computing genetic diversity
        distance_sum = 0.0
        distance_squared_sum = 0.0
        distance_count = 0

        for x, y, i in ti.ndrange(*self.shape):
            fitness, hiff = ti.cast(0.0, ti.float32), ti.cast(0, ti.uint32)
            # Only evaluate fitness of individuals that are alive.
            individual = self.pop[g, x, y, i]
            if individual.id != DEAD_ID:
                fitness, hiff = weighted_hiff(
                    individual.bitstr, environment.weights[x, y])
                fitness_sum += fitness
                fitness_squared_sum += fitness ** 2  # Sum of squares for std deviation calculation
                fitness_count += 1
                # Compare each individual to every other individual in this
                # cell to figure out the genetic diversity there.
                for i2 in range(self.shape[2]):
                    other = self.pop[g, x, y, i2]
                    if i != i2 and other.id != DEAD_ID:
                        # Genetic similarity determined by hamming distance.
                        hamming_dist = self.hamming_distance(individual, other)
                        distance_sum += hamming_dist
                        distance_squared_sum += hamming_dist ** 2
                        distance_count += 1
                self.fitness_values[i][0] = fitness  # Store fitness value in the temporary field
            self.pop[g, x, y, i].fitness = fitness
            self.pop[g, x, y, i].hiff = hiff

        # calc fitness diversity
        fitness_diversity_result = self.evaluate_fitness_diversity(fitness_sum, fitness_squared_sum, fitness_count)
        self.fitness_diversity[g] = fitness_diversity_result[0]
        self.pop_sum_fitness[g] = fitness_diversity_result[1]

        # calc genetic diversity
        genetic_diversity_result = self.evaluate_genetic_diversity(distance_sum, distance_squared_sum, distance_count)
        self.genetic_diversity[g] = genetic_diversity_result[0]
        self.pop_sum_genetic[g] = genetic_diversity_result[1]

        # TODO: calc other whole-pop by-generation stuff?? 
        

    # @ti.kernel
    # def migrate(self, g: int):
    #     # TODO: Try out different migration strategies!
    #     # Ways to do migration:
    #     # - When does migration happen?
    #     #   - Before / after reproduction?
    #     #   - # of migrations per generation?
    #     # - How do individuals move?
    #     #   - "Acorn drop"
    #     #   - "Walk" to adjacent cell
    #     #   - Random shuffling
    #     #   - Move, or clone?
    #     # - How do you resolve competition?
    #     #   - Reserve some spots for migrants
    #     #   - Fill empty spots first
    #     #   - Let migrants overwrite locals
    #     #   - Replace local only if fitter
    #     # - How selective?
    #     #   - Random / more fit individuals migrate
    #     #   - Replace random / less fit individuals
    #     for x, y, i in ti.ndrange(*self.shape):
    #         # This magic number was calculated by brute force such that two
    #         # samples from a standard normal distribution will be (0, 0) 90% of
    #         # the time (meaning, migration happens 10% of the time).
    #         scale_factor = 0.5

    #         # "Acorn drop": draw from a standard normal distribution to find a
    #         # new home for this individual.
    #         dx = ti.round(scale_factor*ti.randn())
    #         dy = ti.round(scale_factor*ti.randn())

    #         # Make sure the new location is in bounds. This might cause
    #         # migrants to "pile up" at the edges of the environment.
    #         new_x = int(ti.math.clamp(x + dx, 0, ENVIRONMENT_SHAPE[0]))
    #         new_y = int(ti.math.clamp(y + dy, 0, ENVIRONMENT_SHAPE[1]))
    #         new_i = ti.random(ti.int32) % CARRYING_CAPACITY

    #         if new_x != x and new_y != y:
    #             # Move this individual to a random spot in a nearby location,
    #             # possibly overwriting an existing individual there. Then,
    #             # leave this spot empty.
    #             individual = self.pop[g, x, y, i]
    #             self.pop[g, new_x, new_y, new_i] = individual
    #             self.pop[g, x, y, i] = DEAD


    # def migrate_experiment(self, gen):
    #     if self.migration_strategy == 'migrate_acorn_drop':
    #         # Call the migrate_acorn_drop function
    #         self.migrate_acorn_drop(gen)       
    #     elif self.migration_strategy == 'migrate_walk':
    #         self.migrate_walk(gen)
    #     elif self.migration_strategy == "migrate_overwriting":
    #         self.migrate_overwriting(gen)
    #     else:
    #         self.migrate_selective(gen)

    # @ti.kernel
    # def migrate_acorn_drop(self, g: int):
    #     # baseline migration strategy - same as migrate() above 
    #     for x, y, i in ti.ndrange(*self.shape):
    #         scale_factor = 0.5
    #         dx = ti.round(scale_factor * ti.randn())
    #         dy = ti.round(scale_factor * ti.randn())
            
    #         new_x = int(ti.math.clamp(x + dx, 0, ENVIRONMENT_SHAPE[0]))
    #         new_y = int(ti.math.clamp(y + dy, 0, ENVIRONMENT_SHAPE[1]))
    #         new_i = ti.random(ti.int32) % CARRYING_CAPACITY

    #         if new_x != x and new_y != y:
    #             individual = self.pop[g, x, y, i]
    #             self.pop[g, new_x, new_y, new_i] = individual
    #             self.pop[g, x, y, i] = DEAD

    # @ti.kernel
    # def migrate_walk(self, g: int):
    #     # This strategy involves walking individuals to an adjacent cell (left, right, up, or down) with a 10% chance of migration.
    #     for x, y, i in ti.ndrange(*self.shape):
    #         # new x, y, i init
    #         new_x = x
    #         new_y = y
    #         new_i = i
    #         if ti.random() < 0.1:  # 10% chance to migrate
    #             direction = ti.random(ti.i32) % 4  # Choose a random direction
                
    #             # Determine new_x and new_y based on direction
    #             if direction == 0:  # Move left
    #                 new_x = max(x - 1, 0)
    #                 new_y = y
    #             elif direction == 1:  # Move right
    #                 new_x = min(x + 1, ENVIRONMENT_SHAPE[0] - 1)
    #                 new_y = y
    #             elif direction == 2:  # Move up
    #                 new_x = x
    #                 new_y = max(y - 1, 0)
    #             else:  # Move down
    #                 new_x = x
    #                 new_y = min(y + 1, ENVIRONMENT_SHAPE[1] - 1)
                
    #             # Choose a random individual index to migrate to the new position
    #             new_i = ti.random(ti.i32) % CARRYING_CAPACITY

    #             if new_x != x and new_y != y:
    #                 individual = self.pop[g, x, y, i]
    #                 self.pop[g, new_x, new_y, new_i] = individual
    #                 self.pop[g, x, y, i] = DEAD


    @ti.kernel
    def migrate_overwriting(self, g: int):
    # Migrants replace local individuals only if they have a higher fitness than the current inhabitant. 
    # This ensures that more fit individuals are prioritized for migration.
        for x, y, i in ti.ndrange(*self.shape):
            scale_factor = 0.5 #same gaussian dist as baseline acorn drop -- but now checks the prev. inhabitant's fitness
            dx = ti.round(scale_factor * ti.randn())
            dy = ti.round(scale_factor * ti.randn())
            
            new_x = int(ti.math.clamp(x + dx, 0, ENVIRONMENT_SHAPE[0]))
            new_y = int(ti.math.clamp(y + dy, 0, ENVIRONMENT_SHAPE[1]))
            new_i = ti.random(ti.int32) % CARRYING_CAPACITY

            if new_x != x and new_y != y:
                individual = self.pop[g, x, y, i]
                current_resident = self.pop[g, new_x, new_y, new_i]

                # Replace resident only if the migrant has higher fitness
                if individual.fitness > current_resident.fitness:
                    self.pop[g, new_x, new_y, new_i] = individual
                    self.pop[g, x, y, i] = DEAD

    # @ti.kernel
    # def migrate_selective(self, g: int):
    # # migrant selected by tournament selection
    #     # selects the fittest indivs for migration
    #     for x, y, i in ti.ndrange(*self.shape):
    #         # Magic number: 10% chance of migration
    #         scale_factor = 0.5

    #         # "Acorn drop": use a normal distribution to find a new home for this individual
    #         dx = ti.round(scale_factor * ti.randn())
    #         dy = ti.round(scale_factor * ti.randn())

    #         # Ensure the new location is within bounds
    #         new_x = int(ti.math.clamp(x + dx, 0, ENVIRONMENT_SHAPE[0]))
    #         new_y = int(ti.math.clamp(y + dy, 0, ENVIRONMENT_SHAPE[1]))
    #         new_i = ti.random(ti.int32) % CARRYING_CAPACITY

    #         if new_x != x or new_y != y:
    #             # competing resident location
    #             individual = self.pop[g, x, y, i] 

    #             # Perform tournament selection to choose a migrant from the current location
    #             migrant = tournament_selection(self.pop, g, new_x, new_y)

    #             # Migrate or replace: replace an individual only if the migrant is fitter
    #             if migrant.fitness > individual.fitness:
    #                 # Overwrite the local individual with the migrant
    #                 self.pop[g, new_x, new_y, new_i] = migrant
    #                 self.pop[g, x, y, i] = DEAD
    #             else:
    #                 # If the migrant is less fit, do not overwrite
    #                 self.pop[g, new_x, new_y, new_i] = individual
    #                 self.pop[g, x, y, i] = DEAD

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
                        # If you pick a DEAD individual, try a few more times.
                        # This makes it significantly easier for a migrant to
                        # colonize an empty cell.
                        for _ in range(4):
                            new_index = ti.random(int) %  CARRYING_CAPACITY
                            individual = self.pop[g, x, y, new_index]
                            if individual.id != DEAD_ID:
                                break
                    else:
                        individual = tournament_selection(self.pop, g, x, y)

            self.pop[g, x, y, i] = individual


    @ti.kernel
    def populate_children(self, environment: ti.template(), g: int, crossover: bool):
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
                if crossover:
                    child.bitstr = diverse_crossover(parent.bitstr, mate.bitstr) #CHANGED TO UNIFORM_CROSSOVER
                else:
                    child.bitstr = parent.bitstr

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


    def propagate(self, environment, generation, migrate, crossover):
        if migrate:
            self.migrate_overwriting(generation)
        self.refill_empty_spaces(generation)
        self.populate_children(environment, generation, crossover)


    def to_numpy(self):
        return self.pop.to_numpy()



