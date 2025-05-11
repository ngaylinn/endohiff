# Supplementary Information

This file is an overview of the results presented in *Ramps and Ratchets: Evolving Spatial Viability Landscapes*, a paper for [ALife 2025](https://www.alife-japan.org/alife-2025), including additional figures that could not fit in the original paper.

***This is a work in progress! The figures here are correct, but the styling and formatting needs work, as do the explanations for what these images mean. Many improvements coming soon.***

## Introduction

This experiment evolves bit strings to solve a fitness function named HIFF, but it does so within a 2D spatial environment, where each location has a minimum fitness score needed to survive and reproduce (a "viability threshold"), which changes gradually. This is able to induce the selection pressure needed for the bit strings to evolve. We experiment with hand-designed environments, but also with environments we evolved using CPPNs.

![A diagram visualizing a population of bit strings as circles, colored by fitness score. The horizontal dimension is space, and the viability thresholds for each location are drawn as bars of increasing height. Some of the circles appear above the viability thresholds, and gradually increase in fitness as they move to the right. Some are below, and these are X'ed out to indicate less fit individuals are eliminated and do not reproduce.](figures/figure1.png)

Figure 1: In this experiment, individuals move through a spatial environment with varying thresholds for survival. (a) With a low threshold, individuals explore phenotype space. (b) A gradual ramp up allows a relatively fit individual to migrate to an empty deme, exploiting its genetic innovation by spawning many offspring. (c) Exploration resumes, but more mutations are fatal here. This creates a ratchet effect where fitness in this deme can increase, but cannot drop be- low the threshold, driving local mean fitness up. (d) The process repeats at the next higher viability threshold.

## Hyperparameters

All results from the paper and in this file were generated using the following hyperparameters, unless otherwise stated:

| Hyperparameter | Value | Description |
|---|:---:|:---|
| `tournament_size` | 6 | How many individuals are considered when doing selection. This determines how much selection depends on fitness. 1 means individuals are chosen at random, while 25 means only the most fit individual is chosen.|
| `mortality_rate` | 0.125 | How likely are individuals to be eliminated at ranom each generation. This determines how often selection acts on the population. |
| `migration_rate` | 1.0 | How far away an individual's offspring can be. This is a multiplier for a gaussian distribution centered on the current location, used to determine the location of each offspring.|
| `fertility_rate` | 25 | The maximum number of offspring each individual can have each generation.|
| `replications` | 5 | The number of times each experiment is run, to account for the natural variability in this experiment.|
| `mutation_rate` | 0.015625 | How likely each bit in a genotype is to flip in each generation. This number is 2<sup>-6</sup>, since that is convenient to implement.|
| `environment_shape` | 64x36 demes | The shape of the 2D space where simulations happen. Each deme has a population of bit strings, which occasionally move according to `migration_rate`.|
| `carrying_capacity` | 25 per deme | How many individuals are in each of the isolated sub-populations.|
| `cppn_population_size` | 30 | How many CPPNs we maintain in our population in order to evolve environments.|
| `inner_generations` | 150 | How many generations we evolve the bit strings for. |
| `outer_generation` | 20 | How many generations we evolve the CPPNs / environments for. |
| `bit_string_length` | 64 | How long each bit string is. |

Note that we ran a hyperparameter sweep over 25x25 different settings for `tournament_size` and `mortality_rate`, as well as for `migration_rate` and `fertility_rate`. The other parameters were chosen by intuition, trial, and error.

## Simulations

We evolved bit strings using the default hyperparameters in two hand-designed environments, and in an environment evolved to produce high fitness scores with those hyperparameter settings. These results represent the best of five replications.
 
| Environment Name | Environment Design | Simulated Evolution |
|---|---|---|
| flat | ![A black rectangle](figures/flat_map.png) | ![Fuzzy deep purple noise gradually transitions to somewhat lighter shifting patterns of purple and red](figures/flat_fitness_map.gif)|
| baym | ![A series of vertical columns ranging from almost black on the left and right to almost white in the center](figures/baym_map.png) | ![Most of the middle goes black immediatley, but random purple activity happens at the edges, then repeatedly surges inwards getting more and more orange as it goes, until a patch of white forms near the middle and spreads](figures/baym_fitness_map.gif)|
| cppn | ![A series of diagonal lines, each with a short gradient going from black to white](figures/cppn_map.png) | ![Diagonal stripes of purple activity and black indicating empty space. The black area becomes red, then orange, then white in one region, which quickly spreads throughout the map.](figures/cppn_fitness_map.gif)|

These charts summarize the evolutionary trajectory of the bit string populations evolving in these three settings. The solid (dashed) lines represent the max (mean) fitness of the population, with the shade region indicating the full range of values across five replications.

|flat|baym|cppn|
|:---:|:---:|:---:|
|![Max fitness grows for about 50 generations then plateaus around 275. Mean fitness follows a similar trajectory, but flattens out at around 150](figures/flat_fitness.png) |![Max fitness grows for about 50 generations then plateaus with an everage value around 360 and a maximum value of 448. Mean fitness follows a similar trajectory, but flattens out at around 250](figures/baym_fitness.png) |![Max fitness grows at a roughly linear rate, achieving a max score of 448 in one replication around generation 80 but continues to grow as the other replications get higher scores. Mean fitness follows a similar trajectory, reaching a little over 200 by the end of the simulation](figures/cppn_fitness.png)|


## Sample CPPNs

When evolving environments, we saw a wide variety of designs:

![A black rectangle](figures/cppn_1.png)
![Diagonals consisting of long linear gradients from black to white offset from one another](figures/cppn_2.png)
![A mostly white image with a small horizontal band near the top that fades to dark blue and back to white](figures/cppn_3.png)
![A smooth horizontal gradient going from purple to light blue](figures/cppn_4.png)
![A pattern of three horizontal gradients along the bottom of the image. Further up the image, this pattern distorts and bends until the three gradients seem to appear at uncorrelated horizontal locations on each line](figures/cppn_5.png)
![A series of light blue vertical bands, alternating with black bands](figures/cppn_6.png)
![A diagonal gradient ranging from black (top middle to bottom right) to white (bottom left corner)](figures/cppn_7.png)
![A horizontal gradient going from black / dark purple at the top and bottom to white in the middle](figures/cppn_8.png)
![A purple rectangle](figures/cppn_9.png)
![A black rectangle with a tiny diagonal gradient going to white in the upper left corner](figures/cppn_10.png)

## Breaking the ramp

Our experiment showed that an accessible ramp-up in difficulty is important for inducing selection pressure. To illustrate the necessary and sufficient conditions for this effect, we demonstrate several ways to break it.

Changing the gradual ramp-up of the baym environment prevents bit strings from migrating to the center:

| | Intermediate steps removed | Lower steps removed |
| --- | :---: | :---: |
| Environment | ![A variation of the baym environment that transitions abruptly from the low steps to the high steps with no intermediate steps in between](figures/gap_map.png) | ![A variation of the baym environment that removes the lower steps, so the graduation goes from relatively high to higher](figures/high_map.png)|
| Final population |  ![Purplish red activity appears on the edges of the map, not penetrating beyond the first two steps](figures/gap_fitness_map.png) |![A black space indicates total extinction](figures/high_fitness_map.png)|

Changing the hyperparameter values:
| | Low `fertility_rate` | Low `migration_rate` |
| --- | :---: | :---: |
| Final population | ![Purplish red activity is strong on the left and right edges, but the population quickly thins out as it moves towards the middle and does not achieve high fitness](figures/low_fertility.png) | ![Purplish red activity is strong on the left and right edges, but on the second step only about half of the demes are populated and none are populated further towards the center](figures/low_migration.png)|

## Hyperparameter sweep

We did a hyperparameter sweep over `tournament_size` and `mortality_rate` which are the primary determiners of selection pressure.

| | flat | baym | cppn |
| --- | :---: | :---: | :---: |
| Absolute | ![Mostly light pink to white, with a large rounded region of the upper left corner fading to deep red and purple. The right edge is black, then red, before it reaches the middle pink / white region](figures/flat.png) | ![Speckled pink with red at the top and right edges, and black on the far right](figures/baym.png) | ![Mostly pink to white, with some slightly darker pink in the upper left corner, red and then black on the far right edge](figures/cppn.png) |
| Relative | | ![Mostly yellow / orange with a rounded corner of bright blue in the upper left](figures/baym_vs_flat.png)<br>baym vs. flat | ![Mostly yellow with speckles of orange / green and a rounded greenish curve just offset from the upper left corner](figures/cppn_vs_both.png) <br> cppn vs. both|

A histogram of fitness values across the hyperparameter sweep:

![A bar chart with buckets for fitness score ranges on the horizontal access, and number of examples in that bucket on the vertical axis.](figures/sweep_dist.png)

`tournament_size = 0.000, mortality_rate = 19`

![A blue rectangle](figures/cppn_0.000_19.png)

`tournament_size = 0.042, mortality_rate = 11`

![Diagonal stripes ranging from black to white](figures/cppn_0.042_11.png)

`tournament_size = 0.125, mortality_rate = 6`

![Diagonal stripes ranging from black to white](figures/cppn_0.125_6.png)

`tournament_size = 0.333, mortality_rate = 19`

![A black rectangle](figures/cppn_0.333_19.png)

`tournament_size = 0.500, mortality_rate = 11`

![A black rectangle with a slight deep purple gradient in the bottom third](figures/cppn_0.500_11.png)

`tournament_size = 0.667, mortality_rate = 19`

![A black rectangle](figures/cppn_0.667_19.png)

## Multi-Objective Optimization

We attempted to evolve environments that could reshape how the HIFF function gets evaluated and bias the solutions to have different mixes of 0 and 1 bits, as well as getting a high score. The results were hard to interpret, but it seems clear that this was doing something interesting:

| Condition | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Trial 5 |
| --- | --- | --- | --- | --- | --- |
| No preference | ![TODO](figures/cppn_any_0.png) | ![TODO](figures/cppn_any_1.png) | ![TODO](figures/cppn_any_2.png) | ![TODO](figures/cppn_any_3.png) | ![TODO](figures/cppn_any_4.png)
| Prefer bit strings with more 0's and ones with more 1's | ![TODO](figures/cppn_diverse_0.png) | ![TODO](figures/cppn_diverse_1.png) | ![TODO](figures/cppn_diverse_2.png) | ![TODO](figures/cppn_diverse_3.png) | ![TODO](figures/cppn_diverse_4.png)
| Prefer bit strings with a mix of 0's and 1's | ![TODO](figures/cppn_mixed_0.png) | ![TODO](figures/cppn_mixed_1.png) | ![TODO](figures/cppn_mixed_2.png) | ![TODO](figures/cppn_mixed_3.png) | ![TODO](figures/cppn_mixed_4.png)
| Prefer all bit strings to have more 0's or more 1's | ![TODO](figures/cppn_uniform_0.png) | ![TODO](figures/cppn_uniform_1.png) | ![TODO](figures/cppn_uniform_2.png) | ![TODO](figures/cppn_uniform_3.png) | ![TODO](figures/cppn_uniform_4.png)
