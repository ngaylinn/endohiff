"""Evolve environments for all hyperparmeter values.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import taichi as ti
from tqdm import trange

from src.bitstrings.population import make_params_field, BitstrPopulation
from src.constants import NUM_TRIALS, ENV_GENERATIONS, ENV_POPULATION_SIZE
from src.environments.population import EnvPopulation
from src.environments.util import make_flat
from src.sweep import SWEEP_KINDS, SWEEP_SIZE, SWEEP_SHAPE, Sweep


def main(path, sweep_kind):
    ti.init(ti.cuda)
    path.mkdir(exist_ok=True, parents=True)
    sweep = Sweep(sweep_kind)

    # This number must be some whole multiple of ENV_POPULATION_SIZE because
    # otherwise we'd need more sophisticated logic to score and propagate the
    # CPPN population. This is the max that fit on our workstation.
    sims_per_batch = NUM_TRIALS * ENV_POPULATION_SIZE

    # GPU data allocations for evolution.
    env_pop = EnvPopulation(NUM_TRIALS)
    bitstr_pop = BitstrPopulation(sims_per_batch)
    params_field = make_params_field(shape=sims_per_batch)
    best_envs = make_flat(SWEEP_SHAPE)

    print('Evolving environments for all parameters...')
    progress = trange(SWEEP_SIZE * SWEEP_SIZE * ENV_GENERATIONS)
    # For each hyperparameter setting...
    for params_data, sweep_index, sample in sweep.iter():
        # Evolve populations for ENV_GENERATIONS.
        env_pop.randomize()
        for og in range(ENV_GENERATIONS):
            # In each environment generation, we evolve a full population of
            # bitstrings then assess their final fitness score.
            env_field = env_pop.make_environments()
            params_field.from_numpy(params_data.repeat(sims_per_batch))
            bitstr_pop.evolve(env_field, params_field)
            env_pop.evaluate_fitness(bitstr_pop, og)
            if og + 1 < ENV_GENERATIONS:
                env_pop.propagate(og)
            progress.update()

        # Grab the final population of rendered environments, and their
        # associated fitness scores. Save the best environment for each
        # hyperparameter setting.
        env_data = env_field.to_numpy()
        final_fitness = env_pop.matchmaker.fitness.to_numpy()[-1]
        best_env_index = final_fitness.argmax()
        best_envs[sweep_index] = env_data[best_env_index]

        # If this hyperparameter setting is a sampel point, capture the full
        # environment evolution logs as well as the final evolved environments
        # for all trials.
        if sample:
            sample_path = path / sweep.summary(*sweep_index) / 'cppn'
            sample_path.mkdir(exist_ok=True, parents=True)
            env_pop.get_logs().write_parquet(sample_path / 'env_log.parquet')
            best_per_trial = final_fitness.argmax(axis=1)
            for t, i in enumerate(best_per_trial):
                e = t * ENV_POPULATION_SIZE + i
                np.save(sample_path / f'cppn_{t}.npy', env_data[e])

        progress.update()

    np.save(path / 'cppn_envs.npy', best_envs)

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Evolve environments across many hyperparameter settings.')
    parser.add_argument(
        'path', type=Path,
        help=f'Path for all sweep data')
    parser.add_argument(
        'sweep_kind', type=str, choices=SWEEP_KINDS,
        help=f'Which kind of sweep to run (one of {SWEEP_KINDS})')
    args = vars(parser.parse_args())

    sys.exit(main(**args))

