from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
from tqdm import trange

from ..bitstrings.population import make_params_field, BitstrPopulation
from ..constants import NUM_TRIALS, OUTER_GENERATIONS, OUTER_POPULATION_SIZE
from .population import EnvironmentPopulation
from .fitness import eval_env_fitness
from .util import make_flat
from ..sweep import SWEEP_KINDS, SWEEP_SIZE, SWEEP_SHAPE, Sweep


def main(sweep_kind, path):
    sweep = Sweep(sweep_kind)

    # This number must be some whole multiple of OUTER_POPULATION_SIZE because
    # otherwise we'd need more sophisticated logic to score and propagate the
    # CPPN population.
    sims_per_batch = NUM_TRIALS * OUTER_POPULATION_SIZE

    env_pop = EnvironmentPopulation(NUM_TRIALS)
    bitstr_pop = BitstrPopulation(sims_per_batch)
    params = make_params_field(shape=sims_per_batch)
    best_envs = make_flat(SWEEP_SHAPE)

    print('Evolving environments for all parameters...')
    progress = trange(SWEEP_SIZE * SWEEP_SIZE * OUTER_GENERATIONS)
    for params_data, sweep_index, sample in sweep.iter():
        env_pop.randomize()
        for og in range(OUTER_GENERATIONS):
            env = env_pop.make_environments()
            params.from_numpy(params_data.repeat(sims_per_batch))
            bitstr_pop.evolve(env, params)
            env_pop.evaluate_fitness(bitstr_pop, og)
            if og + 1 < OUTER_GENERATIONS:
                env_pop.propagate(og)
            progress.update()

        env_data = env.to_numpy()
        final_fitness = evaluator.fitness.to_numpy()[-1]
        best_env_index = final_fitness.argmax()
        best_envs[sweep_index] = env_data[best_env_index]

        if sample:
            sample_path = path / sweep.summary(*sweep_index) / 'cppn'
            sample_path.mkdir(exist_ok=True, parents=True)
            env_pop.get_logs().write_parquet(sample_path / 'outer_log.parquet')
            best_per_trial = final_fitness.argmax(axis=1)
            for t, i in enumerate(best_per_trial):
                e = t * OUTER_POPULATION_SIZE + i
                np.save(sample_path / f'cppn_{t}.npy', env_data[e])

        progress.update()

    np.save(path / 'cppn_envs.npy', best_envs)

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Evolve environments across many hyperparameter settings.')
    parser.add_argument(
        'sweep_kind', type=str, choices=SWEEP_KINDS,
        help=f'Which kind of sweep to run (one of {SWEEP_KINDS})')
    parser.add_argument(
        'path', type=Path,
        help=f'Path for all sweep data')
    args = vars(parser.parse_args())
    sys.exit(main(**args))

