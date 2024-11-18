import polars as pl

import constants
from constants import OUTPUT_PATH, INNER_GENERATIONS

constants.TOURNAMENT_SIZE = 10

from environment import make_baym, make_flat
from chart import chart_hiff_dist
from evolve import evolve
from inner_population import InnerPopulation
from render import save_hiff_map


def selection_pressure():
    variants = {
        'baym': make_baym,
        'flat': make_flat,
    }
    for name, make_env in variants.items():
        path = OUTPUT_PATH / 'selection_pressure' / name
        path.mkdir(exist_ok=True, parents=True)

        env = make_env()
        inner_population = InnerPopulation()
        inner_log, _ = evolve(inner_population, env, True, True)
        last_gen = inner_log.filter(
            pl.col('Generation') == INNER_GENERATIONS - 1
        )

        save_hiff_map(path, name, last_gen)
        chart_hiff_dist(path, name, inner_log)


if __name__ == '__main__':
    selection_pressure()
