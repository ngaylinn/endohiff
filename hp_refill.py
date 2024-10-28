import polars as pl
from tqdm import trange

from evolve import evolve
from inner_population import InnerPopulation
from run_experiments import CONDITIONS

# Repeat all experiments for improved statistical significance.
NUM_REPETITIONS = 10


def print_summary(df, file=None):
    # Print summaries...
    pl.Config(tbl_rows=len(CONDITIONS) * 3 * 2)
    print('Combined effect across experiments:', file=file)
    print(df, file=file)

    print(file=file)
    print('Isolated effect of refill rate across experiments:', file=file)
    print(df.group_by(
        'condition', 'refill_rate',
        maintain_order=True

    ).agg(
        pl.col('population_size').mean(),
        pl.col('hiff').mean()
    ), file=file)

    print(file=file)
    print('Isolated effect of random refill across experiments:', file=file)
    print(df.group_by(
        'condition', 'random_refill',
        maintain_order=True

    ).agg(
        pl.col('population_size').mean(),
        pl.col('hiff').mean()
    ), file=file)


def tune_refill():
    # Run lots of experimental variations...
    frames = []
    inner_population = InnerPopulation()
    progress = trange(len(CONDITIONS) * 3 * 2 * NUM_REPETITIONS)
    for name, make_environment in CONDITIONS.items():
        environment = make_environment()
        for refill_rate in [0.0, 0.5, 1.0]:
            for random_refill in [False, True]:
                for _ in range(NUM_REPETITIONS):
                    inner_population.refill_rate = refill_rate
                    inner_population.random_refill = random_refill
                    inner_log = evolve(inner_population, environment)
                    frames.append(inner_log.filter(
                        pl.col('generation') == 99
                    ).with_columns(
                        condition=pl.lit(name),
                        refill_rate=pl.lit(refill_rate),
                        random_refill=pl.lit(random_refill)
                    ))
                    progress.update()
    progress.close()

    # Aggregate results across all experiments...
    return pl.concat(
        frames
    ).group_by(
        'condition', 'refill_rate', 'random_refill',
        maintain_order=True
    ).agg(
        (pl.col('id') > 0).sum().alias('population_size'),
        pl.col('hiff').mean()
    )


if __name__ == '__main__':
    df = tune_refill()
    print_summary(df)
    with open('output/hp_refill_summary.txt', 'w') as file:
        print_summary(df, file)
