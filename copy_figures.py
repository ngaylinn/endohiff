from itertools import count
from shutil import copyfile

from constants import OUTPUT_PATH


selection_sweep_path = OUTPUT_PATH / 'selection_sweep'
figures_path = OUTPUT_PATH / 'figures'
figures_path.mkdir(exist_ok=True, parents=True)


def copy_bitstring_evolution_visualizations():
    deep_dive_condition = 'mortality_rate_0.125_tournament_size_6'
    deep_dive_path = selection_sweep_path / deep_dive_condition
    for env_name in ['flat', 'baym', 'cppn']:
        copyfile(deep_dive_path / env_name / 'env_fitness.png',
                 figures_path / f'{env_name}_fitness.png')
        with open(deep_dive_path / env_name / 'best_trial.txt', 'r') as file:
            best_trial = file.readline().strip()
        trial_path = deep_dive_path / env_name / best_trial
        copyfile(trial_path / 'env_map.png',
                 figures_path / f'{env_name}_map.png')
        copyfile(trial_path / 'fitness_map.png',
                 figures_path / f'{env_name}_fitness_map.png')


def copy_evolved_environment_samples():
    cppns_path = figures_path / 'cppns'
    cppns_path.mkdir(exist_ok=True, parents=True)
    counter = count()
    for sample_dir in selection_sweep_path.glob('*/'):
        for sample in (sample_dir / 'cppn').glob('cppn*png'):
            copyfile(sample, cppns_path / f'cppn_{next(counter)}.png')


def copy_sweep_visualizations():
    copyfile(selection_sweep_path / 'baym_vs_flat.png',
             figures_path / 'baym_vs_flat.png')
    copyfile(selection_sweep_path / 'cppn_vs_both.png',
             figures_path / 'cppn_vs_both.png')


def copy_baym_variant_visualizations():
    variants_path = OUTPUT_PATH / 'baym_variants'
    copyfile(variants_path / 'final_fitness.png',
             figures_path / 'baym_variants_comparison.png')
    for variant in ['gap', 'high']:
        copyfile(variants_path / variant / 'env_map.png',
                 figures_path / f'{variant}_map.png')
        copyfile(variants_path / variant / 'fitness_map.png',
                 figures_path / f'{variant}_fitness_map.png')


def copy_broken_ramp_visualizations():
    ratchet_sweep_path = OUTPUT_PATH / 'ratchet_sweep'
    low_migration_condition = 'migration_rate_0.125_fertility_rate_7'
    low_migration_path = ratchet_sweep_path / low_migration_condition
    copyfile(low_migration_path / 'baym' / 'trial_0' / 'fitness_map.png',
             figures_path / 'low_migration.png')
    low_fertility_condition = 'migration_rate_1.500_fertility_rate_3'
    low_fertility_path = ratchet_sweep_path / low_fertility_condition
    copyfile(low_fertility_path / 'baym' / 'trial_0' / 'fitness_map.png',
             figures_path / 'low_fertility.png')


if __name__ == '__main__':
    copy_bitstring_evolution_visualizations()
    copy_evolved_environment_samples()
    copy_sweep_visualizations()
    copy_baym_variant_visualizations()
    copy_broken_ramp_visualizations()
