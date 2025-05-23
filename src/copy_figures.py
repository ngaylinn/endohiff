from itertools import count
from pathlib import Path
from shutil import copyfile


OUTPUT_PATH = Path('output')
SELECTION_SWEEP_PATH = OUTPUT_PATH / 'selection_sweep'
FIGURES_PATH = OUTPUT_PATH / 'figures'
FIGURES_PATH.mkdir(exist_ok=True, parents=True)


def copy_bitstring_evolution_visualizations():
    deep_dive_condition = 'mortality_rate_0.125_tournament_size_6'
    deep_dive_path = SELECTION_SWEEP_PATH / deep_dive_condition
    for env_name in ['flat', 'baym', 'cppn']:
        copyfile(deep_dive_path / env_name / 'env_fitness.png',
                 FIGURES_PATH / f'{env_name}_fitness.png')
        with open(deep_dive_path / env_name / 'best_trial.txt', 'r') as file:
            best_trial = file.readline().strip()
        trial_path = deep_dive_path / env_name / best_trial
        copyfile(trial_path / 'env_map.png',
                 FIGURES_PATH / f'{env_name}_map.png')
        copyfile(trial_path / 'fitness_map.png',
                 FIGURES_PATH / f'{env_name}_fitness_map.png')


def copy_evolved_environment_samples():
    cppns_path = FIGURES_PATH / 'cppns'
    cppns_path.mkdir(exist_ok=True, parents=True)
    counter = count()
    for sample_dir in SELECTION_SWEEP_PATH.glob('*/'):
        for sample in (sample_dir / 'cppn').glob('cppn*png'):
            copyfile(sample, cppns_path / f'cppn_{next(counter)}.png')


def copy_sweep_visualizations():
    copyfile(SELECTION_SWEEP_PATH / 'flat.png',
             FIGURES_PATH / 'flat.png')
    copyfile(SELECTION_SWEEP_PATH / 'baym.png',
             FIGURES_PATH / 'baym.png')
    copyfile(SELECTION_SWEEP_PATH / 'cppn.png',
             FIGURES_PATH / 'cppn.png')
    copyfile(SELECTION_SWEEP_PATH / 'baym_vs_flat.png',
             FIGURES_PATH / 'baym_vs_flat.png')
    copyfile(SELECTION_SWEEP_PATH / 'cppn_vs_both.png',
             FIGURES_PATH / 'cppn_vs_both.png')


def copy_missing_step_visualizations():
    variants_path = OUTPUT_PATH / 'missing_steps'
    for variant in ['gap', 'high']:
        copyfile(variants_path / variant / 'env_map.png',
                 FIGURES_PATH / f'{variant}_map.png')
        copyfile(variants_path / variant / 'fitness_map.png',
                 FIGURES_PATH / f'{variant}_fitness_map.png')


def copy_broken_ramp_visualizations():
    ratchet_sweep_path = OUTPUT_PATH / 'ratchet_sweep'
    low_migration_condition = 'migration_rate_0.125_fertility_rate_7'
    low_migration_path = ratchet_sweep_path / low_migration_condition
    copyfile(low_migration_path / 'baym' / 'trial_0' / 'fitness_map.png',
             FIGURES_PATH / 'low_migration.png')
    low_fertility_condition = 'migration_rate_1.500_fertility_rate_3'
    low_fertility_path = ratchet_sweep_path / low_fertility_condition
    copyfile(low_fertility_path / 'baym' / 'trial_0' / 'fitness_map.png',
             FIGURES_PATH / 'low_fertility.png')


if __name__ == '__main__':
    copy_bitstring_evolution_visualizations()
    copy_evolved_environment_samples()
    copy_sweep_visualizations()
    copy_missing_step_visualizations()
    copy_broken_ramp_visualizations()
