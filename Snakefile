from environment import ENVIRONMENTS
from itertools import product

rule all:
  input:
    # Experiment data
    expand('output/migration_{migration}_crossover_{crossover}/{env}/best_trial.parquet',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/hiff_scores.parquet',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/env.npz',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),

    # Single experiment visualizations
    expand('output/migration_{migration}_crossover_{crossover}/{env}/env_map_fitness.png',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/env_map_weights.png',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/hiff_map.gif',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/hiff_map.png',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/one_frac_map.gif',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/one_frac_map.png',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),

    # Single experiment charts
    expand('output/migration_{migration}_crossover_{crossover}/{env}/hiff_dist.png',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),

    # Cross-experiment comparison charts
    expand('output/migration_{migration}_crossover_{crossover}/hiff_dist.png',
           migration=[True, False], crossover=[True, False]),
    expand('output/migration_{migration}_crossover_{crossover}/mannwhitneyu.txt',
           migration=[True, False], crossover=[True, False]),
    expand('output/hiff_dist_{env}.png', env=ENVIRONMENTS),
    expand('output/mannwhitneyu_{env}.txt', env=ENVIRONMENTS),

    # Supplemental experiment (baym variants)
    expand('output/baym_variants/{env}/env_map_fitness.png',
           env=['baym', 'no_ramp', 'stretched']),
    expand('output/baym_variants/{env}/env_map_weights.png',
           env=['baym', 'no_ramp', 'stretched']),
    expand('output/baym_variants/{env}/hiff_dist.png',
           env=['baym', 'stretched']),
    expand('output/baym_variants/{env}/hiff_map.png',
           env=['baym', 'no_ramp', 'stretched']),
    'output/baym_variants/hiff_dist.png',
    'output/baym_variants/mannwhitneyu.txt',

    # Supplemental experiment (selection pressure)
    expand('output/selection_pressure/{env}/hiff_dist.png',
           env=['baym', 'flat']),
    expand('output/selection_pressure/{env}/hiff_map.png',
           env=['baym', 'flat']),
    'output/selection_pressure/hiff_dist.png',
    'output/selection_pressure/mannwhitneyu.txt',

    # Supplemental figures
    'output/hiff.png',
    'output/migration.png',


# -----------------------------------------------------------------------------
# Rules to run just one step of the pipeline in isolation.
# -----------------------------------------------------------------------------

rule evolve_one:
  output:
    'output/migration_{migration}_crossover_{crossover}/{env}/best_trial.parquet',
    'output/migration_{migration}_crossover_{crossover}/{env}/hiff_scores.parquet',
    'output/migration_{migration}_crossover_{crossover}/{env}/env.npz',
  resources: gpu=1 # This process expects to monopolize the GPU.
  params: '{env} {migration} {crossover} -v 0'
  shell: 'python3 run_experiment.py {params}'

rule render_one:
  input:
    'output/migration_{migration}_crossover_{crossover}/{env}/best_trial.parquet',
    'output/migration_{migration}_crossover_{crossover}/{env}/env.npz',
  output:
      'output/migration_{migration}_crossover_{crossover}/{env}/env_map_fitness.png',
      'output/migration_{migration}_crossover_{crossover}/{env}/env_map_weights.png',
      'output/migration_{migration}_crossover_{crossover}/{env}/hiff_map.gif',
      'output/migration_{migration}_crossover_{crossover}/{env}/hiff_map.png',
      'output/migration_{migration}_crossover_{crossover}/{env}/one_frac_map.gif',
      'output/migration_{migration}_crossover_{crossover}/{env}/one_frac_map.png'
  params: 'output/migration_{migration}_crossover_{crossover}/{env}/ -v 0'
  shell: 'python3 render.py {params}'

rule chart_one:
  input:
    'output/migration_{migration}_crossover_{crossover}/{env}/best_trial.parquet',
  output:
    'output/migration_{migration}_crossover_{crossover}/{env}/hiff_dist.png',
  params: 'output/migration_{migration}_crossover_{crossover}/{env}/ -v 0'
  shell: 'python3 chart.py {params}'

rule chart_across_experiments:
  input:
    expand('output/migration_{migration}_crossover_{crossover}/{env}/hiff_scores.parquet',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
  output:
    expand('output/migration_{migration}_crossover_{crossover}/hiff_dist.png',
           migration=[True, False], crossover=[True, False]),
    expand('output/migration_{migration}_crossover_{crossover}/mannwhitneyu.txt',
           migration=[True, False], crossover=[True, False]),
    expand('output/hiff_dist_{env}.png', env=ENVIRONMENTS),
    expand('output/mannwhitneyu_{env}.txt', env=ENVIRONMENTS),
  params: '-v 0'
  shell: 'python3 chart_across_experiments.py {params}'

rule baym_variants:
  output:
    expand('output/baym_variants/{env}/env_map_fitness.png',
           env=['baym', 'no_ramp', 'stretched']),
    expand('output/baym_variants/{env}/env_map_weights.png',
           env=['baym', 'no_ramp', 'stretched']),
    expand('output/baym_variants/{env}/hiff_dist.png',
           env=['baym', 'stretched']),
    expand('output/baym_variants/{env}/hiff_map.png',
           env=['baym', 'no_ramp', 'stretched']),
    'output/baym_variants/hiff_dist.png',
    'output/baym_variants/mannwhitneyu.txt',
  shell: 'python3 baym_variants.py'

rule selection_pressure:
  output:
    expand('output/selection_pressure/{env}/hiff_dist.png',
           env=['baym', 'flat']),
    expand('output/selection_pressure/{env}/hiff_map.png',
           env=['baym', 'flat']),
    'output/selection_pressure/hiff_dist.png',
    'output/selection_pressure/mannwhitneyu.txt',
  shell: 'python3 selection_pressure.py'

rule supplemental_figures:
  output:
    'output/hiff.png',
    'output/migration.png',
  shell: 'python3 supplemental_figures.py'


# -----------------------------------------------------------------------------
# Rules to run all or part of the pipeline on just one environment.
# -----------------------------------------------------------------------------

for env in ENVIRONMENTS.keys():
  rule:
    name: f'evolve:{env}'
    input:
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/best_trial.parquet',
             migration=[True, False], crossover=[True, False]),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/hiff_scores.parquet',
             migration=[True, False], crossover=[True, False]),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/env.npz',
             migration=[True, False], crossover=[True, False]),

  rule:
    name: f'chart:{env}'
    input:
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/hiff_dist.png',
             migration=[True, False], crossover=[True, False]),

  rule:
    name: f'render:{env}'
    input:
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/env_map_fitness.png',
             migration=[True, False], crossover=[True, False]),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/env_map_weights.png',
             migration=[True, False], crossover=[True, False]),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/hiff_map.gif',
             migration=[True, False], crossover=[True, False]),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/hiff_map.png',
             migration=[True, False], crossover=[True, False]),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/one_frac_map.gif',
             migration=[True, False], crossover=[True, False]),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/one_frac_map.png',
             migration=[True, False], crossover=[True, False]),

  rule:
    name: f'{env}'
    input:
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/best_trial.parquet',
             migration=[True, False], crossover=[True, False]),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/hiff_scores.parquet',
             migration=[True, False], crossover=[True, False]),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/env.npz',
             migration=[True, False], crossover=[True, False]),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/hiff_dist.png',
             migration=[True, False], crossover=[True, False]),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/env_map_fitness.png',
             migration=[True, False], crossover=[True, False]),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/env_map_weights.png',
             migration=[True, False], crossover=[True, False]),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/hiff_map.gif',
             migration=[True, False], crossover=[True, False]),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/hiff_map.png',
             migration=[True, False], crossover=[True, False]),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/one_frac_map.gif',
             migration=[True, False], crossover=[True, False]),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/one_frac_map.png',
             migration=[True, False], crossover=[True, False]),


# -----------------------------------------------------------------------------
# Rules to run just one phase of the pipeline for all experiments
# -----------------------------------------------------------------------------

rule evolve:
  input:
    expand('output/migration_{migration}_crossover_{crossover}/{env}/best_trial.parquet',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/hiff_scores.parquet',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/env.npz',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),

rule chart:
  input:
    expand('output/migration_{migration}_crossover_{crossover}/{env}/hiff_dist.png',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
    expand('output/migration_{migration}_crossover_{crossover}/hiff_dist.png',
           migration=[True, False], crossover=[True, False]),
    expand('output/migration_{migration}_crossover_{crossover}/mannwhitneyu.txt',
           migration=[True, False], crossover=[True, False]),
    expand('output/hiff_dist_{env}.png', env=ENVIRONMENTS),
    expand('output/mannwhitneyu_{env}.txt', env=ENVIRONMENTS),

rule render:
  input:
    expand('output/migration_{migration}_crossover_{crossover}/{env}/env_map_fitness.png',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/env_map_weights.png',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/hiff_map.gif',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/hiff_map.png',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/one_frac_map.gif',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/one_frac_map.png',
           migration=[True, False], crossover=[True, False], env=ENVIRONMENTS),
