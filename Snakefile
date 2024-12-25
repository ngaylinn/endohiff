from itertools import product

from constants import NUM_TRIALS
from environments import ALL_ENVIRONMENT_NAMES

ALL_TRIALS = list(range(NUM_TRIALS))


rule all:
  input:
    # Experiment data
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/inner_log.parquet',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env.npz',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),

    # Single experiment visualizations
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env_map_fitness.png',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env_map_weights.png',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_map.gif',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_map.png',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/one_frac_map.gif',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/one_frac_map.png',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_dist.png',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),

    # Cross-experiment comparison charts
    'output/hiff_scores.parquet',
    expand('output/migration_{migration}_crossover_{crossover}/hiff_dist.png',
           migration=[True, False], crossover=[True, False]),
    expand('output/migration_{migration}_crossover_{crossover}/mannwhitneyu.txt',
           migration=[True, False], crossover=[True, False]),
    expand('output/hiff_dist_{env}.png', env=ALL_ENVIRONMENT_NAMES),
    expand('output/mannwhitneyu_{env}.txt', env=ALL_ENVIRONMENT_NAMES),

    # Supplemental experiment (baym variants)
    expand('output/baym_variants/{env}/env_map_fitness.png', env=['baym', 'no_ramp', 'stretched']),
    expand('output/baym_variants/{env}/env_map_weights.png', env=['baym', 'no_ramp', 'stretched']),
    expand('output/baym_variants/{env}/hiff_map.gif', env=['baym', 'no_ramp', 'stretched']),
    expand('output/baym_variants/{env}/hiff_map.png', env=['baym', 'no_ramp', 'stretched']),
    expand('output/baym_variants/{env}/one_frac_map.gif', env=['baym', 'no_ramp', 'stretched']),
    expand('output/baym_variants/{env}/one_frac_map.png', env=['baym', 'no_ramp', 'stretched']),
    'output/baym_variants/hiff_dist.png',
    'output/baym_variants/mannwhitneyu.txt',

    # Supplemental experiment (selection pressure)
    expand('output/selection_pressure/{env}/env_map_fitness.png', env=['baym', 'flat']),
    expand('output/selection_pressure/{env}/env_map_weights.png', env=['baym', 'flat']),
    expand('output/selection_pressure/{env}/hiff_map.gif', env=['baym', 'flat']),
    expand('output/selection_pressure/{env}/hiff_map.png', env=['baym', 'flat']),
    expand('output/selection_pressure/{env}/one_frac_map.gif', env=['baym', 'flat']),
    expand('output/selection_pressure/{env}/one_frac_map.png', env=['baym', 'flat']),
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
    expand('output/migration_{{migration}}_crossover_{{crossover}}/{{env}}/trial{t}/inner_log.parquet', t=ALL_TRIALS),
    expand('output/migration_{{migration}}_crossover_{{crossover}}/{{env}}/trial{t}/env.npz', t=ALL_TRIALS),
  resources: gpu=1 # This process expects to monopolize the GPU.
  params: '{env} {migration} {crossover} -v 0'
  shell: 'python3 run_experiment.py {params}'

rule visualize_one:
  input:
    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/inner_log.parquet',
    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env.npz',
  output:
    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env_map_fitness.png',
    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env_map_weights.png',
    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_map.gif',
    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_map.png',
    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/one_frac_map.gif',
    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/one_frac_map.png',
    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_dist.png',
  params: 'output/migration_{migration}_crossover_{crossover}/{env}/trial{t} -v 0'
  shell: 'python3 visualize_experiment.py {params}'

rule compare_experiments:
  input:
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/inner_log.parquet',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
  output:
    'output/hiff_scores.parquet',
    expand('output/migration_{migration}_crossover_{crossover}/hiff_dist.png',
           migration=[True, False], crossover=[True, False]),
    expand('output/migration_{migration}_crossover_{crossover}/mannwhitneyu.txt',
           migration=[True, False], crossover=[True, False]),
    expand('output/hiff_dist_{env}.png', env=ALL_ENVIRONMENT_NAMES),
    expand('output/mannwhitneyu_{env}.txt', env=ALL_ENVIRONMENT_NAMES),
  params: '-v 0'
  shell: 'python3 compare_experiments.py {params}'

rule run_supplemental_experiments:
  output:
    expand('output/baym_variants/{env}/env_map_fitness.png', env=['baym', 'no_ramp', 'stretched']),
    expand('output/baym_variants/{env}/env_map_weights.png', env=['baym', 'no_ramp', 'stretched']),
    expand('output/baym_variants/{env}/hiff_map.gif', env=['baym', 'no_ramp', 'stretched']),
    expand('output/baym_variants/{env}/hiff_map.png', env=['baym', 'no_ramp', 'stretched']),
    expand('output/baym_variants/{env}/one_frac_map.gif', env=['baym', 'no_ramp', 'stretched']),
    expand('output/baym_variants/{env}/one_frac_map.png', env=['baym', 'no_ramp', 'stretched']),
    'output/baym_variants/hiff_dist.png',
    'output/baym_variants/mannwhitneyu.txt',
    expand('output/selection_pressure/{env}/env_map_fitness.png', env=['baym', 'flat']),
    expand('output/selection_pressure/{env}/env_map_weights.png', env=['baym', 'flat']),
    expand('output/selection_pressure/{env}/hiff_map.gif', env=['baym', 'flat']),
    expand('output/selection_pressure/{env}/hiff_map.png', env=['baym', 'flat']),
    expand('output/selection_pressure/{env}/one_frac_map.gif', env=['baym', 'flat']),
    expand('output/selection_pressure/{env}/one_frac_map.png', env=['baym', 'flat']),
    'output/selection_pressure/hiff_dist.png',
    'output/selection_pressure/mannwhitneyu.txt',
  params: '-v 0'
  shell: 'python3 run_supplemental_experiments.py {params}'

rule generate_supplemental_figures:
  output:
    'output/hiff.png',
    'output/migration.png',
  shell: 'python3 generate_supplemental_figures.py'


# -----------------------------------------------------------------------------
# Rules to run all or part of the pipeline on just one environment.
# -----------------------------------------------------------------------------

for env in ALL_ENVIRONMENT_NAMES:
  rule:
    name: f'evolve:{env}'
    input:
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/inner_log.parquet',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/env.npz',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),

  # TODO: We want to visualize all trials for the evolved environments, but
  # there's not much benefit in doing that for the static environments. It'd be
  # nice to only visualize the best trials in that case, but not sure how to
  # tell Snakemake to do that.
  rule:
    name: f'visualize:{env}'
    input:
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/env_map_fitness.png',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/env_map_weights.png',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/hiff_map.gif',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/hiff_map.png',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/one_frac_map.gif',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/one_frac_map.png',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/hiff_dist.png',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),

  rule:
    name: f'{env}'
    input:
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/inner_log.parquet',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/env.npz',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/env_map_fitness.png',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/env_map_weights.png',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/hiff_map.gif',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/hiff_map.png',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/one_frac_map.gif',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/one_frac_map.png',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/hiff_dist.png',
             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),


# -----------------------------------------------------------------------------
# Rules to run just one phase of the pipeline for all environments
# -----------------------------------------------------------------------------

rule evolve:
  input:
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/inner_log.parquet',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env.npz',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),

rule visualize:
  input:
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env_map_fitness.png',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env_map_weights.png',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_map.gif',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_map.png',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/one_frac_map.gif',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/one_frac_map.png',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_dist.png',
           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
