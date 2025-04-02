from constants import NUM_TRIALS
from sweep import SWEEP_KINDS, all_sweep_sample_dirs

ENV_NAMES = ['baym', 'cppn', 'flat']
TRIALS = list(range(NUM_TRIALS))
SAMPLE_DIRS = all_sweep_sample_dirs()

rule all:
  input:
    expand('output/{sweep_sample_dir}/{env_name}/trial_{trial}/env_map.png',
           sweep_sample_dir=SAMPLE_DIRS, env_name=ENV_NAMES, trial=TRIALS),
    expand('output/{sweep_sample_dir}/{env_name}/trial_{trial}/inner_fitness.png',
           sweep_sample_dir=SAMPLE_DIRS, env_name=ENV_NAMES, trial=TRIALS),
    expand('output/{sweep_sample_dir}/{env_name}/trial_{trial}/fitness_map.gif',
           sweep_sample_dir=SAMPLE_DIRS, env_name=ENV_NAMES, trial=TRIALS),
    expand('output/{sweep_sample_dir}/{env_name}/trial_{trial}/fitness_map.png',
           sweep_sample_dir=SAMPLE_DIRS, env_name=ENV_NAMES, trial=TRIALS),
    expand('output/{sweep_sample_dir}/{env_name}/trial_{trial}/one_frac_map.gif',
           sweep_sample_dir=SAMPLE_DIRS, env_name=ENV_NAMES, trial=TRIALS),
    expand('output/{sweep_sample_dir}/{env_name}/trial_{trial}/one_frac_map.png',
           sweep_sample_dir=SAMPLE_DIRS, env_name=ENV_NAMES, trial=TRIALS),
    expand('output/{sweep_sample_dir}/cppn/cppn_{trial}.png',
           sweep_sample_dir=SAMPLE_DIRS, trial=TRIALS),
    expand('output/{sweep_sample_dir}/cppn/outer_fitness.png',
           sweep_sample_dir=SAMPLE_DIRS, trial=TRIALS),
    expand('output/{sweep_sample_dir}/fitness_dist.png',
           sweep_sample_dir=SAMPLE_DIRS, trial=TRIALS),
    expand('output/{sweep_sample_dir}/mannwhitneyu.txt',
           sweep_sample_dir=SAMPLE_DIRS, trial=TRIALS),
    expand('output/{sweep_kind}_sweep/{env_name}.png',
           sweep_kind=SWEEP_KINDS, env_name=ENV_NAMES),
    expand('output/{sweep_kind}_sweep/baym_vs_flat.png', sweep_kind=SWEEP_KINDS),
    expand('output/{sweep_kind}_sweep/cppn_vs_baym.png', sweep_kind=SWEEP_KINDS),
    expand('output/{sweep_kind}_sweep/cppn_vs_flat.png', sweep_kind=SWEEP_KINDS),

    # TODO: Restore secondary figures!


rule visualize_bitstr_evolution:
  input:
    '{path}/env.npy',
    '{path}/inner_log.parquet',
  output:
    '{path}/env_map.png',
    '{path}/inner_fitness.png',
    '{path}/fitness_map.gif',
    '{path}/fitness_map.png',
    '{path}/one_frac_map.gif',
    '{path}/one_frac_map.png',
  params: '{path} -v 0'
  shell: 'python3 ./visualize_inner_population.py {params}'


rule visualize_environment_evolution:
  input:
    expand('{{path}}/cppn_{trial}.npy', trial=TRIALS),
    '{path}/outer_log.parquet',
  output:
    expand('{{path}}/cppn_{trial}.png', trial=TRIALS),
    '{path}/outer_fitness.png',
  params: '{path} -v 0'
  shell: 'python3 ./visualize_outer_population.py {params}'


rule compare_environments:
  input:
    expand('{{path}}/{env_name}/trial_{trial}/inner_log.parquet',
           env_name=ENV_NAMES, trial=TRIALS),
  output:
    '{path}/fitness_dist.png',
    '{path}/mannwhitneyu.txt',
  params: '{path}'
  shell: 'python3 ./compare_experiments.py {params}'


# TODO: Rules to run the sweep.


#from itertools import product
#
#from constants import NUM_TRIALS, OUTER_POPULATION_SIZE
#from environments import ALL_ENVIRONMENT_NAMES
#
#SAMPLE_ENVIRONMENTS = list(range(OUTER_POPULATION_SIZE))
#ALL_TRIALS = list(range(NUM_TRIALS))
#
#
#rule all:
#  input:
#    # Experiment data
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/inner_log.parquet',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env.npy',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#
#    # Single experiment visualizations
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env_map_fitness.png',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env_map_weights.png',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_map.gif',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_map.png',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/one_frac_map.gif',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/one_frac_map.png',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_dist.png',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#
#    # Extra visualizations for experiments with evolved environmnets
#    expand('output/migration_{migration}_crossover_{crossover}/cppn/trial{t}/outer_fitness.png',
#           migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/cppn/outer_fitness.png',
#           migration=[True, False], crossover=[True, False]),
#    expand('output/migration_{migration}_crossover_{crossover}/cppn/outer_fitness_by_trial.png',
#           migration=[True, False], crossover=[True, False]),
#
#    # Cross-experiment comparison charts
#    'output/hiff_scores.parquet',
#    expand('output/migration_{migration}_crossover_{crossover}/hiff_dist.png',
#           migration=[True, False], crossover=[True, False]),
#    expand('output/migration_{migration}_crossover_{crossover}/mannwhitneyu.txt',
#           migration=[True, False], crossover=[True, False]),
#    expand('output/hiff_dist_{env}.png', env=ALL_ENVIRONMENT_NAMES),
#    expand('output/mannwhitneyu_{env}.txt', env=ALL_ENVIRONMENT_NAMES),
#
#    # Supplemental experiment (baym variants)
#    expand('output/baym_variants/{env}/env_map_fitness.png', env=['baym', 'no_ramp', 'stretched']),
#    expand('output/baym_variants/{env}/env_map_weights.png', env=['baym', 'no_ramp', 'stretched']),
#    expand('output/baym_variants/{env}/hiff_map.gif', env=['baym', 'no_ramp', 'stretched']),
#    expand('output/baym_variants/{env}/hiff_map.png', env=['baym', 'no_ramp', 'stretched']),
#    expand('output/baym_variants/{env}/one_frac_map.gif', env=['baym', 'no_ramp', 'stretched']),
#    expand('output/baym_variants/{env}/one_frac_map.png', env=['baym', 'no_ramp', 'stretched']),
#    'output/baym_variants/hiff_dist.png',
#    'output/baym_variants/mannwhitneyu.txt',
#
#    # Supplemental experiment (selection pressure)
#    expand('output/selection_pressure/{env}/env_map_fitness.png', env=['baym', 'flat']),
#    expand('output/selection_pressure/{env}/env_map_weights.png', env=['baym', 'flat']),
#    expand('output/selection_pressure/{env}/hiff_map.gif', env=['baym', 'flat']),
#    expand('output/selection_pressure/{env}/hiff_map.png', env=['baym', 'flat']),
#    expand('output/selection_pressure/{env}/one_frac_map.gif', env=['baym', 'flat']),
#    expand('output/selection_pressure/{env}/one_frac_map.png', env=['baym', 'flat']),
#    'output/selection_pressure/hiff_dist.png',
#    'output/selection_pressure/mannwhitneyu.txt',
#
#    # Supplemental figures
#    'output/hiff.png',
#    'output/migration.png',
#    expand('output/random_cppn_environments_without_weights/cppn{e}_weights.png', e=SAMPLE_ENVIRONMENTS),
#    expand('output/random_cppn_environments_without_weights/cppn{e}_fitness.png', e=SAMPLE_ENVIRONMENTS),
#    expand('output/random_cppn_environments_with_weights/cppn{e}_weights.png', e=SAMPLE_ENVIRONMENTS),
#    expand('output/random_cppn_environments_with_weights/cppn{e}_fitness.png', e=SAMPLE_ENVIRONMENTS),
#
#
## -----------------------------------------------------------------------------
## Rules to run just one step of the pipeline in isolation.
## -----------------------------------------------------------------------------
#
#rule evolve_one:
#  output:
#    expand('output/migration_{{migration}}_crossover_{{crossover}}/{{env}}/trial{t}/inner_log.parquet', t=ALL_TRIALS),
#    expand('output/migration_{{migration}}_crossover_{{crossover}}/{{env}}/trial{t}/env.npy', t=ALL_TRIALS),
#    expand('output/migration_{{migration}}_crossover_{{crossover}}/{{env}}/outer_log.parquet'),
#  resources: gpu=1 # This process expects to monopolize the GPU.
#  params: '{env} {migration} {crossover} -v 0'
#  shell: 'python3 run_experiment.py {params}'
#
#rule visualize_inner:
#  input:
#    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/inner_log.parquet',
#    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env.npy',
#  output:
#    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env_map_fitness.png',
#    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env_map_weights.png',
#    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_map.gif',
#    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_map.png',
#    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/one_frac_map.gif',
#    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/one_frac_map.png',
#    'output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_dist.png',
#  params: 'output/migration_{migration}_crossover_{crossover}/{env}/trial{t} -v 0'
#  shell: 'python3 visualize_inner_population.py {params}'
#
#rule visualize_outer:
#  input:
#    expand('output/migration_{{migration}}_crossover_{{crossover}}/cppn/outer_log.parquet'),
#  output:
#    expand('output/migration_{{migration}}_crossover_{{crossover}}/cppn/trial{t}/outer_fitness.png', t=ALL_TRIALS),
#    'output/migration_{migration}_crossover_{crossover}/cppn/outer_fitness.png',
#    'output/migration_{migration}_crossover_{crossover}/cppn/outer_fitness_by_trial.png',
#  params: 'output/migration_{migration}_crossover_{crossover}/cppn -v 0'
#  shell: 'python3 visualize_outer_population.py {params}'
#
#rule compare_experiments:
#  input:
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/inner_log.parquet',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#  output:
#    'output/hiff_scores.parquet',
#    expand('output/migration_{migration}_crossover_{crossover}/hiff_dist.png',
#           migration=[True, False], crossover=[True, False]),
#    expand('output/migration_{migration}_crossover_{crossover}/mannwhitneyu.txt',
#           migration=[True, False], crossover=[True, False]),
#    expand('output/hiff_dist_{env}.png', env=ALL_ENVIRONMENT_NAMES),
#    expand('output/mannwhitneyu_{env}.txt', env=ALL_ENVIRONMENT_NAMES),
#  params: '-v 0'
#  shell: 'python3 compare_experiments.py {params}'
#
#rule run_supplemental_experiments:
#  output:
#    expand('output/baym_variants/{env}/env_map_fitness.png', env=['baym', 'no_ramp', 'stretched']),
#    expand('output/baym_variants/{env}/env_map_weights.png', env=['baym', 'no_ramp', 'stretched']),
#    expand('output/baym_variants/{env}/hiff_map.gif', env=['baym', 'no_ramp', 'stretched']),
#    expand('output/baym_variants/{env}/hiff_map.png', env=['baym', 'no_ramp', 'stretched']),
#    expand('output/baym_variants/{env}/one_frac_map.gif', env=['baym', 'no_ramp', 'stretched']),
#    expand('output/baym_variants/{env}/one_frac_map.png', env=['baym', 'no_ramp', 'stretched']),
#    'output/baym_variants/hiff_dist.png',
#    'output/baym_variants/mannwhitneyu.txt',
#    expand('output/selection_pressure/{env}/env_map_fitness.png', env=['baym', 'flat']),
#    expand('output/selection_pressure/{env}/env_map_weights.png', env=['baym', 'flat']),
#    expand('output/selection_pressure/{env}/hiff_map.gif', env=['baym', 'flat']),
#    expand('output/selection_pressure/{env}/hiff_map.png', env=['baym', 'flat']),
#    expand('output/selection_pressure/{env}/one_frac_map.gif', env=['baym', 'flat']),
#    expand('output/selection_pressure/{env}/one_frac_map.png', env=['baym', 'flat']),
#    'output/selection_pressure/hiff_dist.png',
#    'output/selection_pressure/mannwhitneyu.txt',
#  params: '-v 0'
#  shell: 'python3 run_supplemental_experiments.py {params}'
#
#rule generate_supplemental_figures:
#  output:
#    'output/hiff.png',
#    'output/migration.png',
#  shell: 'python3 generate_supplemental_figures.py'
#
#rule visualize_random_cppn_environments:
#  output:
#    expand('output/random_cppn_environments_without_weights/cppn{e}_weights.png', e=SAMPLE_ENVIRONMENTS),
#    expand('output/random_cppn_environments_without_weights/cppn{e}_fitness.png', e=SAMPLE_ENVIRONMENTS),
#    expand('output/random_cppn_environments_with_weights/cppn{e}_weights.png', e=SAMPLE_ENVIRONMENTS),
#    expand('output/random_cppn_environments_with_weights/cppn{e}_fitness.png', e=SAMPLE_ENVIRONMENTS),
#  shell: 'python3 outer_population.py'
#
## -----------------------------------------------------------------------------
## Rules to run all or part of the pipeline on just one environment.
## -----------------------------------------------------------------------------
#
#for env in ALL_ENVIRONMENT_NAMES:
#  # Some outputs are only generated for the cppn environment. We disable those
#  # outputs in other circumstances by calling the expand() macro with an empty
#  # list as one of its arguments.
#  is_cppn = ([True] if env == 'cppn' else [])
#
#  rule:
#    name: f'evolve:{env}'
#    input:
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/inner_log.parquet',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/env.npy',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#
#  # TODO: We want to visualize all trials for the evolved environments, but
#  # there's not much benefit in doing that for the static environments. It'd be
#  # nice to only visualize the best trials in that case, but not sure how to
#  # tell Snakemake to do that.
#  rule:
#    name: f'visualize:{env}'
#    input:
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/env_map_fitness.png',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/env_map_weights.png',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/hiff_map.gif',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/hiff_map.png',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/one_frac_map.gif',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/one_frac_map.png',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/hiff_dist.png',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/cppn/trial{{t}}/outer_fitness.png',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS, use_output=is_cppn),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/cppn/outer_fitness.png',
#             migration=[True, False], crossover=[True, False], use_output=is_cppn),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/cppn/outer_fitness_by_trial.png',
#             migration=[True, False], crossover=[True, False], use_output=is_cppn),
#
#  rule:
#    name: f'{env}'
#    input:
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/inner_log.parquet',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/env.npy',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/env_map_fitness.png',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/env_map_weights.png',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/hiff_map.gif',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/hiff_map.png',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/one_frac_map.gif',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/one_frac_map.png',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#      expand(f'output/migration_{{migration}}_crossover_{{crossover}}/{env}/trial{{t}}/hiff_dist.png',
#             migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#
#
## -----------------------------------------------------------------------------
## Rules to run just one phase of the pipeline for all environments
## -----------------------------------------------------------------------------
#
#rule evolve:
#  input:
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/inner_log.parquet',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env.npy',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#
#rule visualize:
#  input:
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env_map_fitness.png',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/env_map_weights.png',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_map.gif',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_map.png',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/one_frac_map.gif',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/one_frac_map.png',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/{env}/trial{t}/hiff_dist.png',
#           migration=[True, False], crossover=[True, False], env=ALL_ENVIRONMENT_NAMES, t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/cppn/trial{t}/outer_fitness.png',
#           migration=[True, False], crossover=[True, False], t=ALL_TRIALS),
#    expand('output/migration_{migration}_crossover_{crossover}/cppn/outer_fitness.png',
#           migration=[True, False], crossover=[True, False]),
#    expand('output/migration_{migration}_crossover_{crossover}/cppn/outer_fitness_by_trial.png',
#           migration=[True, False], crossover=[True, False]),
