from itertools import chain
from src.constants import ENV_NAMES, NUM_TRIALS
from src.extras.missing_steps import MISSING_STEP_ENVS
from src.sweep import SWEEP_KINDS, sweep_samples

TRIALS = list(range(NUM_TRIALS))
SWEEP_SAMPLES = sweep_samples()
ALL_SAMPLE_DIRS = list(chain(*(
  [f'{sweep_kind}_sweep/{sample_dir}' for sample_dir in sample_dirs]
  for sweep_kind, sample_dirs in SWEEP_SAMPLES.items()
)))

rule all:
  input:
    expand('output/{sweep_sample_dir}/{env_name}/trial_{trial}/env_map.png',
           sweep_sample_dir=ALL_SAMPLE_DIRS, env_name=ENV_NAMES, trial=TRIALS),
    expand('output/{sweep_sample_dir}/{env_name}/trial_{trial}/fitness_map.gif',
           sweep_sample_dir=ALL_SAMPLE_DIRS, env_name=ENV_NAMES, trial=TRIALS),
    expand('output/{sweep_sample_dir}/{env_name}/trial_{trial}/fitness_map.png',
           sweep_sample_dir=ALL_SAMPLE_DIRS, env_name=ENV_NAMES, trial=TRIALS),
    expand('output/{sweep_sample_dir}/{env_name}/best_trial.txt',
           sweep_sample_dir=ALL_SAMPLE_DIRS, env_name=ENV_NAMES),
    expand('output/{sweep_sample_dir}/{env_name}/bitstring_fitness.png',
           sweep_sample_dir=ALL_SAMPLE_DIRS, env_name=ENV_NAMES),
    expand('output/{sweep_sample_dir}/cppn/cppn_{trial}.png',
           sweep_sample_dir=ALL_SAMPLE_DIRS, trial=TRIALS),
    expand('output/{sweep_sample_dir}/cppn/environment_fitness.png',
           sweep_sample_dir=ALL_SAMPLE_DIRS),
    expand('output/{sweep_sample_dir}/fitness_comparison.png',
           sweep_sample_dir=ALL_SAMPLE_DIRS),
    expand('output/{sweep_kind}_sweep/{env_name}.png',
           sweep_kind=SWEEP_KINDS, env_name=ENV_NAMES),
    expand('output/{sweep_kind}_sweep/baym_vs_flat.png', sweep_kind=SWEEP_KINDS),
    expand('output/{sweep_kind}_sweep/cppn_vs_both.png', sweep_kind=SWEEP_KINDS),
    expand('output/missing_steps/{env_name}_env_map.png',
           env_name=MISSING_STEP_ENVS),
    expand('output/missing_steps/{env_name}_fitness_map.png',
           env_name=MISSING_STEP_ENVS),
    'output/hiff.png',
    'output/migration_0.2.png',
    'output/migration_0.5.png',
    'output/migration_1.0.png',
    'output/skew.png',



# ----------------------------------------------------------------------------
# Rules for rendering experiment figures
# ----------------------------------------------------------------------------

rule render_environment_map:
  input:
    '{path}/env.npy',
  output:
    '{path}/env_map.png',
  shell: 'python3 -m src.environments.visualize_one {input} -o {output}'


rule render_extra_cppn_map:
  input:
    '{path}/cppn_{trial,\d}.npy',
  output:
    '{path}/cppn_{trial,\d}.png',
  shell: 'python3 -m src.environments.visualize_one {input}'


rule chart_environment_fitness:
  input:
    '{path}/env_log.parquet',
  output:
    '{path}/environment_fitness.png',
  shell: 'python3 -m src.environments.visualize_fitness {input}'


rule render_bitstring_fitness_map:
  input:
    '{path}/bitstr_log.parquet',
  output:
    '{path}/fitness_map.png',
  shell: 'python3 -m src.bitstrings.visualize_population {input}'


rule render_bitstring_fitness_video:
  input:
    '{path}/bitstr_log.parquet',
  output:
    '{path}/fitness_map.gif',
  shell: 'python3 -m src.bitstrings.visualize_population {input} -f'


rule summarize_sample_dir:
  input:
    expand('{{path}}/{env_name}/trial_{trial}/bitstr_log.parquet',
           env_name=ENV_NAMES, trial=TRIALS),
  output:
    expand('{{path}}/{env_name}/best_trial.txt', env_name=ENV_NAMES),
    expand('{{path}}/{env_name}/bitstring_fitness.png', env_name=ENV_NAMES),
    '{path}/fitness_comparison.png',
  shell: 'python3 -m src.bitstrings.visualize_fitness {wildcards.path}'


rule render_sweep_summaries:
  input:
    expand('output/{{sweep_kind}}_sweep/{env_name}.parquet', env_name=ENV_NAMES),
  output:
    expand('output/{{sweep_kind}}_sweep/{env_name}.png', env_name=ENV_NAMES),
    expand('output/{{sweep_kind}}_sweep/baym_vs_flat.png'), 
    expand('output/{{sweep_kind}}_sweep/cppn_vs_both.png'), 
  shell: 'python3 -m src.sweep output/{wildcards.sweep_kind}_sweep {wildcards.sweep_kind}'



# ----------------------------------------------------------------------------
# Rules for evolving bitstrings and environments across hyperparameter values
# ----------------------------------------------------------------------------

rule selection_sweep_evolve_environments:
  output:
    'output/selection_sweep/cppn_envs.npy',
    expand('output/selection_sweep/{sample_dir}/cppn/cppn_{trial}.npy',
           sample_dir=SWEEP_SAMPLES['selection'], trial=TRIALS),
    expand('output/selection_sweep/{sample_dir}/cppn/env_log.parquet',
           sample_dir=SWEEP_SAMPLES['selection'], trial=TRIALS),
  shell: 'python3 -m src.environments.sweep output/selection_sweep selection'


rule ratchet_sweep_evolve_environments:
  output:
    'output/ratchet_sweep/cppn_envs.npy',
    expand('output/ratchet_sweep/{sample_dir}/cppn/cppn_{trial}.npy',
           sample_dir=SWEEP_SAMPLES['ratchet'], trial=TRIALS),
    expand('output/ratchet_sweep/{sample_dir}/cppn/env_log.parquet',
           sample_dir=SWEEP_SAMPLES['ratchet'], trial=TRIALS),
  shell: 'python3 -m src.environments.sweep output/ratchet_sweep ratchet'


rule selection_sweep_evolve_bitstrings:
  output:
    'output/selection_sweep/{env_name}.parquet',
    expand('output/selection_sweep/{sample_dir}/{{env_name}}/trial_{trial}/env.npy',
           sample_dir=SWEEP_SAMPLES['selection'], trial=TRIALS),
    expand('output/selection_sweep/{sample_dir}/{{env_name}}/trial_{trial}/bitstr_log.parquet',
           sample_dir=SWEEP_SAMPLES['selection'], trial=TRIALS),
  shell: 'python3 -m src.bitstrings.sweep output/selection_sweep selection {wildcards.env_name}'


rule ratchet_sweep_evolve_bitstrings:
  output:
    'output/ratchet_sweep/{env_name}.parquet',
    expand('output/ratchet_sweep/{sample_dir}/{{env_name}}/trial_{trial}/env.npy',
           sample_dir=SWEEP_SAMPLES['ratchet'], trial=TRIALS),
    expand('output/ratchet_sweep/{sample_dir}/{{env_name}}/trial_{trial}/bitstr_log.parquet',
           sample_dir=SWEEP_SAMPLES['ratchet'], trial=TRIALS),
  shell: 'python3 -m src.bitstrings.sweep output/ratchet_sweep ratchet {wildcards.env_name}'



# ----------------------------------------------------------------------------
# Rules for generating additional supplementary figures
# ----------------------------------------------------------------------------

rule analyze_skew:
  output:
    'output/skew.png',
  shell: 'python3 -m src.extras.analyze_skew {output}'


rule missing_steps:
  output:
    expand('output/missing_steps/{env_name}_env_map.png',
           env_name=MISSING_STEP_ENVS),
    expand('output/missing_steps/{env_name}_fitness_map.png',
           env_name=MISSING_STEP_ENVS),
  shell: 'python3 -m src.extras.missing_steps output/missing_steps'


rule visualize_hiff:
  output:
    'output/hiff.png',
  shell: 'python3 -m src.extras.visualize_hiff {output}'


rule visualize_migration:
  output:
    'output/migration_0.2.png',
    'output/migration_0.5.png',
    'output/migration_1.0.png',
  shell: 'python3 -m src.extras.visualize_hiff output'
