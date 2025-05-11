# endohiff
Endosymbiotic HIFF

A research project inspired by endosymbiosis. A "host" will provide an
evolved environment for a population of "symbionts." By shaping the fitness
landscape, we hypothesize the host can influence the evolutionary trajectory of
its symbionts and lead them to achieve higher fitness scores than they would in
a non-agential environment.

***This is a work in progress! Our code is currently lagging behind what was used to generate the results in our paper, and is in need of significant refactoring. Many improvements coming soon.***

## Usage

To run the code in this project, you must install the dependencies listed in
environment.yml, using a tool such as Conda, or manually.

This project is optimized to run on a computer with a single CUDA-enabled GPU.
It should be easily modifiable to run on other system configurations, but this
may require small changes to calls to `ti.init()` and the `Snakefile`.

This project runs several evolutionary experiments, using separate scripts to
generate results, visualize runs, and chart performance within and across
experiments. There are also a few supplemental scripts to run secondary
experiments or generate figures for the write-up. You can run these scripts
directly, or use snakemake to regenerate output files more efficiently in
batches:

* `snakemake all --resources gpu=1`: Ensure all experiments have been run and
  all visualizations, charts, and figures have been generated.
* `snakemake evolve --resources gpu=1`: Ensure all experiments have been run.
* `snakemake render`: Ensure visualizations of all the best experiment runs
  have been rendered.
* `snakemake chart`: Ensure results from all experiments have been charted.
* `snakemake evolve:<env> --resources gpu=1`: Ensure all experiments for the
  given environment have been run.
* `snakemake render:<env>`: Ensure visualizations of all the best experiment
  runs for the given environment have been rendered.
* `snakemake chart:<env>`: Ensure results from all experiments for the given
  environment have been charted.
* `snakemake <env> --resources gpu=1`: Ensure all experiments for the given
  environment have been run and all associated visualizations, charts, and
  figures have  been generated.
