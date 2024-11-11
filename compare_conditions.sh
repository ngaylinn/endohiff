#!/bin/bash

# If any of the below commands fail, exit immediately.
set -e

# Delete old outputs if they exist.
rm -rf output

echo "Simulating evolution (crossover=False, migration=False)"
python3 ./run_experiments.py -c 0 -m 0
for log in output/*/inner_log.parquet
do
    mv "$log" "$(dirname $log)/log_c0_m0.parquet"
done

echo "Simulating evolution (crossover=True, migration=False)"
python3 ./run_experiments.py -c 1 -m 0
for log in output/*/inner_log.parquet
do
    mv "$log" "$(dirname $log)/log_c1_m0.parquet"
done

echo "Simulating evolution (crossover=False, migration=True)"
python3 ./run_experiments.py -c 0 -m 1
for log in output/*/inner_log.parquet
do
    mv "$log" "$(dirname $log)/log_c0_m1.parquet"
done

echo "Simulating evolution (crossover=True, migration=True)"
python3 ./run_experiments.py -c 1 -m 1
for log in output/*/inner_log.parquet
do
    cp "$log" "$(dirname $log)/log_c1_m1.parquet"
done

echo "Generating charts for single run..."
python3 ./chart.py

echo "Rendering visualizations of results single run..."
python3 ./render.py

echo "Generating charts comparing all runs..."
python3 ./chart_across_experiments.py

echo "Done."

