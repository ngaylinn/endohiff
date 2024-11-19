#!/bin/bash

# If any of the below commands fail, exit immediately.
set -e

# Delete old outputs if they exist.
rm -rf output

echo "Simulating evolution..."
python3 ./run_experiments.py

echo "Generating charts for single experiments..."
python3 ./chart.py

echo "Generating charts comparing experiments..."
python3 ./chart_across_experiments.py

echo "Rendering visualizations of results..."
python3 ./render.py

echo "Generating supplemental figures..."
python3 ./supplemental_figures.py

echo "Done."
