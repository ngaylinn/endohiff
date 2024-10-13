#!/bin/bash

rm -rf output

echo "Simulating evolution..."
python3 ./run_experiments.py

echo "Generating charts of progress..."
python3 ./chart.py

echo "Rendering visualizations of results..."
python3 ./render.py

echo "Done."
