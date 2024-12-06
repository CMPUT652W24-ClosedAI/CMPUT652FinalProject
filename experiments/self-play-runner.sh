#!/bin/bash

# Path to the Python script
PYTHON_SCRIPT="ppo_self_play.py"
SEED=$RANDOM


# Loop 30 times
for i in {1..10}
do
    # Generate a random seed using $RANDOM
    echo "Running iteration $i with seed $SEED..."
    python "$PYTHON_SCRIPT" --seed "$SEED" --mapName "map-$i"
done

echo "All iterations complete."
