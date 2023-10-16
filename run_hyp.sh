#!/bin/bash

# Declare the arrays
DECAYS=(0.0 0.1 1.0)
LRS=(0.0001 0.00001 0.000001)

# Output file
OUTPUT_FILE="model_outputs.txt"

# Ensure the output file is empty before starting
> $OUTPUT_FILE

best_value=99999999  # initialize with a large value
best_hyperparams=""

# Loop over the values in DECAYS and LRS
for DECAY in "${DECAYS[@]}"
do
    for LR in "${LRS[@]}"
    do
              # Store the command in a variable
        CMD="python3 main.py --config configs/evals/config_noimdb_debug.yaml \
                             --lr $LR \
                             --num_epochs 0 \
                             --weight_decay $DECAY"

        # Display and execute the command
        echo "$CMD"
        eval $CMD | tee -a $OUTPUT_FILE

        # Extract the mean LatencyAE from the output and compare to the best so far
        current_value=$(awk -F ': ' '/Latency[[:space:]]+AE: mean:/ {print $2}' "$OUTPUT_FILE" | tail -1)

        # Check if current_value is valid
        if [[ ! -z "$current_value" ]] && (( $(echo "$current_value < $best_value" | bc -l) )); then
            best_value=$current_value
            best_hyperparams="LR: $LR, DECAY: $DECAY"
        fi

      echo "---------------------------------------------"
      echo "Best model so far based on LatencyAE:mean value is:"
      echo "Value: $best_value"
      echo "Hyperparameters: $best_hyperparams"

    done
done

echo "---------------------------------------------"
echo "Best model based on LatencyAE:mean value is:"
echo "Value: $best_value"
echo "Hyperparameters: $best_hyperparams"

