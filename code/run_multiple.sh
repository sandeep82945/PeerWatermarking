#!/bin/bash

# Define arrays for gamma and delta values
gamma_values=(0.2 0.4 0.6 0.8 1.0)
delta_values=(1.0 2.0 3.0 4.0 5.0)

# Loop through each combination of gamma and delta
for gamma in "${gamma_values[@]}"; do
    for delta in "${delta_values[@]}"; do
        echo "Running script with gamma=$gamma and delta=$delta"
        
        # Run the Python script with varying gamma and delta values
        python test_paperGreen_without.py \
            --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
            --max_new_tokens 2000 \
            --min_new_tokens 5 \
            --gamma "$gamma" \
            --delta "$delta" \
            --attack_ep 0.0 \
            --ppl False \
            --load_fp16 True
        
        # Optional: Save output to a log file
        # python test_paperGreen_without.py ... > "output_gamma_${gamma}_delta_${delta}.log"
    done
done

echo "All runs completed."