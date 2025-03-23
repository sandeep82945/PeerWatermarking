import os
import json
import numpy as np
from tqdm import tqdm
import gc
import torch

# Placeholder for multiple attack functions
def attack_function_1(watermarked_text):
    """First attack type - Replace with actual attack logic"""
    return watermarked_text  # Modify this

def attack_function_2(watermarked_text):
    """Second attack type - Replace with actual attack logic"""
    return watermarked_text  # Modify this

def attack_function_3(watermarked_text):
    """Third attack type - Replace with actual attack logic"""
    return watermarked_text  # Modify this

# Define attack strategies
ATTACKS = {
    "attack_1": attack_function_1,
    "attack_2": attack_function_2,
    "attack_3": attack_function_3
}

def safe_serialize(obj):
    """
    Helper function to safely serialize objects to JSON.
    Converts numpy objects and handles None gracefully.
    """
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None:
        return None
    return obj

def apply_attacks_and_save(input_file, output_file):
    """
    Reads the JSON file, applies multiple attack functions on the watermarked text, 
    and stores a new JSON file with the updated content.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    # Load the JSON data
    with open(input_file, "r") as f:
        data = json.load(f)

    updated_data = []
    
    for entry in tqdm(data, desc="Applying multiple attacks"):
        title = entry.get("title", "Unknown")

        # Apply each attack and store results
        if "peer_review_with_watermark" in entry:
            entry["attacked_versions"] = {}  # Store results for each attack type
            
            for attack_name, attack_fn in ATTACKS.items():
                attacked_text = attack_fn(entry["peer_review_with_watermark"])
                entry["attacked_versions"][attack_name] = attacked_text
        
        updated_data.append(entry)

        # Free up memory
        torch.cuda.empty_cache()
        gc.collect()

    # Save the modified data to a new JSON file
    with open(output_file, "w") as f:
        json.dump(updated_data, f, indent=4)

    print(f"Updated results with attacks saved to {output_file}")

if __name__ == "__main__":
    gamma = 0.5  # Replace with actual gamma value
    delta = 3  # Replace with actual delta value

    input_file = f"result/Outputs_{gamma}_GD{delta}_Greeen_term.json"
    output_file = f"result/Outputs_Attacked.json"

    apply_attacks_and_save(input_file, output_file)

