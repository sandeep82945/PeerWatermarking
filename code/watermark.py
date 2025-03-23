import torch
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def evaluate_generation_fluency_from_output(input_p_output, baseline_output, model_name, model, tokenizer, device):
    """
    Evaluate generation fluency using the provided model and tokenizer.

    Args:
        input_p_output (str): Concatenated input and generated output.
        baseline_output (str): Generated output without watermark.
        model_name (str): Name of the model used.
        model: Preloaded model for evaluation.
        tokenizer: Preloaded tokenizer corresponding to the model.

    Returns:
        tuple: (loss, perplexity) of the generated text.
    """
    with torch.no_grad():
        # Tokenize input and output
        tokd_prefix = tokenizer(input_p_output, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings)
        tokd_suffix = tokenizer(baseline_output, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings)

        tokd_inputs = tokd_prefix["input_ids"].to(device)
        tokd_suffix_len = tokd_suffix["input_ids"].shape[1]

        # Create labels to calculate loss only for the generated portion
        tokd_labels = tokd_inputs.clone()
        tokd_labels[:, :-tokd_suffix_len] = -100

        # Forward pass through the model
        outputs = model(input_ids=tokd_inputs, labels=tokd_labels)
        loss = outputs.loss

        # Calculate perplexity
        perplexity = math.exp(loss.item()) if loss.item() < 20 else float("inf")  # Handle overflow

    return loss.item(), perplexity