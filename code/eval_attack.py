
import os
import random
import argparse
from argparse import Namespace
from pprint import pprint
from functools import partial
from pandas.core.frame import DataFrame 
import pandas as pd
import numpy  # for gradio hot reload
from watermark_processor_Green import WatermarkLogitsProcessor_with_preferance, WatermarkDetector_with_preferance
from attack import attack_process
import read_json, read_json_term
from tqdm import tqdm
import torch
import gc
torch.manual_seed(123)
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

from watermark import evaluate_generation_fluency_from_output

run_all = True

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          OPTForCausalLM,
                          LogitsProcessorList)


def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(
        description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--ppl",
        type=str2bool,
        default=False,
        help="To evaluate ppl instead of run generating and detecting",
    )
    parser.add_argument(
        "--attack_ep",
        type=float,
        default=0.1,
        help="attack epsilon. 0 for not attack",
    )
    parser.add_argument(
        "--wm_mode",
        type=str,
        default="combination",
        help="previous1 or combination",
    )
    parser.add_argument(
        "--detect_mode",
        type=str,
        default="normal",
        help="normal",
    )
    parser.add_argument(
        "--epsilon",
        type=int,
        default=0,
        help="intensity of attack",
    )

    parser.add_argument(
        "--gen_mode",
        type=str,
        default="depth_d",
        help="depth_d, normal",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="sub list number",
    )
    parser.add_argument(
        "--attack_type",
        type= str,
        default='gpt',
        help="type of attack",
    )
    parser.add_argument(
        "--dataset",
        type = str,
        default = 'c4',
        help = "c4, squad, xsum, PubMedQA, writingprompts",
    )
    parser.add_argument(
        "--user_dist",
        type=str,
        default="dense",
        help="sparse or dense",
    )
    parser.add_argument(
        "--user_magnitude",
        type=int,
        default=10,
        help="user number = 2**magnitude",
    )
    parser.add_argument(  
        "--delta",
        type=float,
        default=5,
        help="The (max) amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--decrease_delta",
        type=str2bool,
        default=False,
        help="Modify delta according to output length.",
        
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=25,  # 200
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=100,  
        help="Minimum number of new tokens to generate.",
    )
    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-1.3b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None, #None
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True, 
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7, #0.7
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=True,
        help="Whether to run model in float16 precsion.",
    )

    args = parser.parse_args()
    return args



def load_model(args):
    """Load and return the model and tokenizer"""

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    return tokenizer, device

def generate(prompt, args, model=None, device=None, tokenizer=None,index=None, title=None, paperlist=None):
    if title == None:
        raise ValueError("Error: 'title' cannot be None.")
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens,min_new_tokens=args.min_new_tokens)

    watermark_processor = WatermarkLogitsProcessor_with_preferance(title=title,
                                                                paperlist=paperlist,
                                                                vocab=list(tokenizer.get_vocab().values()),
                                                                args=args
                                                                )
    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))
    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )

    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]),
        return_dict_in_generate=True, 
        output_scores=True,
        **gen_kwargs
    )
    if args.prompt_max_length:
        # Use user-specified value, ensuring it's within the model's limit
        args.prompt_max_length = min(args.prompt_max_length, model.config.max_position_embeddings)
    else:
        # Default to the model's full capacity minus generated tokens
        args.prompt_max_length = model.config.max_position_embeddings - args.max_new_tokens

    tokd_input = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True, truncation=True,
                           max_length=args.prompt_max_length, return_dict=True).to(device)
    
    tokd_input = {k: v.to(model.device) for k, v in tokd_input.items()}
    
    if isinstance(tokd_input, torch.Tensor):
        tokd_input = {"input_ids": tokd_input}
    
    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)
    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)
    out = generate_with_watermark(**tokd_input)
    torch.manual_seed(args.generation_seed)
    output_with_watermark = out[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]
    
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    # print(decoded_output_with_watermark)
    # print("----------------------------------------------")
    # print(decoded_output_without_watermark)
    return (tokd_input["input_ids"].shape[-1],
            output_with_watermark.shape[-1],
            redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark,
            decoded_output_with_watermark,
            watermark_processor,
            args)

def detect(input_text, args, device=None, tokenizer=None, title=None, paperlist=None):

    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    watermark_detector = WatermarkDetector_with_preferance(vocab=list(tokenizer.get_vocab().values()),
                                                           gamma=args.gamma,
                                                           device=device,
                                                           tokenizer=tokenizer,
                                                           z_threshold=args.detection_z_threshold,
                                                           normalizers=args.normalizers,
                                                           ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                                           select_green_tokens=args.select_green_tokens,
                                                           title=title,
                                                           paperlist=paperlist,
                                                           args=args)
    
    score_dict, gr_score, mark = watermark_detector.detect(input_text)
    return score_dict, gr_score, mark

import json
import os
import numpy as np

import json
import os
import numpy as np

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

def return_watermarked_text(result_dict, title):
    """
    Returns the watermarked text for a given title from the result dictionary.

    Parameters:
    - result_dict (list of dicts): List of dictionaries containing paper details.
    - title (str): Title of the paper for which the watermarked text is needed.

    Returns:
    - str: Watermarked peer review text if found, otherwise None.
    """
    for entry in result_dict:
        if entry.get("title") == title:
            output_without = entry.get("output_without")
            output_with = entry.get("output_with")
            return entry.get("peer_review_with_watermark", None), entry.get("peer_review_without_watermark", None), output_without,output_with
    return None  # Return None if title not found


def main(args):
    # Load datasets
    with open(f"result/Outputs_{args.gamma}_GD{args.delta}_Greeen_term.json") as f:
        result_dic = json.load(f)
    
    type_attack = ''

    checkpoint_file = f"result_attack/attack_checkpoint_{args.gamma}_GD{args.delta}.json"
    output_file = f"result_attack/attack_{args.gamma}_GD{args.delta}_Greeen_term_attack.json"
    tokenizer, device = load_model(args)

    # Load existing progress if checkpoint exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            processed_data = json.load(f)
        processed_titles = {entry["title"] for entry in processed_data}
    else:
        processed_data = []
        processed_titles = set()

    if run_all==True:
        read_file = read_json.read_old()
    else:
        read_file = read_json.data
    
    term_data = read_json_term.data

    for each_dic in tqdm(read_file):
        title = each_dic['title']
        if title in processed_titles:
            print(f"Skipping already processed: {title}")
            continue  # Skip already processed entries

        abstract = each_dic['abstract']
        paper_text = each_dic['paper_text']
        paper_content = abstract + " "+ paper_text
        paperid = each_dic['paperid']
        term_paper_data = term_data[paperid]
        #paperlist = tokenizer(paper_content)['input_ids']
        paperlist = tokenizer(term_paper_data)['input_ids']

        decoded_output_with_watermark, decoded_output_without_watermark, output_without, output_with = return_watermarked_text(result_dic,title)
        
        new_decoded_output_with_watermark = attack_process(decoded_output_with_watermark, epsilon=args.epsilon, type=args.attack_type)
        
        # Detect scores for watermarked output
        if args.detect_mode == 'normal':
            gr_score_list = []
            max_sim = 0
            max_sim_idx = -1
            
            output_dict_with, gr_score_with, mark = detect(
                new_decoded_output_with_watermark,
                args,
                device=device,
                tokenizer=tokenizer,
                title=title,
                paperlist=paperlist
            )

        # Append the results to the list
        result = {
            "title": title,
            "abstract": abstract,
            "peer_review_without_watermark": decoded_output_without_watermark,
            "peer_review_with_watermark": decoded_output_with_watermark,
            "peer_review_with_watermark_attack": new_decoded_output_with_watermark,
            "gr_score_with": safe_serialize(gr_score_with),  # Serialize safely
            "output_with_attack": safe_serialize(output_dict_with),
            "output_with": safe_serialize(output_with),
            "output_without": safe_serialize(output_without),
        }

        processed_data.append(result)
        processed_titles.add(title)

        # Save checkpoint after each iteration
        with open(checkpoint_file, "w") as f:
            json.dump(processed_data, f, indent=4)
        
        del decoded_output_without_watermark
        del decoded_output_with_watermark
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Processed: {title}")

    # Save the final results to the output file
    with open(output_file, "w") as f:
        json.dump(processed_data, f, indent=4)

    # Remove the checkpoint file once done
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print(f"Saved results to {output_file}")
            

if __name__ == "__main__":
    args = parse_args()
    main(args)
