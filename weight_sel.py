import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from multiprocessing import Process, get_context
from typing import Any, Callable, Dict, List, Type

import analysisTools as ats
import mmluToolSet as mts
import pandas as pd
import strategies as strat
import toolSet as ts
import torch
from filelock import FileLock
from lm_eval import evaluator  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402
from torch import Tensor
from transformers import MambaForCausalLM, PreTrainedModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
PreTrainedTokenizerType = PreTrainedTokenizer or PreTrainedTokenizerFast

import utils as ut
import plot_utils as pt
from datasets import load_dataset
import copy
import argparse

def main(args):
    ut.clear_memory()
    # device_name = "cuda:0"
    device_name = args.device
    # model_name = "state-spaces/mamba-1.4b-hf"
    model_name = args.model
    device = torch.device(device_name)
    model, tokenizer = ut.load_model(model_name, model_name, device=device)

    model_size = model.get_memory_footprint()/ 1024 / 1024 /1024
    print(f'Model Size: {model_size:.4f} GB')

    wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ppl, loss, _ = ts.calculate_perplexity_loss_and_length(model, tokenizer, wiki_data, device=device, size=20)
    print(f'WikiText Perplexity: {ppl:.4f}; Model Loss: {loss:.4f}')

    accuracy_flag: bool = True
    acf = ut.get_acc_func(['arc_easy'], limit=128, batch_size=8)
    original_accuracy: float = (
        acf(model, tokenizer, device_name)#[0]["acc"] if accuracy_flag else 0.0
    )
    print(f'ARC-Easy Accuracy: {original_accuracy[0]["acc"]*100:.4f}')

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    optimizer.zero_grad()
    ut.get_gradients_wikidata(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,  # Optimizer isn't stepped here
        dataset=wiki_data,  # HF-style dataset with "text" field
        size=128,           # Process 128 examples
        batch_size=4,
        device=device
    )
    gradients: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        gradients[name] = param.grad

    ut.clear_memory()
    saved_state_dict = copy.deepcopy(model.state_dict())

    layer_sensitivity = ut.sensitivity_ablation(model, tokenizer, optimizer, saved_state_dict, gradients, wiki_data, device)

    flattened_data = []
    for layer, sensitivities in layer_sensitivity.items():
        percent_of_weights_flipped = sensitivities['percentage_of_weights_flipped']
        number_of_weights_flipped = sensitivities['number_of_weights_flipped']
        for sensitivity_type, values in sensitivities.items():
            if sensitivity_type not in ['percentage_of_weights_flipped', 'number_of_weights_flipped']:
                for index, value in enumerate(values):
                    flattened_data.append({
                        'Layer': layer,
                        'Attack_type': sensitivity_type,
                        'percentage_of_weights_flipped': percent_of_weights_flipped[index],
                        'number_of_weights_flipped': number_of_weights_flipped[index],
                        'Model_loss': value
                    })
    # Create a DataFrame
    df = pd.DataFrame(flattened_data)
    # Save to CSV
    model_id= model_name.split('/')[1]
    df.to_csv(f'./sensitivity_analysis/{model_id}_layers_sensitivity_fp16_sign_bit_flip_all_layers_hybrid.csv', index=False)

    filter_strings = ['embeddings.weight','norm.weight','mixer.A_log','mixer.D','mixer.conv1d.weight',\
                      'mixer.conv1d.bias','mixer.in_proj.weight','mixer.x_proj.weight','mixer.dt_proj.weight',\
                        'mixer.dt_proj.bias','mixer.out_proj.weight','norm_f.weight']#['in_proj', 'out_proj', 'conv1d', 'norm', 'x_proj', 'dt_proj']

    def get_layer_type(layer, filters):
        for f in filters:
            if f in layer:
                return f
        return 'other'

    df['Layer_type'] = df['Layer'].apply(lambda x: get_layer_type(x, filter_strings))

    df['loss_fn'] = (df['Model_loss']) / df['number_of_weights_flipped']
    df = df[df['number_of_weights_flipped'] != 0]
    df.to_csv(f'./sensitivity_analysis/{model_id}_layers_sensitivity_fp16_sign_bit_flip_all_layers_filtered_hybrid.csv', \
              index=False)
    df = pd.read_csv(f'./sensitivity_analysis/{model_id}_layers_sensitivity_fp16_sign_bit_flip_all_layers_filtered_hybrid.csv')
    filtered_df = df[df['Model_loss']>args.loss_threshold]
    filtered_df = filtered_df[filtered_df['Attack_type'] == 'Hybrid']
    string_to_remove='norm'
    # filtered_df = filtered_df['norm' not in filtered_df['Layer']]
    filtered_df=filtered_df[~filtered_df['Layer'].str.contains(string_to_remove, case=False, na=False)]
    filtered_df=filtered_df[~filtered_df['Layer'].str.contains('bias', case=False, na=False)]
    # sorted_df = filtered_df.sort_values(by=['number_of_weights_flipped', 'Model_loss'], ascending=True)
    sorted_df = filtered_df.sort_values(by=['number_of_weights_flipped', 'Model_loss'],ascending=[True, False])

    sensitive_layer_type = sorted_df['Layer_type'].iloc[0]
    sensitive_layer = sorted_df['Layer'].iloc[0]
    print(f'Sensitive Layer: {sensitive_layer}')

    selected_layers = ['mixer.A_log','mixer.D', 'mixer.conv1d.weight', 'mixer.in_proj.weight', \
                       'mixer.x_proj.weight', 'mixer.dt_proj.weight', 'mixer.out_proj.weight']

    selected_layers_new = ['A_log', 'D', 'conv1d', 'in_proj', 'x_proj', 'dt_proj', 'out_proj']

    pt.plot_layer_sensitivity(df, model_id, loss, sensitive_layer_type)
    pt.plot_layer_selection_loss(df, model_id, loss, selected_layers, selected_layers_new)
    pt.plot_layer_selection_loss_fn(df, model_id, selected_layers, selected_layers_new)

    subset_sensitivity = ut.sensitivity_ablation_for_layer(model, tokenizer, optimizer, saved_state_dict, \
                                                           gradients, sensitive_layer, wiki_data, device)

    layer = sensitive_layer
    data = subset_sensitivity[layer]
    df = pd.DataFrame({
        'number_of_weights_flipped': data['number_of_weights_flipped'],
        'percentage_of_weights_flipped': data['percentage_of_weights_flipped'],
        'Magnitude': data['Magnitude'],
        'Gradient': data['Gradient'],
        'alpha_0.25': data['alpha_0.25'],
        'alpha_0.5': data['alpha_0.5'],
        'alpha_0.75': data['alpha_0.75'],
    })
    df['Layer'] = layer
    df.to_csv(f'./sensitivity_analysis/{model_id}_{layer}_sensitivity.csv', index=False)

    alpha_label = 'alpha_'+str(args.alpha)
    filtered_df = df[df[alpha_label]>args.loss_threshold]
    
    filtered_df = filtered_df[['Layer', 'number_of_weights_flipped', 'percentage_of_weights_flipped', alpha_label]]
    string_to_remove='norm'
    filtered_df=filtered_df[~filtered_df['Layer'].str.contains(string_to_remove, case=False, na=False)]
    filtered_df=filtered_df[~filtered_df['Layer'].str.contains('bias', case=False, na=False)]
    sorted_df = filtered_df.sort_values(by=['number_of_weights_flipped'],ascending=[True])
    percent_of_weight = (sorted_df['percentage_of_weights_flipped'].iloc[0]) + sorted_df['percentage_of_weights_flipped'].iloc[0]/2

    pt.plot_alpha_ablation(subset_sensitivity, model_id, sensitive_layer)

    weight_indices = ut.get_weights(model, tokenizer, optimizer, saved_state_dict, \
                                    gradients, percent_of_weight=percent_of_weight, layers=[sensitive_layer], alpha=args.alpha)
    
    with open(f"./sensitivity_analysis/{model_id}_initial_weights_subset.json", "w") as f:
        json.dump(weight_indices, f, indent=4, default=ut.serialize)

    loss_post_attack = ut.get_attack_loss(model, tokenizer, saved_state_dict, gradients, weight_indices, device, wiki_data)
    print(f"Post Attack Loss: {loss_post_attack[0]:.4f}, Post Attack Perplexity: {loss_post_attack[1]:.4f}")
    print(f"Initial Weight Subset Length: {len(weight_indices[list(weight_indices.keys())[0]]['weight_indices'])}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="For a huggingface mamba model, perform RAMBO's initial weight subset selection."
    )
    parser = argparse.ArgumentParser(
        description="For a huggingface mamba model, perform RAMBO's weight optimization."
    )
    # Positional argument: required
    parser.add_argument(
        "--model", 
        required=True,
        choices=[
        "state-spaces/mamba-1.4b-hf",
        "state-spaces/mamba-2.8b-hf",
        "state-spaces/mamba-370m-hf",
        "AntonV/mamba2-370m-hf",
        "AntonV/mamba2-1.3b-hf",
        "AntonV/mamba2-2.7b-hf",
    ],
    help="HuggingFace Mamba model repository (must be one of the supported models)"
    )

    parser.add_argument(
        "--device",
        default='cpu',
        help="Torch device"
    )

    parser.add_argument(
        "--alpha",
        default=0.25,
        help="Imp-score value"
    )

    parser.add_argument(
        "--loss_threshold",
        default=10,
        help="Loss Threshold"
    )

    args = parser.parse_args()

    main(args)