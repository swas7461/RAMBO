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
import opt_utils as opt
import plot_utils as pt
from datasets import load_dataset
import copy
import argparse

from dataset_tools import lambada_tools as lambada
from dataset_tools import arceasy_tools as arceasy
from dataset_tools import hellaswag_tools as hellaswag
from dataset_tools import piqa_tools as piqa
from dataset_tools import openbookqa_tools as openbookqa
from dataset_tools import winogrande_tools as winogrande
from dataset_tools import mmlu_tools as mmlu

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

    model_id = model_name.split('/')[1]

    with open(f"./sensitivity_analysis/{model_id}_initial_weights_subset.json", "r") as f:
        weight_indices = json.load(f)

    reduced_weight_indices = opt.exclusionary_optimization(model, tokenizer, saved_state_dict, gradients, weight_indices, \
                                                           ut.get_attack_loss, wiki_data, device, args.loss_threshold, \
                                                            loss_tol=1e-3, max_iter=100, verbose=True)
    
    with open(f"./final_weight_results/{model_id}_critical_weights.json", "w") as f:
        json.dump(reduced_weight_indices, f, indent=4, default=ut.serialize)

    pt.plot_optimization(reduced_weight_indices, model_id)

    acc_lambada = lambada.get_attack_acc_lambada(model, tokenizer, saved_state_dict, gradients, reduced_weight_indices)
    print(f'Post Attack Accuracy of LAMBADA: {acc_lambada*100:.4f}%')

    acc_hellaswag = hellaswag.get_attack_acc_hellaswag(model, tokenizer, saved_state_dict, gradients, reduced_weight_indices)
    print(f'Post Attack Accuracy of Hellaswag: {acc_hellaswag*100:.4f}%')
    
    acc_piqa = piqa.get_attack_acc_piqa(model, tokenizer, saved_state_dict, gradients, reduced_weight_indices)
    print(f'Post Attack Accuracy of PIQA: {acc_piqa*100:.4f}%')
    
    acc_openbookqa = openbookqa.get_attack_acc_openbookqa(model, tokenizer, saved_state_dict, gradients, reduced_weight_indices)
    print(f'Post Attack Accuracy of OpenbookQA: {acc_openbookqa*100:.4f}%')
    
    acc_winogrande = winogrande.get_attack_acc_winogrande(model, tokenizer, saved_state_dict, gradients, reduced_weight_indices)
    print(f'Post Attack Accuracy of Winogrande: {acc_winogrande*100:.4f}%')

    acc_mmlu = mmlu.get_attack_acc_mmlu(model, tokenizer, saved_state_dict, gradients, reduced_weight_indices)
    print(f'Post Attack Accuracy of MMLU: {acc_mmlu*100:.4f}%')

    ppl_wikitext = ut.get_attack_loss(model, tokenizer, saved_state_dict, gradients, weight_indices, device, wiki_data)[1]
    print(f'Post Attack Perplexity of WikiText: {ppl_wikitext:.4f}')
    

if __name__ == "__main__":

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
        "--loss_threshold",
        default=10,
        help="Loss Threshold"
    )

    args = parser.parse_args()

    main(args)