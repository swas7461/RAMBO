import gc
import torch
from transformers import MambaForCausalLM, PreTrainedModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
PreTrainedTokenizerType = PreTrainedTokenizer or PreTrainedTokenizerFast

from typing import Any, Callable, Dict, List, Type, Tuple
import toolSet as ts
from lm_eval import evaluator  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import copy
import numpy as np

def load_model(
    model_name: str,
    tokenizer_name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> tuple[MambaForCausalLM, PreTrainedTokenizer ]:
    """
    # Configure BitsAndBytes for int8 quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Set for 8-bit loading
        # bnb_8bit_quant_type=mode,  # Keep 'int8' mode (change to your preferred quantization method if any)
        # bnb_8bit_compute_dtype=torch.float16,  # Use float16 for computation, can change if needed
        # bnb_8bit_use_double_quant=True,  # Double quantization may improve performance
    )
    """

    model = MambaForCausalLM.from_pretrained(
        model_name,
        device_map=device,  # Automatically map model layers to devices
        # quantization_config=bnb_config,
        torch_dtype=dtype,  # float16 is standard for compute with quantization
        # trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True, use_fast=True
    )
    tokenizer.padding_side = "left"

    return model, tokenizer

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def get_acc_func(
    challenges: List[str], limit: int = 600, batch_size: int = 8
) -> Callable[
    [
        PreTrainedModel,
        ts.PreTrainedTokenizerType,
        str,
    ],
    List[Dict[str, Any]],
]:
    if len(challenges) == 0:
        raise ValueError("Challenges list cannot be empty")

    def inner_acc_func(
        model: PreTrainedModel,
        tokenizer: ts.PreTrainedTokenizerType,
        device_name: str,
    ) -> List[Dict[str, Any]]:
        post_attack_results = None

        lm_eval_model = HFLM(
            pretrained=model,
            batch_size=batch_size,
            device=device_name,
            tokenizer=tokenizer,
            disable_tqdm=True,
        )
        post_attack_results = evaluator.simple_evaluate(
            verbosity="ERROR",
            model=lm_eval_model,
            tasks=challenges,  # ARC-Easy challenge
            num_fewshot=0,  # Zero-shot evaluation
            limit=limit,
            device=device_name,
        )
        # arc_challenge
        # hellaswag
        # arc_easy
        # mmlu_astronomy
        result_list = []
        for challenge in challenges:
            result_list.append(
                {
                    "name": challenge,
                    "acc": post_attack_results["results"][challenge]["acc,none"],
                    "result": post_attack_results,
                }
            )
        return result_list

    return inner_acc_func

def get_gradients_wikidata(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerType,
    optimizer,
    dataset,
    size: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    
    total_loss = 0
    total_length = 0
    i = 0  # random.randrange(0, len(dataset['text'])-size)
    for example in dataset["text"][i : size + i]:
        if example != "":
            input_text = example
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss = torch.nan_to_num(loss)
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_length += inputs["input_ids"].size(1)
    optimizer.zero_grad()
    grad_flag = False
    if total_loss.backward():
        grad_flag = True

    perplexity = torch.exp(torch.tensor(total_loss / total_length))

    return perplexity.item(), grad_flag

def get_gradients_wikidata(
    model: torch.nn.Module,
    tokenizer,
    optimizer,
    dataset,
    size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.train()  # ensure model is in training mode
    model.zero_grad()

    total_loss = 0.0
    total_tokens = 0
    grad_norms = []

    # Prepare the dataset slice (only 'size' number of non-empty samples)
    examples = [ex for ex in dataset["text"] if ex.strip() != ""]
    examples = examples[:size]

    dataloader = DataLoader(examples, batch_size=batch_size, shuffle=False)

    for batch_texts in dataloader:
        # Tokenize the batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        break

    return True 



def flip_bits_in_tensor(tensor, bit_position):
    bit_mask = 1 << bit_position
    flipped_tensor = tensor ^ bit_mask
    return flipped_tensor
def custom_load_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state_dict:
            # print(name)
            if 'weight_format' not in name:
                model_state_dict[name].copy_(param)
 
def custom_load_state_dict_single_layer(model, state_dict, layer_name):
    model_state_dict = model.state_dict()
    model_state_dict[layer_name].copy_(state_dict[layer_name].to(model_state_dict[layer_name].dtype))

def importance_score(w, g, alpha=0.5):
    w_abs = w.detach().abs()
    w_min, w_max = w_abs.min(), w_abs.max()
    w_norm = (w_abs - w_min) / (w_max - w_min + 1e-8)  # Avoid division by zero

    # Normalize g (min-max normalization) in-place
    g_abs = g.detach().abs()
    g_min, g_max = g_abs.min(), g_abs.max()
    g_norm = (g_abs - g_min) / (g_max - g_min + 1e-8)  # Avoid division by zero

    # Compute score in a memory-efficient way using in-place operations
    score = (alpha * g_norm) + ((1 - alpha) * w_norm)

    return score

def sensitivity_ablation(model, tokenizer, optimizer, saved_state_dict, gradients, wiki_data, device):
    percent_of_weights = [0.001, 0.01, 0.1,1, 5, 10, 50]

    layer_sensitivity = {}
    clear_memory()
    custom_load_state_dict(model, saved_state_dict)
    #line 18
    for name, param in model.named_parameters():
        if (gradients[name] is not None):# and ('weight' in name):# and 'language_model.model.embed_tokens' not in name:
            
            attack_args = {'idx' : [0], 'attack_bit' : 7}

            sensitivity = {'Magnitude':[], 'Gradient':[],  'Hybrid':[], 'number_of_weights_flipped': [], \
                           'percentage_of_weights_flipped': percent_of_weights}
    
            k_tops =  [int((k/100)*gradients[name].detach().view(-1).size()[0]) for k in percent_of_weights]
            print(name, k_tops)
            w = param.data.detach().contiguous().view(-1)
            g = gradients[name].float().detach().view(-1)
            # print(w.shape)
    
            # imp_score = importance_score(w, g, alpha=0.5)  

            print(f'Layer name: {name}')
    
            for k_top in k_tops:
                # BFLIP
                print(f'k_top: {k_top}')
                wval, w_idx = w.detach().abs().reshape(-1).topk(k_top)
                gval, g_idx = gradients[name].detach().abs().view(-1).topk(k_top)
                imp_score = importance_score(w.detach().abs().reshape(-1), gradients[name].detach().abs().view(-1))
                ival, i_idx  = imp_score.topk(k_top)
                clear_memory()
    
                state_dict_copy = copy.deepcopy(saved_state_dict)
                if param.dtype==torch.int8:
                    w[w_idx] = flip_bits_in_tensor(w[w_idx], 7)
                    # print(attack_args['idx'])
                    state_dict_copy[name].data[:] = w.reshape(state_dict_copy[name].data.shape)
                else:
                    # print(w[w_idx])
                    w[w_idx] = -w[w_idx]
                    state_dict_copy[name] = w.reshape(state_dict_copy[name].data.shape)
                custom_load_state_dict(model, state_dict_copy)
                clear_memory()
    
                perplexity, l, _  = ts.calculate_perplexity_loss_and_length(model, tokenizer, wiki_data, \
                                                                            device=device, size=4)
                sensitivity['Magnitude'].append(l)
    
                print(name, "Magnitude based:",l)
    
                custom_load_state_dict(model, saved_state_dict)
                clear_memory()
            
                w = model.state_dict()[name].data.detach().reshape(-1)
                state_dict_copy = copy.deepcopy(saved_state_dict)
                if param.dtype==torch.int8:
                    w[g_idx] = flip_bits_in_tensor(w[g_idx], 7)
                    # print(attack_args['idx'])
                    state_dict_copy[name].data[:] = w.reshape(state_dict_copy[name].data.shape)
                else:
                    # print(w[w_idx])
                    w[g_idx] = -w[g_idx]
                    state_dict_copy[name] = w.reshape(state_dict_copy[name].data.shape)
                custom_load_state_dict(model, state_dict_copy)
                clear_memory()
    
                perplexity, l, _  = ts.calculate_perplexity_loss_and_length(model, tokenizer, wiki_data, \
                                                                            device=device, size=4)
    
                sensitivity['Gradient'].append(l)
    
                print(name, "Gradient based:",l)
    
                custom_load_state_dict(model, saved_state_dict)
                clear_memory()

                w = model.state_dict()[name].data.detach().reshape(-1)
                state_dict_copy = copy.deepcopy(saved_state_dict)
                if param.dtype==torch.int8:
                    w[i_idx] = flip_bits_in_tensor(w[i_idx], 7)
                    # print(attack_args['idx'])
                    state_dict_copy[name].data[:] = w.reshape(state_dict_copy[name].data.shape)
                else:
                    # print(w[w_idx])
                    w[i_idx] = -w[i_idx]
                    state_dict_copy[name] = w.reshape(state_dict_copy[name].data.shape)
                custom_load_state_dict(model, state_dict_copy)
                clear_memory()
    
                perplexity, l, _  = ts.calculate_perplexity_loss_and_length(model, tokenizer, wiki_data, \
                                                                            device=device, size=4)
    
                sensitivity['Hybrid'].append(l)
    
                print(name, "Imp-Score based:",l)
    
                custom_load_state_dict(model, saved_state_dict)
                clear_memory()
            
                custom_load_state_dict(model, saved_state_dict)
                sensitivity['number_of_weights_flipped'].append(k_top)
                print(name, "Sensitivity:",sensitivity)
                clear_memory()
            layer_sensitivity[name] = sensitivity
    return layer_sensitivity



def sensitivity_ablation_for_layer(model, tokenizer, optimizer, saved_state_dict, gradients, layer, wiki_data, device):
    percent_of_weights = [0.001, 0.005,0.008,  0.01,0.015, 0.02, 0.03, 0.05, 0.1,0.2, 1]

    layer_sensitivity = {}
    clear_memory()
    custom_load_state_dict(model, saved_state_dict)
    #line 18
    for name, param in model.named_parameters():
        if (gradients[name] is not None) and name == layer:# and ('weight' in name):# and 'language_model.model.embed_tokens' not in name:
            
            attack_args = {'idx' : [0], 'attack_bit' : 7}

            sensitivity = {'Magnitude':[], 'Gradient':[],  'number_of_weights_flipped': [], \
                           'percentage_of_weights_flipped': percent_of_weights, 'alpha_0.25':[], \
                            'alpha_0.5':[], 'alpha_0.75':[]}
    
            k_tops =  [int((k/100)*gradients[name].detach().view(-1).size()[0]) for k in percent_of_weights]
            print(name, k_tops)
            w = param.data.detach().contiguous().view(-1)
            g = gradients[name].float().detach().view(-1)
            # print(w.shape)
    
            # imp_score = importance_score(w, g, alpha=0.5)  

            print(f'Layer name: {name}')
    
            for k_top in k_tops:
                # BFLIP
                print(f'k_top: {k_top}')
                wval, w_idx = w.detach().abs().reshape(-1).topk(k_top)
                gval, g_idx = gradients[name].detach().abs().view(-1).topk(k_top)
                # ival, i_idx  = imp_score.topk(k_top)
                clear_memory()
    
                state_dict_copy = copy.deepcopy(saved_state_dict)
                if param.dtype==torch.int8:
                    w[w_idx] = flip_bits_in_tensor(w[w_idx], 7)
                    # print(attack_args['idx'])
                    state_dict_copy[name].data[:] = w.reshape(state_dict_copy[name].data.shape)
                else:
                    # print(w[w_idx])
                    w[w_idx] = -w[w_idx]
                    state_dict_copy[name] = w.reshape(state_dict_copy[name].data.shape)
                custom_load_state_dict(model, state_dict_copy)
                clear_memory()
    
                perplexity, l, _  = ts.calculate_perplexity_loss_and_length(model, tokenizer, wiki_data, \
                                                                            device=device, size=4)
                sensitivity['Magnitude'].append(l)
    
                print(name, "Magnitude based:",l)
    
                custom_load_state_dict(model, saved_state_dict)
                clear_memory()
            
                w = model.state_dict()[name].data.detach().reshape(-1)
                state_dict_copy = copy.deepcopy(saved_state_dict)
                if param.dtype==torch.int8:
                    w[g_idx] = flip_bits_in_tensor(w[g_idx], 7)
                    # print(attack_args['idx'])
                    state_dict_copy[name].data[:] = w.reshape(state_dict_copy[name].data.shape)
                else:
                    # print(w[w_idx])
                    w[g_idx] = -w[g_idx]
                    state_dict_copy[name] = w.reshape(state_dict_copy[name].data.shape)
                custom_load_state_dict(model, state_dict_copy)
                clear_memory()
    
                perplexity, l, _  = ts.calculate_perplexity_loss_and_length(model, tokenizer, wiki_data, \
                                                                            device=device, size=4)
    
                sensitivity['Gradient'].append(l)
    
                print(name, "Gradient based:",l)
    
                custom_load_state_dict(model, saved_state_dict)
                clear_memory()

                for a in [0.25, 0.5, 0.75]:
                    w = model.state_dict()[name].data.detach().reshape(-1)
                    imp_score = importance_score(w, g, alpha=a)
                    ival, i_idx  = imp_score.topk(k_top)
                    state_dict_copy = copy.deepcopy(saved_state_dict)
                    if param.dtype==torch.int8:
                        w[i_idx] = flip_bits_in_tensor(w[i_idx], 7)
                        # print(attack_args['idx'])
                        state_dict_copy[name].data[:] = w.reshape(state_dict_copy[name].data.shape)
                    else:
                        # print(w[w_idx])
                        w[i_idx] = -w[i_idx]
                        state_dict_copy[name] = w.reshape(state_dict_copy[name].data.shape)
                    custom_load_state_dict(model, state_dict_copy)
                    clear_memory()
        
                    perplexity, l, _  = ts.calculate_perplexity_loss_and_length(model, tokenizer, wiki_data, \
                                                                                device=device, size=8)
        
                    sensitivity['alpha_'+str(a)].append(l)
        
                    print(name, 'alpha_'+str(a),l)
        
                    custom_load_state_dict(model, saved_state_dict)
                    clear_memory()
            
                custom_load_state_dict(model, saved_state_dict)
                sensitivity['number_of_weights_flipped'].append(k_top)
                print(name, "Sensitivity:",sensitivity)
                clear_memory()
            layer_sensitivity[name] = sensitivity
    return layer_sensitivity


def get_weights(model, tokenizer, optimizer, saved_state_dict, gradients, percent_of_weight, layers, alpha):
    
    clear_memory()
    custom_load_state_dict(model, saved_state_dict)
    weights_dict = {}
    #line 18
    for name, param in model.named_parameters():
        for layer in layers:
            if gradients[name] is not None and name == layer:# and ('weight' in name):# and 'language_model.model.embed_tokens' not in name:        
                k_top =  int((percent_of_weight/100)*gradients[name].detach().view(-1).size()[0])
                print(name, k_top)
                w = param.data.detach().contiguous().view(-1)
                g = gradients[name].float().detach().view(-1)
                imp_score = importance_score(w, g, alpha=alpha)  
                ival, i_idx  = imp_score.topk(k_top)
                weights_dict[name] = {'weight_indices': i_idx, 'weight_values': w[i_idx], \
                                      'gradient_values': g[i_idx], 'importance_values': imp_score[i_idx]}
    return weights_dict

def serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def get_attack_acc(model, tokenizer, saved_state_dict, gradients, weight_indices, device_name, wiki_data):
    clear_memory()
    custom_load_state_dict(model, saved_state_dict)
    #line 18
    count = 0
    l, perplexity=0,0
    for name, param in model.named_parameters():
        for layer in weight_indices.keys():
            if name == layer:
                # print(name, perplexity, l)
                g_idx = weight_indices[name]['weight_indices']
                w = model.state_dict()[name].data.detach().reshape(-1)
                state_dict_copy = copy.deepcopy(saved_state_dict)
                count+=len(g_idx)
                if param.dtype==torch.int8:
                    w[g_idx] = flip_bits_in_tensor(w[g_idx], 7)
                    # print(attack_args['idx'])
                    state_dict_copy[name].data[:] = w.reshape(state_dict_copy[name].data.shape)
                else:
                    # print(w[w_idx])
                    w[g_idx] = -w[g_idx]
                    state_dict_copy[name] = w.reshape(state_dict_copy[name].data.shape)
                custom_load_state_dict(model, state_dict_copy)
                clear_memory()

    perplexity, l, _  = ts.calculate_perplexity_loss_and_length(model, tokenizer, wiki_data, \
                                                                device=torch.device(device_name),size=4)
    accuracy_flag: bool = True
    acf = get_acc_func(['arc_easy'], limit=128, batch_size=8)
    original_accuracy: float = (
        acf(model, tokenizer, device_name)[0]["acc"] if accuracy_flag else 0.0
    )
    acc=original_accuracy*100
    print(name, 'Perplexity:', perplexity,'Loss:', l,'number of indices:', count, 'accuracy:',acc)
                
    custom_load_state_dict(model, saved_state_dict)
    clear_memory()

    return l, perplexity, acc

def get_attack_loss(model, tokenizer, saved_state_dict, gradients, weight_indices, device, wiki_data):
    clear_memory()
    custom_load_state_dict(model, saved_state_dict)
    #line 18
    count = 0
    l, perplexity=0,0
    for name, param in model.named_parameters():
        for layer in weight_indices.keys():
            if name == layer:
                # print(name, perplexity, l)
                g_idx = weight_indices[name]['weight_indices']
                w = model.state_dict()[name].data.detach().reshape(-1)
                state_dict_copy = copy.deepcopy(saved_state_dict)
                count+=len(g_idx)
                if param.dtype==torch.int8:
                    w[g_idx] = flip_bits_in_tensor(w[g_idx], 7)
                    # print(attack_args['idx'])
                    state_dict_copy[name].data[:] = w.reshape(state_dict_copy[name].data.shape)
                else:
                    # print(w[w_idx])
                    w[g_idx] = -w[g_idx]
                    state_dict_copy[name] = w.reshape(state_dict_copy[name].data.shape)
                custom_load_state_dict(model, state_dict_copy)
                clear_memory()

    perplexity, l, _  = ts.calculate_perplexity_loss_and_length(model, tokenizer, wiki_data, device=device, size=4)
    print(name, 'Perplexity:', perplexity,'Loss:', l,'number of indices:', count)
                
    custom_load_state_dict(model, saved_state_dict)
    clear_memory()

    return l, perplexity