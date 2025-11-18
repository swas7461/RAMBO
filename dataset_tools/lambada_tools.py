# eval_no_harness.py
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy
from utils import *


@torch.no_grad()
def seq_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Return token-level logprobs for each next token.
    output shape = (batch, seq_len-1)
    logprob[i,t] = log P(x_{t+1} | x_{<=t})
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]                     # predict next token
    targets = input_ids[:, 1:]                         # next tokens
    logps = torch.log_softmax(logits, dim=-1)
    token_logp = logps.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logp

def truncate_pair(tokenizer, ctx_ids: List[int], cont_ids: List[int], max_len: int) -> Tuple[List[int], List[int]]:
    """
    Ensure len(ctx)+len(cont) <= max_len. Trim context from the left if needed.
    """
    total = len(ctx_ids) + len(cont_ids)
    if total <= max_len:
        return ctx_ids, cont_ids
    # keep at least 1 token of context
    cut = total - max_len
    cut = min(cut, max(0, len(ctx_ids) - 1))
    return ctx_ids[cut:], cont_ids

def option_logprob(model, tokenizer, context: str, option: str, device, normalize=True):
    # tokenize separately to know the boundary
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    opt_ids = tokenizer.encode(option, add_special_tokens=False)

    max_len = getattr(model.config, "max_position_embeddings", 2048)
    ctx_ids, opt_ids = truncate_pair(tokenizer, ctx_ids, opt_ids, max_len)

    full_ids = ctx_ids + opt_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    token_logp = seq_logprobs(model, input_ids, attention_mask)  # (1, L-1)
    # We only want logprobs for option tokens, i.e., positions covering opt_ids
    # These correspond to indices [len(ctx_ids)-1 : len(full_ids)-1] in token_logp
    start = len(ctx_ids) - 1
    start = max(start, 0)  # handle empty context edge case
    end = len(full_ids) - 1
    opt_logp = token_logp[0, start:end].sum()
    if normalize and (end - start) > 0:
        opt_logp = opt_logp / (end - start)
    return float(opt_logp)

# ---------------------------
# LAMBADA (token-level next-token accuracy)
# ---------------------------
def eval_lambada(model, tokenizer, device, size = None):
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    correct = 0
    total = 0
    for ex in ds:
        text = ex["text"]
        # tokenize, split last token
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 2:
            continue
        ctx_ids, last_id = ids[:-1], ids[-1]

        max_len = getattr(model.config, "max_position_embeddings", 2048)
        if len(ctx_ids) >= max_len:
            ctx_ids = ctx_ids[-(max_len - 1):]

        input_ids = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
        next_token = int(logits[0, -1].argmax(-1))
        correct += (next_token == last_id)
        total += 1
        if (size is not None) and (total >= size):
            break
    acc = correct / total if total else 0.0
    print(f"LAMBADA accuracy: {acc:.4f}  (n={total})")
    return acc

def get_attack_acc_lambada(model, tokenizer, saved_state_dict, gradients, weight_indices):
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

    acc = eval_lambada(model, tokenizer, model.device, size = 150)
    print(name, 'Acc:', acc,'number of indices:', count)
                
    custom_load_state_dict(model, saved_state_dict)
    clear_memory()

    return acc