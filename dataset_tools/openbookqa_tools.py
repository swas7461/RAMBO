# eval_no_harness.py
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import List, Optional, Tuple

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
# OpenBookQA (MC: 4 choices)
# ---------------------------
def eval_openbookqa(model, tokenizer, device, size = None):
    # 'main' config has labels; validation used for scoring
    ds = load_dataset("openbookqa", "main", split="validation")
    correct = 0
    total = 0
    for ex in ds:
        stem = ex["question_stem"].strip()
        choices = ex["choices"]["text"]  # list[str]
        labels = ex["choices"]["label"]  # ['A','B','C','D']
        gold_letter = ex["answerKey"].strip()
        gold = labels.index(gold_letter)

        ctx = f"Question: {stem}\nAnswer:"
        scores = [option_logprob(model, tokenizer, ctx + " ", c.strip(), device, normalize=True) for c in choices]
        pred = int(max(range(len(choices)), key=lambda i: scores[i]))
        correct += (pred == gold)
        total += 1
        if (size is not None) and (total >= size):
            break
    acc = correct / total
    print(f"OPENBOOKQA accuracy: {acc:.4f}  (n={total})")
    return acc

@torch.no_grad()
def _generate_text(model, tokenizer, prompt: str, device, max_new_tokens=40, temperature=0.0, do_sample=False):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Only the newly generated part
    out = gen[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(out, skip_special_tokens=True).strip()

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _extract_choice_letter(text: str, labels: List[str]) -> Optional[int]:
    """
    Try to extract the chosen option index from generated text using letters/numbers.
    labels: e.g., ['A','B','C','D'] or ['1','2']
    Returns index or None.
    """
    t = text.strip()

    # 1) Strict single-letter at start or after "Answer:" etc.
    m = re.search(r"\b([A-Ha-h])\b", t)
    if m and any(m.group(1).upper() == L for L in labels):
        letter = m.group(1).upper()
        return labels.index(letter)

    # 2) Number form (for labels like ['1','2'] or numeric answers)
    m = re.search(r"\b([1-9])\b", t)
    if m and any(m.group(1) == L for L in labels):
        return labels.index(m.group(1))

    # 3) Variants like "Answer: A", "(A)", "Option B"
    m = re.search(r"(?:answer|option)\s*[:\-]?\s*([A-Ha-h1-9])", t, flags=re.IGNORECASE)
    if m:
        val = m.group(1)
        valU = val.upper()
        if valU in labels:
            return labels.index(valU)
        if val in labels:
            return labels.index(val)

    # 4) “The answer is A/B/1/2”
    m = re.search(r"the answer is\s*([A-Ha-h1-9])", t, flags=re.IGNORECASE)
    if m:
        val = m.group(1)
        valU = val.upper()
        if valU in labels:
            return labels.index(valU)
        if val in labels:
            return labels.index(val)

    return None

def _extract_choice_by_text(gen_text: str, choices: List[str]) -> Optional[int]:
    """
    Fallback: if model prints the answer text instead of a letter/number,
    match the option whose normalized text appears inside the generation.
    """
    g = _normalize(gen_text)
    # try exact contain first, then longest-match heuristic
    hits = []
    for i, c in enumerate(choices):
        cn = _normalize(c)
        if cn and cn in g:
            hits.append((len(cn), i))
    if hits:
        # choose the longest matched option to avoid spurious short overlaps
        hits.sort(reverse=True)
        return hits[0][1]
    return None

def _choose_from_generation(gen_text: str, labels: List[str], choices: List[str]) -> Optional[int]:
    # Try letter/number first
    idx = _extract_choice_letter(gen_text, labels)
    if idx is not None:
        return idx
    # Then text matching
    idx = _extract_choice_by_text(gen_text, choices)
    return idx

def _print_item(i, question, labels, choices, gen_text, pred_idx, gold_label):
    print(f"\nQ{i}: {question}")
    for lab, choice in zip(labels, choices):
        print(f"  {lab}) {choice}")
    print(f"  Model generated: {gen_text}")
    pred_disp = labels[pred_idx] if pred_idx is not None else "N/A"
    mark = "✓" if (pred_idx is not None and labels[pred_idx] == gold_label) else "✗"
    print(f" → Predicted: {pred_disp} | Correct: {gold_label} | {mark}")

# ---------------------------
# OpenBookQA (validation, "main" config)
# ---------------------------
@torch.no_grad()
def eval_openbookqa_gen(model, tokenizer, device=None, size=None, max_new_tokens=40, do_sample=False):
    ds = load_dataset("openbookqa", "main", split="validation")
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))

    correct = 0
    total = 0

    for i, ex in enumerate(ds, start=1):
        stem = ex["question_stem"].strip()
        choices = ex["choices"]["text"]
        labels = [L.upper() for L in ex["choices"]["label"]]  # ['A','B','C','D']
        gold_label = ex["answerKey"].strip().upper()

        opts_str = "\n".join(f"{L}) {c}" for L, c in zip(labels, choices))
        prompt = (
            "Answer the science question by replying with only the letter.\n\n"
            f"Question: {stem}\n{opts_str}\nAnswer:"
        )

        gen_text = _generate_text(model, tokenizer, prompt, device, max_new_tokens, do_sample=do_sample)
        pred_idx = _choose_from_generation(gen_text, labels, choices)

        is_correct = (pred_idx is not None and labels[pred_idx] == gold_label)
        correct += int(is_correct)
        total += 1

        # _print_item(i, stem, labels, choices, gen_text, pred_idx, gold_label)

    acc = correct / total if total else 0.0
    print(f"\nOpenBookQA (generation) accuracy: {acc:.4f}  ({correct}/{total})")
    return acc


def get_attack_acc_openbookqa(model, tokenizer, saved_state_dict, gradients, weight_indices):
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

    acc = eval_openbookqa_gen(model, tokenizer, model.device, size = 150)
    print(name, 'Acc:', acc,'number of indices:', count)
                
    custom_load_state_dict(model, saved_state_dict)
    clear_memory()

    return acc