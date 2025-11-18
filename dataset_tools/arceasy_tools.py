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

def _safe_max_len(tokenizer, config, default=2048):
    # Prefer tokenizer hint if it's not extremely large (some tokenizers use 1e30 as "no limit")
    tml = getattr(tokenizer, "model_max_length", None)
    if isinstance(tml, int) and 0 < tml < 10_000_000:
        return tml
    # Try common config field names
    for k in ["max_seq_len", "max_sequence_length", "seq_len", "n_positions", "max_position_embeddings"]:
        v = getattr(config, k, None)
        if isinstance(v, int) and v > 0:
            return v
    return default

def eval_arc_easy(model, tokenizer, device="cuda", size = None, normalize = None):
    """
    Evaluate model accuracy on ARC-Easy validation set.
    Prints each prompt, predicted answer, and gold answer.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_len = _safe_max_len(tokenizer, model.config, default=2048)

    def option_logprob(context, option):
        # Tokenize separately to know boundary
        ctx_ids = tokenizer.encode(context, add_special_tokens=False)
        opt_ids = tokenizer.encode(" " + option, add_special_tokens=False)  # leading space helps many tokenizers

        # Truncate from the left to fit model window
        total = len(ctx_ids) + len(opt_ids)
        if total > max_len:
            # keep at least 1 token of context
            cut = total - max_len
            cut = min(cut, max(0, len(ctx_ids) - 1))
            ctx_ids = ctx_ids[cut:]

        full_ids = ctx_ids + opt_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)

        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[:, :-1, :]            # positions predicting next tokens
        targets = input_ids[:, 1:]                # next tokens
        logps = torch.log_softmax(logits, dim=-1)
        token_logp = logps.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, L-1)

        # Option token region in token_logp is [len(ctx_ids)-1 : len(full_ids)-1]
        start = max(len(ctx_ids) - 1, 0)
        end = len(full_ids) - 1
        if end <= start:
            return float("-inf")  # empty/degenerate option after tokenization

        opt_lp = token_logp[0, start:end].sum()
        if normalize:
            opt_len = end - start
            if opt_len > 0:
                opt_lp = opt_lp / opt_len
        return float(opt_lp)

    ds = load_dataset("ai2_arc", "ARC-Easy", split="validation")
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))

    correct = 0
    total = 0

    for ex in ds:
        q = ex["question"]
        choices = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        gold_label = ex["answerKey"]
        gold_idx = labels.index(gold_label)

        context = f"Question: {q}\nAnswer:"
        scores = [option_logprob(context, c) for c in choices]
        pred_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        pred_label = labels[pred_idx]

        is_correct = (pred_label == gold_label)
        correct += is_correct
        total += 1

        prompt = f"{context}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gen = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,  # deterministic
            temperature=0.0
        )
        output = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  Model generated: {output.strip()}")

        print(f"\nQ{total}: {q}")
        for i, (lab, choice) in enumerate(zip(labels, choices)):
            print(f"  {lab}) {choice}")
        print(f" → Predicted: {pred_label} | Correct: {gold_label} | {'✓' if is_correct else '✗'}")

        if (size is not None) and (total >= size):
            break

    acc = correct / total if total else 0
    print(f"\nARC-Easy accuracy: {acc:.4f} ({correct}/{total})")
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
# ARC-Easy (ai2_arc / ARC-Easy)
# ---------------------------
@torch.no_grad()
def eval_arc_easy_gen(model, tokenizer, device=None, size=None, max_new_tokens=40, do_sample=False):
    ds = load_dataset("ai2_arc", "ARC-Easy", split="validation")
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))

    correct = 0
    total = 0

    for i, ex in enumerate(ds, start=1):
        q = ex["question"].strip()
        choices = ex["choices"]["text"]
        labels = [L.upper() for L in ex["choices"]["label"]]  # e.g., ['A','B','C','D']
        gold_label = ex["answerKey"].strip().upper()

        # Build a clear prompt
        opts_str = "\n".join(f"{L}) {c}" for L, c in zip(labels, choices))
        prompt = (
            f"Answer the multiple choice question by replying with only the letter.\n\n"
            f"Question: {q}\n{opts_str}\nAnswer:"
        )

        gen_text = _generate_text(model, tokenizer, prompt, device, max_new_tokens, do_sample=do_sample)
        pred_idx = _choose_from_generation(gen_text, labels, choices)

        is_correct = (pred_idx is not None and labels[pred_idx] == gold_label)
        correct += int(is_correct)
        total += 1

        # _print_item(i, q, labels, choices, gen_text, pred_idx, gold_label)

    acc = correct / total if total else 0.0
    print(f"\nARC-Easy (generation) accuracy: {acc:.4f}  ({correct}/{total})")
    return acc

def get_attack_acc_arceasy(model, tokenizer, saved_state_dict, gradients, weight_indices):
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

    acc = eval_arc_easy_gen(model, tokenizer, model.device, size = 150)
    print(name, 'Acc:', acc,'number of indices:', count)
                
    custom_load_state_dict(model, saved_state_dict)
    clear_memory()

    return acc
