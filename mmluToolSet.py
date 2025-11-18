import json
import os
import time
from typing import List, Tuple

import pandas as pd
import toolSet as ts
import torch
from transformers import BatchEncoding, PreTrainedModel

# MMLU Tasks
TASKS: List[str] = [
    "astronomy",
    "business_ethics",
    "high_school_us_history",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]
# MMLU Choices
choices = ["A", "B", "C", "D"]


# MMLU Accuracy computation
def compute_metric(run_results):
    # with open(output_filename, 'r') as f:
    #     run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            # print("pred: %s, gold: %s" % (pred, gold))
            if gold == pred.replace(' ', ''): acc += 1
        print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc/total_num))
    return total_acc/total_num


# Format subject of prompt
def format_subject(subject: str) -> str:
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


# Format prompt
def format_example(df: pd.DataFrame, idx: int, subject: str, include_answer=True) -> str:
    prompt = "The following are multiple choice questions (with answers) about {}. ANSWER SHOULD BE IN ANY ONE OF A, B, C OR D. DO NOT ANSWER ANYTHING ELSE. THE ANSWER SHOULD ONLY BE A LETTER AND NOT A NUMBER\n".format(format_subject(subject))
    prompt += df.iloc[idx, 0]
    k = len(df['choices'].iloc[idx])
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df['choices'].iloc[idx][j])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(choices[df['answer'].iloc[idx]])
    return prompt


# Generate the prompt for prediction
def gen_prompt(train_df: pd.DataFrame, subject: str, k=-1) -> str:
    prompt = "The following are multiple choice questions (with answers) about {}. ANSWER SHOULD BE IN ANY ONE OF A, B, C OR D. DO NOT ANSWER ANYTHING ELSE. THE ANSWER SHOULD ONLY BE A LETTER AND NOT A NUMBER\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i, subject)
    return prompt


# Prepare the tokens
def prepare_input(
    tokenizer: ts.PreTrainedTokenizerType,
    prompts: List[str],
    device: torch.device,
) -> BatchEncoding:
    tokenizer.padding_side = "right"                  # important for Hymba’s mask handling
    tokenizer.truncation_side = "left"                # or "right", pick what to drop
    tokenizer.pad_token = tokenizer.eos_token
    max_len = tokenizer.model_max_length          # safe fallback

    input_tokens = tokenizer(
        prompts[0],
        return_tensors="pt",
        padding=True,
        # truncation=True,
        max_length=256,
    )

    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(device=device)

    return input_tokens

    # input_tokens = tokenizer.batch_encode_plus(
    #     prompts, return_tensors="pt", padding=True, max_length=max_len
    # )
    # for t in input_tokens:
    #     if torch.is_tensor(input_tokens[t]):
    #         input_tokens[t] = input_tokens[t].to(device=device)

    # return input_tokens


# Split all prompts into batches
def batch_split(prompts: List[str], batch_num: int) -> List[List[str]]:
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts


# Perform inference on all batches
def batch_infer(
    model: PreTrainedModel,
    tokenizer: ts.PreTrainedTokenizerType,
    prompts: List[str],
) -> List[str]:
    batch_size = 1
    answers: List[str] = []
    batch_prompts = batch_split(prompts, batch_size)
    for batch_input in batch_prompts:
        # print(batch_input)
        encode_inputs = prepare_input(tokenizer, batch_input, model.device)
        # print(encode_inputs.dtype)
        outputs = model.generate(**encode_inputs, max_new_tokens=1, pad_token_id=128001)
        # print(outputs.loss)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        # print('answers: ', answers)
        # break
    answers = [answer[-1] for answer in answers]
    return answers


# Calculate loss with gradients
def batch_infer_bp_loss(
    model: PreTrainedModel,
    tokenizer: ts.PreTrainedTokenizerType,
    prompts: List[str],
    optimizer: torch.optim.Optimizer,
    batch_size=4,
):
    batch_size = 4
    batch_prompts = batch_split(prompts, batch_size)
    for batch_input in batch_prompts:
        encode_inputs = prepare_input(tokenizer, batch_input, model.device)
        # print(encode_inputs)
        # st = time.time()
        outputs = model(**encode_inputs, labels=encode_inputs['input_ids'])
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        break


# Calculate loss without gradients (zeroth order), and logits
def batch_infer_zo_loss(
    model: PreTrainedModel,
    tokenizer: ts.PreTrainedTokenizerType,
    prompts: List[str],
    optimizer: torch.optim.Optimizer,
    batch_size=4,
) -> Tuple[float, torch.Tensor]:
    accum_loss = 0
    i = 0
    batch_prompts = batch_split(prompts, batch_size)
    for batch_input in batch_prompts:
        encode_inputs = prepare_input(tokenizer, batch_input, model.device)
        # print(encode_inputs)
        with torch.no_grad():
            outputs = model(**encode_inputs, labels=encode_inputs["input_ids"])
            # outputs = model(**encode_inputs)
        loss = outputs.loss
        i = i + 1
        accum_loss = accum_loss + loss.item()
        break
    return loss.item(), outputs.logits


# Calculate MMLU accuracy
def mmlu_test(
    model: PreTrainedModel,
    tokenizer: ts.PreTrainedTokenizerType,
    file_name: str,
    TASKS: List[str],
) -> float:
    run_results = {}
    output_filename = "run_results_%s.json" % (file_name)
    start_time = time.time()
    for task in TASKS:
        print("Testing %s ..." % task)
        records = []
        # train = pd.read_csv("./arc_data/train.csv", header=None)[:10]
        # test_df = pd.read_csv("./arc_data/test.csv", header=None)
        # _ = pd.read_csv(
        #     os.path.join("mmlu_data/", "dev", task + "_dev.csv"), header=None
        # )[:5]
        # test_df = pd.read_csv(
        #     os.path.join("mmlu_data/", "test", task + "_test.csv"), header=None
        # )
        splits = {'test': task+'/test-00000-of-00001.parquet', 'validation': task+'/validation-00000-of-00001.parquet', 'dev': task+'/dev-00000-of-00001.parquet'}
        dev_df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["dev"])[:5]
        test_df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])
        # print(test_df)
        # break
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            # k = 5
            prompt_end = format_example(test_df, i, task, include_answer=False)
            # print(prompt_end)
            # train_prompt = gen_prompt(dev_df, task, k)
            prompt = prompt_end
            # print(prompt)
            while len(tokenizer.tokenize(prompt)) >= 2048:  # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = "\n\n".join(prompt_split)
            # print(prompt)
            correct_answer = test_df.iloc[i, test_df.shape[1] - 1]
            records.append({"prompt": prompt, "answer": correct_answer})

        pred_answers = batch_infer(
            model, tokenizer, [record["prompt"] for record in records]
        )

        gold_answers = [choices[record["answer"]] for record in records]
        run_results[task] = {"pred_answers": pred_answers, "gold_answers": gold_answers}
        # print(run_results)
    # with open(output_filename, "w") as f:
    #     json.dump(run_results, f, ensure_ascii=False, indent=2)

    accuracy = compute_metric(run_results)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
    return accuracy


# mmlu_test('')


# Calculate MMLU loss and logits
def mmlu_loss(
    model: PreTrainedModel,
    tokenizer: ts.PreTrainedTokenizerType,
    optimizer: torch.optim.Optimizer,
    file_name: str,
    TASKS: List[str],
    mode="zo",
    batch_size=4,
):
    loss, logits = None, None
    # run_results = {}
    # output_filename = "run_results_%s.json" % (file_name)
    # start_time = time.time()
    for task in TASKS:
        print("Testing %s ..." % task)
        records = []
        # dev_df = pd.read_csv(
        #     os.path.join("../attnbreaker/mmlu_data/", "dev", task + "_dev.csv"), header=None
        # )[:5]
        # test_df = pd.read_csv(
        #     os.path.join("../attnbreaker/mmlu_data/", "test", task + "_test.csv"), header=None
        # )
        splits = {'test': task+'/test-00000-of-00001.parquet', 'validation': task+'/validation-00000-of-00001.parquet', 'dev': task+'/dev-00000-of-00001.parquet'}
        dev_df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["dev"])[:5]
        test_df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])
        # print(test_df)
        # break
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = 5
            prompt_end = format_example(test_df, i, task, include_answer=False)
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = train_prompt + prompt_end
            while len(tokenizer.tokenize(prompt)) + 1 > 2048:  # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = "\n\n".join(prompt_split)
            label = test_df.iloc[i, test_df.shape[1] - 1]
            records.append({"prompt": prompt, "answer": label})

        if mode == "zo":
            loss, logits = batch_infer_zo_loss(
                model,
                tokenizer,
                [record["prompt"] for record in records],
                optimizer,
                batch_size,
            )
        elif mode == "bp":
            batch_infer_bp_loss(
                model,
                tokenizer,
                [record["prompt"] for record in records],
                optimizer,
                batch_size,
            )

        return loss, logits
