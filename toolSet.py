import gc
import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset as hf_load_dataset
from rich.progress import TaskID
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

MAMBA_130M_SIZE = 129_135_360
MAMBA_LAYER_TYPES_SIZE = 8
MAMBA_2_8B_SIZE = 2_768_345_600
MAMBA_BLOCK_NUM = (24, 48, 48, 48, 64)


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()


PreTrainedTokenizerType = PreTrainedTokenizer or PreTrainedTokenizerFast

access_token = "" # replace with HF access token here


def load_dataset():
    return hf_load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="test", token=access_token
    )


class AttackType(Enum):
    MSEB = auto()
    SB = auto()
    BOTH = auto()


@dataclass
class SelectionStrategy(ABC):
    name: str
    attack_rate_list: List[float]
    params: Dict
    attack_type: AttackType = AttackType.SB
    _counter: int = 0
    _current_attack_rate: float = None

    @abstractmethod
    def _select_weights(
        self, w: torch.Tensor, g: torch.Tensor, topk: int
    ) -> torch.Tensor:
        pass

    def gauge_attack_rate(
        self,
    ):
        while self._counter < len(self.attack_rate_list):
            self._current_attack_rate = self.attack_rate_list[self._counter]
            yield self._current_attack_rate
            self._counter += 1
        self._counter = 0

    def get_weights_to_target(
        self,
        model: PreTrainedModel,
        gradients: Dict[str, torch.Tensor],
        layer_name: str,
        topk: int = None,
    ) -> Tuple[int, torch.Tensor]:
        """
        Computes the indices of weights in a specified layer of a model that are
        selected based on their gradients and returns the count and the indices.

        Args:
            model (PreTrainedModel): The pre-trained model containing the weigh
            gradients (Dict[str, torch.Tensor]): A dictionary mapping layer names
                to their corresponding gradient tensors.
            layer_name (str): The name of the layer whose weights are to be targeted.

        Returns:
            Tuple[int, torch.Tensor]: A tuple containing:
                - The number of selected weights (int).
                - A tensor of indices corresponding to the selected weights (torch.Tensor).
        """

        orig_data: torch.Tensor = model.state_dict()[layer_name].data
        w: torch.Tensor = orig_data.reshape(-1)
        g: torch.Tensor = gradients[layer_name].float().detach().reshape(-1)
        target_weights_idx: torch.Tensor = self._select_weights(w, g, topk)

        return target_weights_idx.shape[0], target_weights_idx


@dataclass
class LayerSensitivityParams:
    layer_name: str
    attack_rate: float
    number_of_targeted_weights: int
    post_attack_perplexity: float
    post_attack_loss: float
    original_perplexity: float
    original_loss: float
    post_attack_accuracy: float = None
    original_accuracy: float = None
    layer_type: str = None

    def post_init__(self):
        if "input_layernorm.weight" in self.layer_name:
            self.layer_type = "norm.weight"
        elif "mamba" in self.layer_name:
            dirty_layer_type = self.layer_name.split(".mamba.")[1]
            if ".0" in dirty_layer_type:
                self.layer_type = dirty_layer_type.replace(".0", "")
            elif "proj_" in dirty_layer_type:
                self.layer_type = dirty_layer_type.replace("proj_", "proj.")
            else:
                self.layer_type = dirty_layer_type
        elif "mixer" in self.layer_name:
            self.layer_type = self.layer_name.split("mixer.")[1]
        else:
            self.layer_type = ".".join(self.layer_name.split(".")[-2:])

    def uuid(self) -> str:
        hashtext = f"{self.layer_name},{self.attack_rate}"
        hash_obj = hashlib.md5(hashtext.encode("utf-8"))
        # Convert the hash to UUID (version 3 or 5)
        # Using version 3 (MD5-based) UUID
        return str(uuid.UUID(hex=hash_obj.hexdigest()))


@dataclass
class BlockRangeAttackResults:
    starting_index: int
    targeting_range: int
    ending_range: int
    attack_rate: int
    targeted_weights: int
    post_attack_perplexity: float
    post_attack_loss: float
    original_perplexity: float
    original_loss: float
    post_attack_accuracy: float = None
    original_accuracy: float = None

    def uuid(self) -> str:
        hashtext = f"{self.starting_index},{self.targeting_range},{self.attack_rate}"
        hash_obj = hashlib.md5(hashtext.encode("utf-8"))
        # Convert the hash to UUID (version 3 or 5)
        # Using version 3 (MD5-based) UUID
        return str(uuid.UUID(hex=hash_obj.hexdigest()))


def get_layer_type(layer_name: str) -> str:
    if "input_layernorm.weight" in layer_name:
        layer_type = "norm.weight"
    elif "mamba" in layer_name:
        dirty_layer_type = layer_name.split(".mamba.")[1]
        if ".0" in dirty_layer_type:
            layer_type = dirty_layer_type.replace(".0", "")
        elif "proj_" in dirty_layer_type:
            layer_type = dirty_layer_type.replace("proj_", "proj.")
        else:
            layer_type = dirty_layer_type
    elif "mixer" in layer_name:
        layer_type = layer_name.split("mixer.")[1]
    elif ".norm.weight" in layer_name:
        layer_type = ".".join(layer_name.split(".")[-2:])
    else:
        raise Exception("unknown layer type")

    return layer_type


def get_layer_index(layer_name: str) -> int:
    """
    Extracts the layer index from the layer name.

    Args:
        layer_name (str): The name of the layer.

    Returns:
        int: The index of the layer.
    """
    if "input_layernorm.weight" in layer_name:
        return int(layer_name.split(".input_layernorm")[0].split(".")[-1])
    elif "mamba" in layer_name:
        return int(layer_name.split(".mamba")[0].split(".")[-1])
    elif ".mixer" in layer_name:
        return int(layer_name.split(".mixer")[0].split(".")[-1])
    elif ".norm." in layer_name:
        return int(layer_name.split(".norm.")[0].split(".")[-1])
    else:
        raise Exception("unknown layer type")


def load_model(
    model_name: str,
    tokenizer_name: str,
    device: torch.device,
    dtype: torch.dtype = None,
    token: str = access_token,
) -> tuple[PreTrainedModel, PreTrainedTokenizer or PreTrainedTokenizerFast]:
    """
    # Configure BitsAndBytes for int8 quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Set for 8-bit loading
        # bnb_8bit_quant_type=mode,  # Keep 'int8' mode (change to your preferred quantization method if any)
        # bnb_8bit_compute_dtype=torch.float16,  # Use float16 for computation, can change if needed
        # bnb_8bit_use_double_quant=True,  # Double quantization may improve performance
    )
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,  # Automatically map model layers to devices
        # quantization_config=bnb_config,
        torch_dtype=dtype
        if dtype is not None
        else "auto",  # float16 is standard for compute with quantization
        trust_remote_code=True,
        token=token,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True, use_fast=True, token=token
    )
    tokenizer.padding_side = "left"

    return model, tokenizer


def calculate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerType,
    dataset,
    size: int,
) -> float:
    """
    Calculate the perplexity of a given language model on a specified dataset.
    Args:
        model (PreTrainedModel): The language model to evaluate.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer associated with the model.
        dataset: The dataset to evaluate the model on.
        size (int): The number of samples from the dataset to use for evaluation.
    Returns:
        float: The calculated perplexity of the model on the dataset.
    """

    perplexity, _, _ = calculate_perplexity_loss_and_length(
        model, tokenizer, dataset, size, model.device
    )

    return perplexity


def calculate_perplexity_loss_and_length(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerType,
    dataset,
    size: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Calculate the perplexity, total loss, and total length of a given dataset using a language model.
    Args:
        model (PreTrainedModel): The language model to evaluate.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer to preprocess the input text.
        dataset (Dataset): The dataset containing text examples.
        size (int): The number of examples to evaluate from the dataset.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
    Returns:
        Tuple[float, float, float]: A tuple containing the perplexity, total loss, and total length of the evaluated examples.
    """
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

    perplexity = torch.exp(torch.tensor(total_loss / total_length))

    return perplexity.item(), total_loss / total_length, total_length


# bravo-6 genetics kinda start here


def read_layer_sensitivity_from_file(
    file_name: str,
) -> Dict[str, LayerSensitivityParams]:
    """
    Reads layer sensitivity parameters from a CSV file and returns them as a dictionary.

    Args:
        file_name (str): The path to the CSV file containing the layer sensitivity parameters.

    Returns:
        Dict[str, LayerSensitivityParams]: A dictionary where the keys are layer names and the values are
                                           LayerSensitivityParams objects containing the sensitivity parameters
                                           for each layer.

    The CSV file is expected to have the following columns:
        - layer_name: The name of the layer.
        - alpha: The alpha value for the layer.
        - attack_rate: The attack ratio for the layer.
        - number_of_targeted_weights: The number of targeted weights for the layer.
        - post_attack_perplexity: The perplexity of the layer after the attack.
        - original_perplexity: The original perplexity of the layer before the attack.
    """
    df = pd.read_csv(file_name)
    sensitivity_result = {}
    for _, row in df.iterrows():
        layer_data = LayerSensitivityParams(
            layer_name=row["layer_name"],
            alpha=row["alpha"],
            attack_rate=row["attack_rate"],
            number_of_targeted_weights=row["number_of_targeted_weights"],
            post_attack_perplexity=row["post_attack_perplexity"],
            post_attack_loss=row["post_attack_loss"],
            original_perplexity=row["original_perplexity"],
            original_loss=row["original_loss"],
            arc_e_accuracy=row["arc_e_accuracy"] if "arc_e_accuracy" in row else None,
        )
        sensitivity_result[layer_data.uuid()] = layer_data
    return sensitivity_result


def file_name(date: str, strat: str, model_name: str) -> str:
    return f"./results/{date}/{strat}_{model_name}_sensitivity.csv"


custom_types = {
    "layer_id": "string",
    "layer_type": "string",
    "layer_name": "string",
    "attack_rate": "float32",
    "number_of_targeted_weights": "int32",
    "post_attack_perplexity": "float32",
    "post_attack_loss": "float32",
    "original_perplexity": "float32",
    "original_loss": "float32",
    "arc_e_accuracy": "int8",
}


def get_df(
    fitness=0, date="2025-05-17_05-44", current_strat="mag", current_model="370"
) -> Tuple[Dict[float, pd.DataFrame], pd.DataFrame]:
    df = pd.read_csv(
        file_name(
            date,
            current_strat,
            current_model,
        ),
        dtype=custom_types,
    )
    df["block_index"] = df["layer_name"].apply(
        lambda layer_name: get_layer_index(layer_name)
    )
    df["layer_type"] = df["layer_name"].apply(
        lambda layer_name: get_layer_type(layer_name)
    )

    corrections = {
        0.004999999888241291: 0.005,
        0.009999999776482582: 0.001,
        0.0010000000474974513: 0.001,
        0.05000000074505806: 0.05,
        0.10000000149011612: 0.1,
    }
    for key, value in corrections.items():
        df.loc[df["attack_rate"] == key, "attack_rate"] = value

    df.loc[df["post_attack_perplexity"] == np.inf, "post_attack_perplexity"] = (
        df.loc[df["post_attack_perplexity"] != np.inf, "post_attack_perplexity"].max()
        * 1.01
    )

    unique_rates: List = df["attack_rate"].unique().tolist()
    unique_rates.sort()
    rate_grouped: Dict[float, pd.DataFrame] = {}

    original_loss = df.loc[0, "original_loss"]
    original_perplexity = df.loc[0, "original_perplexity"]

    df.loc[
        (df["post_attack_perplexity"] == 0.0) | (df["post_attack_loss"] == 0.0),
        "post_attack_perplexity",
    ] = original_perplexity
    df.loc[
        (df["post_attack_perplexity"] == 0.0) | (df["post_attack_loss"] == 0.0),
        "post_attack_loss",
    ] = original_loss

    df["post_attack_loss_log_rate"] = df["post_attack_loss"] / (
        original_loss * (fitness * df["number_of_targeted_weights"] + 1)
    )
    df.loc[df["post_attack_loss_log_rate"] == np.inf, "post_attack_loss_log_rate"] = (
        df.loc[
            df["post_attack_loss_log_rate"] != np.inf,
            "post_attack_loss_log_rate",
        ].max()
        * 1.01
    )
    df["post_attack_perplexity_log_rate"] = df["post_attack_perplexity"] / (
        original_perplexity * (fitness * df["number_of_targeted_weights"] + 1)
    )
    df.loc[
        df["post_attack_perplexity_log_rate"] == np.inf,
        "post_attack_perplexity_log_rate",
    ] = (
        df.loc[
            df["post_attack_perplexity_log_rate"] != np.inf,
            "post_attack_perplexity_log_rate",
        ].max()
        * 1.01
    )

    for rate in unique_rates:
        rate_grouped[rate] = df[(df["attack_rate"] == rate)]
        rate_grouped[rate] = rate_grouped[rate].reset_index(drop=True)

    return rate_grouped, df


class NullProgress:
    def __init__(self, *args, **kwargs):
        pass

    def add_task(self, *args, **kwargs) -> TaskID:
        return 0  # Dummy task ID

    def update(self, task_id, **kwargs):
        pass

    def remove_task(self, task_id):
        pass

    def advance(self, task_id, advance=1):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
