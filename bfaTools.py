from typing import Tuple

import torch
from transformers import MambaForCausalLM  # noqa: E402


def flip_bits_in_tensor_float16(tensor: torch.Tensor) -> torch.Tensor:
    """
    Flips the sign bit of each element in a given float16 tensor.

    Args:
        tensor (torch.Tensor): A tensor of type float16.

    Returns:
        torch.Tensor: A new tensor with the sign bit of each element flipped.
    """
    original_type = tensor.dtype
    bit_mask = 1 << 15
    flipped_tensor = tensor.view(torch.int16).bitwise_xor(bit_mask)
    return flipped_tensor.view(original_type)


def flip_bits_in_tensor_float32(tensor: torch.Tensor) -> torch.Tensor:
    """
    Flips the sign bit of each element in a given float32 tensor.

    Args:
        tensor (torch.Tensor): A tensor of type float32.

    Returns:
        torch.Tensor: A new tensor with the sign bit of each element flipped.
    """
    original_type = tensor.dtype
    bit_mask = 1 << 31
    flipped_tensor = tensor.view(torch.int32).bitwise_xor(bit_mask)
    return flipped_tensor.view(original_type)


def flip_bits_in_tensor(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Flips the bits in the given tensor based on its data type.

    This function processes a tensor and flips its bits depending on whether
    its data type is `torch.float16` or `torch.float32`. If the data type is
    unsupported, it raises a `ValueError`.

    Args:
        tensor (torch.Tensor): The input tensor whose bits are to be flipped.
        model (PreTrainedModel): The model instance, though its usage is not
            defined in the current implementation.

    Returns:
        torch.Tensor: A tensor with its bits flipped.

    Raises:
        ValueError: If the tensor's data type is not `torch.float16` or `torch.float32`.

    Note:
        This function delegates the actual bit-flipping logic to
        `flip_bits_in_tensor_float16` or `flip_bits_in_tensor_float32`
        based on the tensor's data type.
    """
    data_type = tensor.dtype
    if data_type == torch.float16:
        return flip_bits_in_tensor_float16(tensor)
    elif data_type == torch.float32:
        return flip_bits_in_tensor_float32(tensor)
    elif data_type == torch.bfloat16:
        return flip_bits_in_tensor_float16(tensor)
    else:
        raise ValueError(
            f"Unsupported data type: {data_type}. Only float16, float32 and bfloat16 are supported."
        )


def flip_custom_bit_in_tensor(
    target_idxs: torch.Tensor, w: torch.Tensor
) -> Tuple[torch.Tensor, int]:
    original_data_type = w.dtype
    view_type: torch.dtype = None
    if original_data_type == torch.float16 or original_data_type == torch.bfloat16:
        view_type = torch.int16
    elif original_data_type == torch.float32:
        view_type = torch.int32
    else:
        raise ValueError(
            f"Unsupported data type: {original_data_type}. Only float16, float32 and bfloat16 are supported."
        )
    bfas_num = 0

    w = w.view(view_type)
    for weight_index, bit_index in target_idxs:
        target_weight = w[weight_index]

        bit_mask = 1 << bit_index
        target_weight = target_weight.bitwise_xor(bit_mask)
        if torch.isfinite(target_weight.to(original_data_type)).all().item():
            w[weight_index] = target_weight
            bfas_num += 1

    return w.to(original_data_type), bfas_num


def custom_load_state_dict_single_layer(
    model: MambaForCausalLM, state_dict, layer_name: str
):
    """
    Loads the state dictionary for a single layer of the model.

    Args:
        model (MambaForCausalLM): The model instance to load the state dictionary into.
        state_dict (dict): A dictionary containing the state of the model.
        layer_name (str): The name of the layer to load the state dictionary for.

    Returns:
        None
    """
    model_state_dict = model.state_dict()
    model_state_dict[layer_name].copy_(state_dict[layer_name])
