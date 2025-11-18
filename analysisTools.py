import copy
import time
from typing import Any, Callable, Dict, List

import bfaTools as bfa
import toolSet as ts
import torch
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from transformers import MambaForCausalLM, PreTrainedModel
from transformers.models.mamba.modeling_mamba import MambaBlock

PROGRESS_FLAG = False


def analyze_model_using_selection_strategy(
    model: PreTrainedModel,
    tokenizer: PreTrainedModel,
    gradients: Dict[str, torch.Tensor],
    dataset,
    perplexity_size: int,
    selection_strategy: ts.SelectionStrategy,
    device_name: str,
    targeting_lambda: Callable[[str, torch.nn.Parameter], bool],
    original_perplexity: float,
    original_loss: float,
    cuda_instances: int = 1,
    accuracy_func: Callable[
        [PreTrainedModel, ts.PreTrainedTokenizerType, str],
        List[Dict[str, Any]],
    ] = None,
) -> Dict[str, ts.LayerSensitivityParams]:
    cuda_index = int(device_name.split(":")[-1])

    layer_sensitivity_params_list: Dict[str, ts.LayerSensitivityParams] = {}
    layers_count = sum(1 for _ in model.parameters())
    print("cuda_index=>", cuda_index, " cuda_instances=>", cuda_instances)
    with Progress(
        TextColumn("[bold cyan][/]{task.description}"),  # Custom label before task name
        TextColumn("[bold green]Prog:[/]"),
        TaskProgressColumn(),  # Custom label before percentage
        TextColumn("[bold yellow]Done:[/]"),
        MofNCompleteColumn(),  # Custom label before n/total
        TextColumn("[bold magenta]Speed:[/]"),
        TransferSpeedColumn(),  # Custom label before updates/sec
        TextColumn("[bold red]Elapsed:[/]"),
        TimeElapsedColumn(),  # Custom label before elapsed time
        TextColumn("[bold blue]Remaining:[/]"),
        TimeRemainingColumn(),  # Custom label before remaining time
    ) as progress:
        layer_task = progress.add_task("[cyan]layer=", total=layers_count)
        iteration_index = 0
        for layer_name, param in model.named_parameters():
            progress.update(layer_task, advance=1)

            if not targeting_lambda(layer_name, param):
                continue

            iteration_index += 1
            if (iteration_index % cuda_instances) != (cuda_index % cuda_instances):
                continue

            model_state_dict = model.state_dict()

            attack_rate_task = progress.add_task(
                "[cyan]Attack Rate=", total=len(selection_strategy.attack_rate_list)
            )
            for attack_rate in selection_strategy.gauge_attack_rate():
                ts.clear_memory()

                progress.update(attack_rate_task, advance=1)
                number_of_targets, target_idxs = (
                    selection_strategy.get_weights_to_target(
                        model, gradients, layer_name
                    )
                )
                accuracy_func_result = 0.0
                post_attack_perplexity, post_attack_loss = (
                    original_perplexity,
                    original_loss,
                )

                # Bravo-6 todo attack layer and restore
                if number_of_targets > 0:
                    orig_data: torch.Tensor = (
                        model_state_dict[layer_name].data.detach().clone()
                    )
                    original_shape: torch.Size = orig_data.shape
                    w: torch.Tensor = model_state_dict[layer_name].data.reshape(-1)

                    w[target_idxs] = bfa.flip_bits_in_tensor(w[target_idxs])

                    model_state_dict[layer_name].data.copy_(w.reshape(original_shape))

                    post_attack_perplexity, post_attack_loss, _ = (
                        ts.calculate_perplexity_loss_and_length(
                            model,
                            tokenizer,
                            dataset,
                            size=perplexity_size,
                            device=model.device,
                        )
                    )
                    if accuracy_func is not None:
                        accuracy_func_result = accuracy_func(
                            model, tokenizer, device_name
                        )

                    model_state_dict[layer_name].data.copy_(orig_data)

                ########

                layer_sensitivity = ts.LayerSensitivityParams(
                    layer_name,
                    attack_rate,
                    number_of_targets,
                    post_attack_perplexity,
                    post_attack_loss,
                    original_perplexity,
                    original_loss,
                    accuracy_func_result,
                )
                layer_sensitivity_params_list[layer_sensitivity.uuid()] = (
                    layer_sensitivity
                )

                # human_readable_post_attack_perplexity=f"{post_attack_perplexity:.4f}"
                # post_attack_perplexity_list.append(human_readable_post_attack_perplexity)
                # number_of_targeted_weights_list.append(number_of_targeted_weights)
                # Restore to original
            progress.remove_task(attack_rate_task)
        progress.remove_task(layer_task)
    ts.clear_memory()

    return layer_sensitivity_params_list


def analyze_accuracy_on_targeted_layers(
    model: PreTrainedModel,
    tokenizer: PreTrainedModel,
    gradients: Dict[str, torch.Tensor],
    selection_strategy: ts.SelectionStrategy,
    original_accuracy: float,
    device_name: str,
    targeting_lambda: Callable[[str, torch.nn.Parameter], bool],
    cuda_instances: int = 1,
    accuracy_func: Callable[
        [PreTrainedModel, ts.PreTrainedTokenizerType, str],
        List[Dict[str, Any]],
    ] = None,
) -> Dict[str, ts.LayerSensitivityParams]:
    cuda_index = int(device_name.split(":")[-1])

    layer_sensitivity_params_list: Dict[str, ts.LayerSensitivityParams] = {}
    layers_count = sum(1 for _ in model.parameters())
    print("cuda_index=>", cuda_index, " cuda_instances=>", cuda_instances)
    with Progress(
        TextColumn("[bold cyan][/]{task.description}"),  # Custom label before task name
        TextColumn("[bold green]Prog:[/]"),
        TaskProgressColumn(),  # Custom label before percentage
        TextColumn("[bold yellow]Done:[/]"),
        MofNCompleteColumn(),  # Custom label before n/total
        TextColumn("[bold magenta]Speed:[/]"),
        TransferSpeedColumn(),  # Custom label before updates/sec
        TextColumn("[bold red]Elapsed:[/]"),
        TimeElapsedColumn(),  # Custom label before elapsed time
        TextColumn("[bold blue]Remaining:[/]"),
        TimeRemainingColumn(),  # Custom label before remaining time
    ) as progress:
        layer_task = progress.add_task("[cyan]layer=", total=layers_count)
        iteration_index = 0
        for layer_name, param in model.named_parameters():
            progress.update(layer_task, advance=1)

            if not targeting_lambda(layer_name, param):
                continue

            iteration_index += 1
            if (iteration_index % cuda_instances) != (cuda_index % cuda_instances):
                continue

            model_state_dict = model.state_dict()

            attack_rate_task = progress.add_task(
                "[cyan]Attack Rate=", total=len(selection_strategy.attack_rate_list)
            )
            for attack_rate in selection_strategy.gauge_attack_rate():
                ts.clear_memory()

                progress.update(attack_rate_task, advance=1)
                number_of_targets, target_idxs = (
                    selection_strategy.get_weights_to_target(
                        model, gradients, layer_name
                    )
                )
                post_attack_accuracy = original_accuracy

                # Bravo-6 todo attack layer and restore
                if number_of_targets > 0:
                    orig_data: torch.Tensor = (
                        model_state_dict[layer_name].data.detach().clone()
                    )
                    original_shape: torch.Size = orig_data.shape
                    w: torch.Tensor = model_state_dict[layer_name].data.reshape(-1)

                    w[target_idxs] = bfa.flip_bits_in_tensor(w[target_idxs])

                    model_state_dict[layer_name].data.copy_(w.reshape(original_shape))

                    post_attack_accuracy = accuracy_func(model, tokenizer, device_name)[
                        0
                    ]["acc"]

                    model_state_dict[layer_name].data.copy_(orig_data)

                ########

                layer_sensitivity = ts.LayerSensitivityParams(
                    layer_name,
                    attack_rate,
                    number_of_targets,
                    0,
                    0,
                    0,
                    0,
                    post_attack_accuracy,
                    original_accuracy,
                )
                layer_sensitivity_params_list[layer_sensitivity.uuid()] = (
                    layer_sensitivity
                )

            progress.remove_task(attack_rate_task)
        progress.remove_task(layer_task)
    ts.clear_memory()

    return layer_sensitivity_params_list


def get_param_name(model: MambaForCausalLM, param: torch.nn.Parameter) -> str:
    for name, p in model.named_parameters():
        if p is param:
            return name
    raise ValueError(f"Parameter {param} not found in model {model.name_or_path}")


def attack_block(
    model: MambaForCausalLM,
    block: MambaBlock,
    gradients: Dict[str, torch.Tensor],
    selection_strategy: ts.SelectionStrategy,
    target_parameter_types: List[str],
    effective_attack_rate: float = None,
) -> int:
    total_number_of_targets = 0
    for parameter in block.parameters():
        parameter_name = get_param_name(model, parameter)
        if ts.get_layer_type(parameter_name) not in target_parameter_types:
            continue

        number_of_targets, target_idxs = selection_strategy.get_weights_to_target(
            model, gradients, parameter_name, topk=effective_attack_rate
        )
        if number_of_targets == 0:
            continue
        total_number_of_targets += number_of_targets
        original_shape: torch.Size = model.state_dict()[parameter_name].data.shape
        w: torch.Tensor = model.state_dict()[parameter_name].data.reshape(-1)
        if len(target_idxs.shape) == 1:
            w[target_idxs] = bfa.flip_bits_in_tensor(w[target_idxs])
        else:
            w, _ = bfa.flip_custom_bit_in_tensor(target_idxs, w)
            if not torch.isfinite(w).all().item():
                raise ValueError(
                    f"Non-finite values encountered after flipping bits in parameter {parameter_name}."
                )

        model.state_dict()[parameter_name].data.copy_(w.reshape(original_shape))
    return total_number_of_targets


def restore_block(
    model: MambaForCausalLM,
    targeted_block: MambaBlock,
    intact_block: MambaBlock,
    target_param_types: List[str],
):
    for targeted_parameter, intact_parameter in zip(
        targeted_block.parameters(), intact_block.parameters()
    ):
        parameter_name = get_param_name(model, targeted_parameter)
        if ts.get_layer_type(parameter_name) not in target_param_types:
            continue
        model.state_dict()[parameter_name].data.copy_(intact_parameter.data)


def hook_fn(module, input, output):
    # Store the output of the module

    assert isinstance(output, torch.Tensor), "Output is not a tensor"
    assert torch.isfinite(input[0]).all().item(), (
        f"input contains non-finite values {torch.isfinite(input[0]).all()}"
    )
    assert not torch.isnan(output).any().item(), (
        f"Output contains non-finite values {torch.isfinite(output).all()}"
    )
    # You can store outputs in a dictionary or list
    # hook_outputs[module] = output


def analyze_accuracy_on_the_entire_layer(
    model: MambaForCausalLM,
    tokenizer: PreTrainedModel,
    gradients: Dict[str, torch.Tensor],
    selection_strategy: ts.SelectionStrategy,
    target_param_types: List[str],
    original_accuracy: float,
    device_name: str,
    targeting_window_start_idx: List[int],
    targeting_window_range: int,
    cuda_instances: int = 1,
    accuracy_func: Callable[
        [PreTrainedModel, ts.PreTrainedTokenizerType, str],
        List[Dict[str, Any]],
    ] = None,
) -> Dict[str, ts.BlockRangeAttackResults]:
    dataset = ts.load_dataset()
    perplexity_size: int = 10
    original_perplexity, original_loss, _ = ts.calculate_perplexity_loss_and_length(
        model,
        tokenizer,
        dataset,
        size=perplexity_size,
        device=model.device,
    )
    cuda_index = int(device_name.split(":")[-1])
    param_multiplier = sum(p.numel() for p in model.parameters()) / ts.MAMBA_130M_SIZE
    max_multiplier = (
        param_multiplier * targeting_window_range * ts.MAMBA_LAYER_TYPES_SIZE
    )
    attack_results_list: Dict[str, ts.BlockRangeAttackResults] = {}
    num_blocks = len(model.backbone.layers)
    print("cuda_index=>", cuda_index, " cuda_instances=>", cuda_instances)
    Progress_cls = Progress if PROGRESS_FLAG else ts.NullProgress
    with Progress_cls(
        TextColumn("[bold cyan][/]{task.description}"),  # Custom label before task name
        TextColumn("[bold green]Prog:[/]"),
        TaskProgressColumn(),  # Custom label before percentage
        TextColumn("[bold yellow]Done:[/]"),
        MofNCompleteColumn(),  # Custom label before n/total
        TextColumn("[bold magenta]Speed:[/]"),
        TransferSpeedColumn(),  # Custom label before updates/sec
        TextColumn("[bold red]Elapsed:[/]"),
        TimeElapsedColumn(),  # Custom label before elapsed time
        TextColumn("[bold blue]Remaining:[/]"),
        TimeRemainingColumn(),  # Custom label before remaining time
    ) as progress:
        block_task = progress.add_task("[cyan]block=", total=num_blocks)

        iteration_index = 0
        starting_time = time.time()
        for block_index in range(num_blocks):
            progress.update(block_task, advance=1)

            if block_index not in targeting_window_start_idx:
                continue

            targeting_task = progress.add_task(
                "[cyan]Targeting=", total=targeting_window_range
            )
            for targeting_offset in range(targeting_window_range):
                progress.update(targeting_task, advance=1)
                max_targeting_index = block_index + targeting_offset
                if max_targeting_index >= len(model.backbone.layers):
                    break

                attack_rate_multiplier = max_multiplier / (
                    (targeting_offset + 1) * len(target_param_types)
                )
                intact_block_list: List[MambaBlock] = []

                # Bravo-6 0 - Get the intact blocks
                for targeting_index in range(block_index, max_targeting_index + 1):
                    target_block: MambaBlock = model.backbone.layers[targeting_index]
                    intact_block_list.append(copy.deepcopy(target_block))

                attack_rate_task = progress.add_task(
                    "[cyan]Attack Rate=", total=len(selection_strategy.attack_rate_list)
                )
                for attack_rate in selection_strategy.gauge_attack_rate():
                    progress.update(attack_rate_task, advance=1)
                    ts.clear_memory()
                    iteration_index += 1
                    if (iteration_index % cuda_instances) != (
                        cuda_index % cuda_instances
                    ):
                        continue
                    elapsed_time = time.time() - starting_time
                    remaining_time = (
                        (1728 - iteration_index) * elapsed_time / iteration_index
                    )
                    print(
                        f"Iteration {iteration_index} of 1728, Elapsed: {(elapsed_time) / 60:.1f}m, Remaining: {(remaining_time) / 60:.1f}m"
                    )
                    effective_attack_rate = round(attack_rate * attack_rate_multiplier)
                    # Bravo-6 1 target the blocks here
                    targeted_weights = 0
                    for targeting_index in range(block_index, max_targeting_index + 1):
                        target_block: MambaBlock = model.backbone.layers[
                            targeting_index
                        ]
                        # hook_handle = target_block.mixer.in_proj.register_forward_hook(
                        #     hook_fn
                        # )

                        targeted_weights += attack_block(
                            model,
                            target_block,
                            gradients,
                            selection_strategy,
                            target_param_types,
                            effective_attack_rate,
                        )
                        ###### Bravo-6 2 get post attack accuracy
                    post_attack_accuracy = accuracy_func(model, tokenizer, device_name)[
                        0
                    ]["acc"]
                    # hook_handle.remove()
                    if False and (
                        post_attack_accuracy * 100 < 23.5
                        and post_attack_accuracy * 100 > 23.0625
                    ):
                        print(
                            f"Post attack accuracy is too low: {post_attack_accuracy * 100:.2f}%"
                        )
                        print(block_index, max_targeting_index)
                        exit(0)
                    post_attack_perplexity, post_attack_loss, _ = (
                        ts.calculate_perplexity_loss_and_length(
                            model,
                            tokenizer,
                            dataset,
                            size=perplexity_size,
                            device=model.device,
                        )
                    )

                    ###### Bravo-6 3 save the attack results
                    attack_results = ts.BlockRangeAttackResults(
                        block_index + 1,
                        targeting_offset + 1,
                        max_targeting_index + 1,
                        attack_rate,
                        targeted_weights,
                        post_attack_perplexity,
                        post_attack_loss,
                        original_perplexity,
                        original_loss,
                        post_attack_accuracy,
                        original_accuracy,
                    )
                    attack_results_list[attack_results.uuid()] = attack_results

                    ###### Bravo-6 4 Restore the block
                    for targeting_index in range(block_index, max_targeting_index + 1):
                        targeted_block: MambaBlock = model.backbone.layers[
                            targeting_index
                        ]
                        restore_block(
                            model,
                            targeted_block,
                            intact_block_list[targeting_index - block_index],
                            target_param_types,
                        )

                progress.remove_task(attack_rate_task)
            progress.remove_task(targeting_task)
        progress.remove_task(block_task)
    ts.clear_memory()

    return attack_results_list


def analyze_accuracy_on_the_entire_layer_reversed(
    model: MambaForCausalLM,
    tokenizer: PreTrainedModel,
    gradients: Dict[str, torch.Tensor],
    selection_strategy: ts.SelectionStrategy,
    target_param_types: List[str],
    original_accuracy: float,
    device_name: str,
    targeting_window_start_idx: List[int],
    targeting_window_range: int,
    cuda_instances: int = 1,
    accuracy_func: Callable[
        [PreTrainedModel, ts.PreTrainedTokenizerType, str],
        List[Dict[str, Any]],
    ] = None,
) -> Dict[str, ts.BlockRangeAttackResults]:
    dataset = ts.load_dataset()
    perplexity_size: int = 10
    original_perplexity, original_loss, _ = ts.calculate_perplexity_loss_and_length(
        model,
        tokenizer,
        dataset,
        size=perplexity_size,
        device=model.device,
    )
    cuda_index = int(device_name.split(":")[-1])
    param_multiplier = sum(p.numel() for p in model.parameters()) / ts.MAMBA_130M_SIZE
    max_multiplier = (
        param_multiplier * targeting_window_range * ts.MAMBA_LAYER_TYPES_SIZE
    )
    attack_results_list: Dict[str, ts.BlockRangeAttackResults] = {}
    num_blocks = len(model.backbone.layers)
    print("cuda_index=>", cuda_index, " cuda_instances=>", cuda_instances)
    targeting_window_start_idx = [
        num_blocks - i - 1 for i in targeting_window_start_idx
    ]
    iteration_index = 0
    with Progress(
        TextColumn("[bold cyan][/]{task.description}"),  # Custom label before task name
        TextColumn("[bold green]Prog:[/]"),
        TaskProgressColumn(),  # Custom label before percentage
        TextColumn("[bold yellow]Done:[/]"),
        MofNCompleteColumn(),  # Custom label before n/total
        TextColumn("[bold magenta]Speed:[/]"),
        TransferSpeedColumn(),  # Custom label before updates/sec
        TextColumn("[bold red]Elapsed:[/]"),
        TimeElapsedColumn(),  # Custom label before elapsed time
        TextColumn("[bold blue]Remaining:[/]"),
        TimeRemainingColumn(),  # Custom label before remaining time
    ) as progress:
        block_task = progress.add_task("[cyan]block=", total=num_blocks)

        for block_index in range(num_blocks - 1, -1, -1):
            progress.update(block_task, advance=1)

            if block_index not in targeting_window_start_idx:
                continue

            targeting_task = progress.add_task(
                "[cyan]Targeting=", total=targeting_window_range
            )
            for targeting_offset in range(targeting_window_range):
                progress.update(targeting_task, advance=1)
                min_targeting_index = block_index - targeting_offset
                if min_targeting_index < 0:
                    break

                attack_rate_multiplier = max_multiplier / (
                    (targeting_offset + 1) * len(target_param_types)
                )
                intact_block_dict: Dict[int, MambaBlock] = {}

                # Bravo-6 0 - Get the intact blocks
                for targeting_index in range(block_index, min_targeting_index - 1, -1):
                    target_block: MambaBlock = model.backbone.layers[targeting_index]
                    intact_block_dict[targeting_index] = copy.deepcopy(target_block)

                attack_rate_task = progress.add_task(
                    "[cyan]Attack Rate=", total=len(selection_strategy.attack_rate_list)
                )
                for attack_rate in selection_strategy.gauge_attack_rate():
                    progress.update(attack_rate_task, advance=1)
                    ts.clear_memory()
                    iteration_index += 1
                    if (iteration_index % cuda_instances) != (
                        cuda_index % cuda_instances
                    ):
                        continue
                    effective_attack_rate = round(attack_rate * attack_rate_multiplier)
                    # Bravo-6 1 target the blocks here
                    targeted_weights = 0
                    for targeting_index in range(
                        block_index, min_targeting_index - 1, -1
                    ):
                        target_block: MambaBlock = model.backbone.layers[
                            targeting_index
                        ]
                        targeted_weights += attack_block(
                            model,
                            target_block,
                            gradients,
                            selection_strategy,
                            target_param_types,
                            effective_attack_rate,
                        )
                    ###### Bravo-6 2 get post attack accuracy
                    post_attack_accuracy = accuracy_func(model, tokenizer, device_name)[
                        0
                    ]["acc"]
                    post_attack_perplexity, post_attack_loss, _ = (
                        ts.calculate_perplexity_loss_and_length(
                            model,
                            tokenizer,
                            dataset,
                            size=perplexity_size,
                            device=model.device,
                        )
                    )

                    ###### Bravo-6 3 save the attack results
                    attack_results = ts.BlockRangeAttackResults(
                        min_targeting_index + 1,
                        targeting_offset + 1,
                        block_index + 1,
                        attack_rate,
                        targeted_weights,
                        post_attack_perplexity,
                        post_attack_loss,
                        original_perplexity,
                        original_loss,
                        post_attack_accuracy,
                        original_accuracy,
                    )
                    attack_results_list[attack_results.uuid()] = attack_results

                    ###### Bravo-6 4 Restore the block
                    for targeting_index in range(
                        block_index, min_targeting_index - 1, -1
                    ):
                        targeted_block: MambaBlock = model.backbone.layers[
                            targeting_index
                        ]
                        restore_block(
                            model,
                            targeted_block,
                            intact_block_dict[targeting_index],
                            target_param_types,
                        )

                progress.remove_task(attack_rate_task)
            progress.remove_task(targeting_task)
        progress.remove_task(block_task)
    ts.clear_memory()

    return attack_results_list
    return attack_results_list
