import numpy as np
import torch

def exclusionary_optimization(
    model, tokenizer, saved_state_dict, gradients, weight_indices, 
    get_attack_loss_fn, wiki_data, device, loss_threshold, loss_tol=1e-3, max_iter=100, verbose=True
):
    """
    Try to reduce the number of indices in weight_indices while maintaining the original loss.
    Exclusion can be a single index or up to half of all indices at once, chosen at random.
    Tracks progress for visualization.
    """
    reduced_indices = {}
    progress = {}  # Track progress for each layer

    for k, v in weight_indices.items():
        reduced_indices[k] = v['weight_indices'].clone().detach().cpu().numpy() if hasattr(v['weight_indices'], 'cpu') else np.array(v['weight_indices'])
        progress[k] = []  # List of (iteration, num_indices, loss)

    orig_loss, _ = get_attack_loss_fn(model, tokenizer, saved_state_dict, gradients, weight_indices, device, wiki_data)
    if verbose:
        print(f"Original loss: {orig_loss:.6f} with {len(reduced_indices[list(reduced_indices.keys())[0]])} indices")

    rng = np.random.default_rng()
    for layer in reduced_indices:
        indices = reduced_indices[layer]
        keep = np.ones(len(indices), dtype=bool)
        improved = True
        iter_count = 0
        # Track initial state
        progress[layer].append((iter_count, len(indices), orig_loss))
        while improved and iter_count < max_iter and np.sum(keep) > 1:
            improved = False
            iter_count += 1
            n = len(indices)
            tries_per_iter = 100  # Try 10 random exclusions per iteration
            for _ in range(tries_per_iter):
                if np.sum(keep) <= 1:
                    break
                num_exclude = rng.integers(1, max(2, n // 2 + 1))
                exclude_candidates = rng.choice(np.where(keep)[0], size=num_exclude, replace=False)
                test_keep = keep.copy()
                test_keep[exclude_candidates] = False
                test_indices = indices[test_keep]
                test_weight_indices = {
                    layer: {
                        'weight_indices': test_indices,
                        'weight_values': None,
                        'gradient_values': None,
                        'importance_values': None
                    }
                }
                test_loss, _ = get_attack_loss_fn(model, tokenizer, saved_state_dict, gradients, weight_indices, device, wiki_data)
                if test_loss >= loss_threshold:#orig_loss:
                    keep[exclude_candidates] = False
                    improved = True
                    if verbose:
                        print(f"Excluding {num_exclude} indices: loss={test_loss:.6f} (kept)")
                    break  # Only apply one successful exclusion per iteration
            indices = indices[keep]
            keep = np.ones(len(indices), dtype=bool)  # Reset keep to match new indices length
            # Track progress
            progress[layer].append((iter_count, len(indices), test_loss if improved else orig_loss))
        reduced_indices[layer] = indices
        if verbose:
            print(f"Reduced indices for {layer}: {len(indices)}")
    # Output for optimization and progress
    output = {}
    for layer in reduced_indices:
        output[layer] = {
            'weight_indices': reduced_indices[layer],
            'weight_values': None,
            'gradient_values': None,
            'importance_values': None,
            'progress': progress[layer]  # Add progress tracking
        }
    return output
